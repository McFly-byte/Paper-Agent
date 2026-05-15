"""paper_filter_node：检索结果标准化、去重与规则过滤（无 LLM）。"""

from __future__ import annotations

import os
import traceback
from typing import Any

from langgraph.runtime import Runtime

from src.core.config import config
from src.core.workflow_recovery import set_recovery_target
from src.core.state_models import BackToFrontData, ExecutionState, NodeError, PaperAgentState, PaperRunContext, State
from src.domain.paper.filter import filter_candidates, normalize_paper_candidates
from src.domain.paper.models import PaperCandidate
from src.runtime.events import TraceEvent
from src.runtime.trace_utils import append_trace_event, append_workflow_error
from src.services.run_tmp_state_store import ensure_run_tmp_kb, get_json, put_json, KEY_FILTERED_PAPERS, KEY_SEARCH_RESULTS
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

NODE = "paper_filter_node"


def _trace_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_TRACE_EVENTS")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_trace_events", True)


def _paper_filter_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_PAPER_FILTER")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_paper_filter", True)


def _strict_paper_filter(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_STRICT_PAPER_FILTER")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_strict_paper_filter", False)


def _candidate_to_search_dict(c: PaperCandidate) -> dict[str, Any]:
    if c.raw and isinstance(c.raw, dict):
        d = dict(c.raw)
        d.setdefault("paper_id", c.paper_id)
        d.setdefault("title", c.title)
        if c.abstract is not None:
            d.setdefault("abstract", c.abstract)
        if c.authors:
            d.setdefault("authors", c.authors)
        if c.published:
            d.setdefault("published", c.published)
        if c.url:
            d.setdefault("url", c.url)
        if c.pdf_url:
            d.setdefault("pdf_url", c.pdf_url)
        if c.categories:
            d.setdefault("categories", c.categories)
        return d
    return c.model_dump(mode="json", exclude={"raw"})


async def paper_filter_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    val: PaperAgentState = state["value"]
    q = runtime.context.state_queue
    run_id = val.run_id
    te = _trace_enabled(val)

    append_trace_event(
        val,
        TraceEvent(
            run_id=run_id,
            event_type="PAPER_FILTER_STARTED",
            node_name=NODE,
            status="started",
            input_summary=f"search_results={len(val.search_results or [])}",
        ),
        enabled=te,
    )
    try:
        await q.put(
            BackToFrontData(
                step=ExecutionState.SEARCHING,
                state="filtering",
                data={"message": "正在过滤检索结果…"},
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("paper_filter SSE filtering 推送失败: %s", exc)

    backup_results = list(val.search_results or [])
    strict = _strict_paper_filter(val)

    try:
        await ensure_run_tmp_kb(val)
    except Exception as exc:  # noqa: BLE001
        logger.warning("paper_filter ensure_run_tmp_kb: %s", exc)

    try:
        if not _paper_filter_enabled(val):
            raw = list(val.search_results or [])
            if not raw:
                loaded = await get_json(val, KEY_SEARCH_RESULTS)
                if isinstance(loaded, list):
                    raw = loaded
            cands = normalize_paper_candidates(raw)
            val.paper_candidates = cands
            val.filtered_papers = list(cands)[: max(1, int(val.max_papers))]
            if not val.search_results and raw:
                val.search_results = raw
            append_trace_event(
                val,
                TraceEvent(
                    run_id=run_id,
                    event_type="PAPER_FILTER_FINISHED",
                    node_name=NODE,
                    status="skipped",
                    output_summary="disabled_passthrough",
                ),
                enabled=te,
            )
            try:
                await q.put(
                    BackToFrontData(
                        step=ExecutionState.SEARCHING,
                        state="filter_completed",
                        data={"skipped": True},
                    )
                )
            except Exception:
                pass
            return {"value": val}

        raw_results: list[dict[str, Any]] = []
        if val.search_results:
            for x in val.search_results:
                if isinstance(x, dict):
                    raw_results.append(x)
        if not raw_results:
            loaded = await get_json(val, KEY_SEARCH_RESULTS)
            if isinstance(loaded, list):
                raw_results = [x for x in loaded if isinstance(x, dict)]

        candidates = normalize_paper_candidates(raw_results)
        val.paper_candidates = candidates

        min_score = config.get_float("workflow_v2.paper_filter_min_score", 0.05)
        raw_min = os.environ.get("PAPER_AGENT_PAPER_FILTER_MIN_SCORE")
        if raw_min is not None and str(raw_min).strip() != "":
            try:
                min_score = float(raw_min)
            except ValueError:
                pass
        fb_topk = config.get_bool("workflow_v2.paper_filter_fallback_keep_top_k", True)
        if os.environ.get("PAPER_AGENT_PAPER_FILTER_FALLBACK_KEEP_TOP_K", "").strip().lower() in (
            "0",
            "false",
            "no",
        ):
            fb_topk = False
        max_reject = config.get_int("workflow_v2.paper_filter_max_reject_log", 20)

        brief_topic = val.brief.clarified_topic if val.brief else None
        filt = filter_candidates(
            candidates,
            val.plan,
            val.background_context,
            val.max_papers,
            val.config or {},
            user_request=val.user_request or "",
            brief_topic=brief_topic,
            min_score=min_score,
            fallback_keep_top_k=fb_topk,
            max_reject_log=max_reject,
        )

        selected = list(filt.selected)
        if not selected and candidates:
            k = max(1, min(int(val.max_papers), len(candidates)))
            selected = candidates[:k]
            filt.warnings.append("empty_selection_fallback_original_order")

        val.filtered_papers = selected
        out_dicts = [_candidate_to_search_dict(c) for c in selected]
        val.search_results = out_dicts

        val.config = dict(val.config or {})
        val.config["paper_filter_report"] = filt.model_dump(mode="json")

        await put_json(val, KEY_FILTERED_PAPERS, [c.model_dump(mode="json") for c in selected])
        await put_json(val, KEY_SEARCH_RESULTS, out_dicts)

        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="PAPER_FILTER_FINISHED",
                node_name=NODE,
                status="finished",
                output_summary=f"in={filt.input_count} out={len(selected)}",
                metrics={"warnings": filt.warnings[:5]},
            ),
            enabled=te,
        )
        try:
            await q.put(
                BackToFrontData(
                    step=ExecutionState.SEARCHING,
                    state="filter_completed",
                    data={
                        "input_count": filt.input_count,
                        "output_count": len(selected),
                        "warnings": filt.warnings,
                    },
                )
            )
        except Exception:
            pass

    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("paper_filter_node 失败: %s", msg)
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="PAPER_FILTER_FAILED",
                node_name=NODE,
                status="failed",
                error=msg[:2000],
            ),
            enabled=te,
        )
        append_workflow_error(
            val,
            {"node": NODE, "phase": "paper_filter", "error": msg, "traceback": traceback.format_exc()[:8000]},
        )
        val.search_results = backup_results
        val.filtered_papers = []
        val.paper_candidates = normalize_paper_candidates(backup_results)
        try:
            await q.put(
                BackToFrontData(
                    step=ExecutionState.SEARCHING,
                    state="filter_warning",
                    data={"message": "论文过滤失败，已回退到原始检索结果", "detail": msg[:500]},
                )
            )
        except Exception:
            pass
        if strict:
            if val.error is None:
                val.error = NodeError()
            val.error.paper_filter_node_error = msg[:4000]
            set_recovery_target(val, "paper_filter")

    return {"value": val}
