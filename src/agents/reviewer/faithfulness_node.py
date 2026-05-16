"""faithfulness_review_node：规则为主的事实与引用一致性审查（默认不阻断）。"""

from __future__ import annotations

import json
import os
import re
import traceback

from langgraph.runtime import Runtime

from src.core.config import config
from src.core.workflow_recovery import set_recovery_target
from src.core.state_models import BackToFrontData, ExecutionState, NodeError, PaperAgentState, PaperRunContext, State
from src.domain.paper.citation import CitationMap
from src.domain.paper.faithfulness import (
    FaithfulnessReviewReport,
    SectionFaithfulnessResult,
    build_faithfulness_report,
    merge_llm_report,
)
from src.runtime.events import TraceEvent
from src.runtime.trace_utils import append_trace_event, append_workflow_error
from src.services.run_tmp_state_store import get_json, put_json, KEY_CITATION_MAP, KEY_EVIDENCE_BOUND_SECTIONS, KEY_FAITHFULNESS_REVIEW, KEY_WRITTED_SECTIONS
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

NODE = "faithfulness_review_node"


def _trace_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_TRACE_EVENTS")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_trace_events", True)


def _enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_FAITHFULNESS_REVIEW")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_faithfulness_review", True)


def _strict(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_STRICT_FAITHFULNESS_REVIEW")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_strict_faithfulness_review", False)


def _llm_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_LLM_FAITHFULNESS_REVIEW")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_llm_faithfulness_review", False)


async def _load_cmap(val: PaperAgentState) -> CitationMap | None:
    raw = val.citation_map if isinstance(val.citation_map, dict) else None
    if raw and raw.get("refs"):
        try:
            return CitationMap.model_validate(raw)
        except Exception:
            pass
    blob = await get_json(val, KEY_CITATION_MAP)
    if isinstance(blob, dict) and blob.get("refs"):
        try:
            return CitationMap.model_validate(blob)
        except Exception:
            return None
    return None


async def faithfulness_review_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    val: PaperAgentState = state["value"]
    q = runtime.context.state_queue
    run_id = val.run_id
    te = _trace_enabled(val)
    strict = _strict(val)

    append_trace_event(
        val,
        TraceEvent(
            run_id=run_id,
            event_type="FAITHFULNESS_REVIEW_STARTED",
            node_name=NODE,
            status="started",
        ),
        enabled=te,
    )
    try:
        await q.put(
            BackToFrontData(
                step=ExecutionState.WRITING,
                state="faithfulness_reviewing",
                data={"message": "正在进行证据忠实性审查…"},
            )
        )
    except Exception:
        pass

    if not _enabled(val):
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="FAITHFULNESS_REVIEW_FINISHED",
                node_name=NODE,
                status="skipped",
                output_summary="disabled",
            ),
            enabled=te,
        )
        return {"value": val}

    try:
        sections = list(val.writted_sections or [])
        if not sections:
            loaded = await get_json(val, KEY_WRITTED_SECTIONS)
            if isinstance(loaded, list):
                sections = [str(s) for s in loaded if s is not None]

        cmap = await _load_cmap(val)
        long_thr = config.get_int("workflow_v2.faithfulness_long_section_chars", 500)

        report = build_faithfulness_report(sections, cmap, long_section_chars=long_thr)

        if _llm_enabled(val) and cmap and report.section_results:
            try:
                from src.core.llm_infra.invoke import chat_completion_text_routed

                llm_secs: list[SectionFaithfulnessResult] = []
                ev_blob = await get_json(val, KEY_EVIDENCE_BOUND_SECTIONS)
                ev_txt = json.dumps(ev_blob, ensure_ascii=False)[:6000] if ev_blob else ""
                cmap_txt = json.dumps(cmap.to_jsonable(), ensure_ascii=False)[:4000]
                for i, body in enumerate(sections):
                    prompt = (
                        "你是忠实度审查助手。只判断证据支持情况，不要编造事实。\n"
                        "输出单个 JSON 对象，字段：verdict, unsupported_claims, citation_issues, revision_instruction。\n"
                        f"Evidence context 摘要:\n{ev_txt}\n\nCitationMap:\n{cmap_txt}\n\n章节正文:\n{body[:8000]}\n"
                    )
                    raw_txt = await chat_completion_text_routed(
                        "search-model",
                        prompt,
                        node_name="faithfulness_review",
                        temperature=0.1,
                        source="faithfulness_review",
                    )
                    m = re.search(r"\{[\s\S]*\}", raw_txt)
                    if m:
                        data = json.loads(m.group(0))
                        if "section_title" not in data:
                            data["section_title"] = report.section_results[i].section_title
                        llm_secs.append(SectionFaithfulnessResult.model_validate(data))
                    else:
                        llm_secs.append(report.section_results[i])
                report = merge_llm_report(report, llm_secs)
            except Exception as llm_exc:  # noqa: BLE001
                logger.warning("LLM faithfulness 跳过: %s", llm_exc)

        val.faithfulness_review = report.to_jsonable()  # type: ignore[assignment]
        val.config = dict(val.config or {})
        val.config["faithfulness_summary"] = report.summary

        await put_json(val, KEY_FAITHFULNESS_REVIEW, report.to_jsonable())

        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="FAITHFULNESS_REVIEW_FINISHED",
                node_name=NODE,
                status="finished",
                output_summary=report.summary,
                metrics={"verdict": report.verdict},
            ),
            enabled=te,
        )
        try:
            await q.put(
                BackToFrontData(
                    step=ExecutionState.WRITING,
                    state="faithfulness_reviewed",
                    data={
                        "verdict": report.verdict,
                        "unsupported_claims": report.total_unsupported_claims,
                        "citation_issues": report.total_citation_issues,
                    },
                )
            )
        except Exception:
            pass

        if strict and report.verdict == "fail":
            if val.error is None:
                val.error = NodeError()
            val.error.faithfulness_review_node_error = report.summary[:4000]
            set_recovery_target(val, "faithfulness")

    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("faithfulness_review_node: %s", msg)
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="FAITHFULNESS_REVIEW_FAILED",
                node_name=NODE,
                status="failed",
                error=msg[:2000],
            ),
            enabled=te,
        )
        append_workflow_error(
            val,
            {"node": NODE, "error": msg, "traceback": traceback.format_exc()[:4000]},
        )
        try:
            await q.put(
                BackToFrontData(
                    step=ExecutionState.WRITING,
                    state="faithfulness_warning",
                    data={"message": "忠实度审查异常，已跳过", "detail": msg[:400]},
                )
            )
        except Exception:
            pass
        if strict:
            if val.error is None:
                val.error = NodeError()
            val.error.faithfulness_review_node_error = msg[:4000]
            set_recovery_target(val, "faithfulness")

    return {"value": val}
