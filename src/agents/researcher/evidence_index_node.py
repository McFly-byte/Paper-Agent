"""evidence_index_node：将结构化阅读结果整理为 EvidenceLedger（无伪造页码）。"""

from __future__ import annotations

import os
import traceback
from typing import Any

from langgraph.runtime import Runtime

from src.core.config import config
from src.core.workflow_recovery import set_recovery_target
from src.core.state_models import NodeError
from src.core.state_models import BackToFrontData, ExecutionState, PaperAgentState, PaperRunContext, State
from src.domain.paper.evidence import EvidenceLedger
from src.domain.paper.evidence_indexer import build_evidence_ledger, extracted_papers_to_reading_snapshots
from src.domain.paper.models import PaperCandidate
from src.runtime.events import TraceEvent
from src.runtime.trace_utils import append_trace_event, append_workflow_error
from src.services.run_tmp_state_store import ensure_run_tmp_kb, get_json, put_json, KEY_EVIDENCE_LEDGER, KEY_EXTRACTED_DATA, KEY_FILTERED_PAPERS, KEY_READING_SUCCESSFUL_PAPERS, KEY_SEARCH_RESULTS
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

NODE = "evidence_index_node"


def _trace_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_TRACE_EVENTS")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_trace_events", True)


def _evidence_index_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_EVIDENCE_INDEX")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_evidence_index", True)


def _strict_evidence_index(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_STRICT_EVIDENCE_INDEX")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_strict_evidence_index", False)


    if val.filtered_papers:
        out: list[dict[str, Any]] = []
        for p in val.filtered_papers:
            if isinstance(p, PaperCandidate):
                d = p.model_dump(mode="json")
                out.append(d)
            elif isinstance(p, dict):
                out.append(dict(p))
        return out
    return []


async def evidence_index_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    val: PaperAgentState = state["value"]
    q = runtime.context.state_queue
    run_id = val.run_id
    te = _trace_enabled(val)
    strict = _strict_evidence_index(val)
    max_ctx = config.get_int("workflow_v2.evidence_max_items_to_context", 64)

    try:
        await ensure_run_tmp_kb(val)
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence_index ensure_run_tmp_kb: %s", exc)

    append_trace_event(
        val,
        TraceEvent(
            run_id=run_id,
            event_type="EVIDENCE_INDEX_STARTED",
            node_name=NODE,
            status="started",
        ),
        enabled=te,
    )
    try:
        await q.put(
            BackToFrontData(
                step=ExecutionState.READING,
                state="indexing_evidence",
                data={"message": "正在构建证据账本…"},
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence_index SSE 推送失败: %s", exc)

    if not _evidence_index_enabled(val):
        val.evidence_ledger = EvidenceLedger(items=[])
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="EVIDENCE_INDEX_FINISHED",
                node_name=NODE,
                status="skipped",
                output_summary="disabled",
            ),
            enabled=te,
        )
        try:
            await q.put(
                BackToFrontData(
                    step=ExecutionState.READING,
                    state="evidence_indexed",
                    data={"skipped": True},
                )
            )
        except Exception:
            pass
        return {"value": val}

    try:
        extracted: dict[str, Any] | None = None
        if val.extracted_data is not None:
            extracted = val.extracted_data.model_dump(mode="json")
        if not extracted:
            loaded = await get_json(val, KEY_EXTRACTED_DATA)
            if isinstance(loaded, dict):
                extracted = loaded

        papers_raw: list[dict[str, Any]] = []
        if isinstance(extracted, dict):
            pr = extracted.get("papers")
            if isinstance(pr, list):
                papers_raw = [p for p in pr if isinstance(p, dict)]

        if not papers_raw:
            append_workflow_error(
                val,
                {"node": NODE, "phase": "evidence_index", "warning": "no_extracted_data"},
            )
            val.evidence_ledger = EvidenceLedger(items=[])
            val.config = dict(val.config or {})
            val.config["evidence_summary"] = {}
            append_trace_event(
                val,
                TraceEvent(
                    run_id=run_id,
                    event_type="EVIDENCE_INDEX_FINISHED",
                    node_name=NODE,
                    status="finished",
                    output_summary="no_extracted_empty_ledger",
                ),
                enabled=te,
            )
            try:
                await q.put(
                    BackToFrontData(
                        step=ExecutionState.READING,
                        state="evidence_indexed",
                        data={"items": 0, "warning": "no_extracted_data"},
                    )
                )
            except Exception:
                pass
            return {"value": val}

        metas: list[dict[str, Any]] | None = None
        loaded_ok = await get_json(val, KEY_READING_SUCCESSFUL_PAPERS)
        if isinstance(loaded_ok, list) and loaded_ok:
            metas = [x for x in loaded_ok if isinstance(x, dict)]

        if not metas:
            metas = _paper_meta_list_from_filtered(val)

        if not metas:
            loaded_f = await get_json(val, KEY_FILTERED_PAPERS)
            if isinstance(loaded_f, list):
                metas = [x for x in loaded_f if isinstance(x, dict)]

        if not metas and val.search_results:
            metas = [x for x in (val.search_results or []) if isinstance(x, dict)]

        if not metas:
            loaded_s = await get_json(val, KEY_SEARCH_RESULTS)
            if isinstance(loaded_s, list):
                metas = [x for x in loaded_s if isinstance(x, dict)]

        metas = metas or []

        ledger = build_evidence_ledger(papers_raw, metas, run_id=run_id)
        val.evidence_ledger = ledger

        snaps = extracted_papers_to_reading_snapshots(papers_raw, metas)
        if snaps and sum(len(str(s)) for s in snaps) < 500_000:
            val.paper_readings = snaps

        summ = ledger.summary()
        val.config = dict(val.config or {})
        val.config["evidence_summary"] = summ
        val.config["citation_context"] = ledger.to_citation_context(max_items=max_ctx)

        await put_json(val, KEY_EVIDENCE_LEDGER, ledger.to_jsonable())

        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="EVIDENCE_INDEX_FINISHED",
                node_name=NODE,
                status="finished",
                output_summary=f"items={len(ledger.items)}",
                metrics={"summary": summ},
            ),
            enabled=te,
        )
        try:
            await q.put(
                BackToFrontData(
                    step=ExecutionState.READING,
                    state="evidence_indexed",
                    data={"items": len(ledger.items), "summary": summ},
                )
            )
        except Exception:
            pass

    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("evidence_index_node 失败: %s", msg)
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="EVIDENCE_INDEX_FAILED",
                node_name=NODE,
                status="failed",
                error=msg[:2000],
            ),
            enabled=te,
        )
        append_workflow_error(
            val,
            {"node": NODE, "phase": "evidence_index", "error": msg, "traceback": traceback.format_exc()[:8000]},
        )
        val.evidence_ledger = EvidenceLedger(items=[])
        val.config = dict(val.config or {})
        val.config["evidence_summary"] = {}
        try:
            await q.put(
                BackToFrontData(
                    step=ExecutionState.READING,
                    state="evidence_warning",
                    data={"message": "证据索引失败，已使用空账本继续", "detail": msg[:500]},
                )
            )
        except Exception:
            pass
        if strict:
            if val.error is None:
                val.error = NodeError()
            val.error.evidence_index_node_error = msg[:4000]
            set_recovery_target(val, "evidence_index")

    return {"value": val}
