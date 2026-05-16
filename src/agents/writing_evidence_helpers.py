"""写作子图：从 EvidenceLedger 构建 CitationMap 与章节证据上下文（writing_node 调用）。"""

from __future__ import annotations

import os
import re
from typing import Any

from src.core.config import config
from src.core.state_models import PaperAgentState
from src.domain.paper.citation import CitationMap
from src.domain.paper.evidence import EvidenceLedger
from src.domain.paper.evidence_context import (
    build_citation_writing_instruction,
    build_global_evidence_context,
    build_section_evidence_context,
)
from src.domain.paper.models import PaperCandidate
from src.services.run_tmp_state_store import (
    KEY_CITATION_MAP,
    KEY_EVIDENCE_BOUND_SECTIONS,
    KEY_EVIDENCE_LEDGER,
    KEY_FILTERED_PAPERS,
    get_json,
    put_json,
)
from src.runtime.trace_utils import append_workflow_error
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _evidence_bound_enabled() -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_EVIDENCE_BOUND_WRITING")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_evidence_bound_writing", True)


def _year_from_published(published: str | None, explicit: Any = None) -> int | None:
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    if not published:
        return None
    m = re.search(r"(19|20)\d{2}", str(published))
    return int(m.group(0)) if m else None


def _row_to_paper_meta(row: Any) -> tuple[str, dict[str, Any]]:
    """从 PaperCandidate 或 dict 提取 paper_id 与可用于 CitationMap 的元数据。"""
    if isinstance(row, PaperCandidate):
        d = row.model_dump(mode="python")
        raw = d.get("raw") if isinstance(d.get("raw"), dict) else {}
        pid = str(d.get("paper_id") or "").strip()
        pub = d.get("published")
        meta: dict[str, Any] = {
            "title": d.get("title"),
            "authors": d.get("authors"),
            "year": _year_from_published(pub, d.get("year")),
            "url": d.get("url") or raw.get("url"),
            "pdf_url": d.get("pdf_url") or raw.get("pdf_url"),
            "source": d.get("source") or raw.get("source"),
            "source_label": raw.get("source_label") or d.get("source_label"),
            "raw": raw,
        }
        return pid, {k: v for k, v in meta.items() if v not in (None, "", [])}

    if isinstance(row, dict):
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
        pid = str(row.get("paper_id") or "").strip()
        pub = row.get("published")
        meta = {
            "title": row.get("title"),
            "authors": row.get("authors"),
            "year": _year_from_published(pub, row.get("year")),
            "url": row.get("url") or raw.get("url"),
            "pdf_url": row.get("pdf_url") or raw.get("pdf_url"),
            "source": row.get("source") or raw.get("source"),
            "source_label": row.get("source_label") or raw.get("source_label"),
            "raw": raw,
        }
        return pid, {k: v for k, v in meta.items() if v not in (None, "", [])}

    return "", {}


async def _paper_metadata_by_id_from_state(state: PaperAgentState) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in state.filtered_papers or []:
        pid, meta = _row_to_paper_meta(row)
        if pid and meta:
            out[pid] = {**out.get(pid, {}), **meta}
    for row in state.paper_candidates or []:
        pid, meta = _row_to_paper_meta(row)
        if pid and meta:
            out[pid] = {**out.get(pid, {}), **meta}
    try:
        loaded = await get_json(state, KEY_FILTERED_PAPERS)
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取 KEY_FILTERED_PAPERS 失败（忽略）: %s", exc)
        loaded = None
    if isinstance(loaded, list):
        for row in loaded:
            pid, meta = _row_to_paper_meta(row)
            if pid and meta:
                out[pid] = {**out.get(pid, {}), **meta}
    return out


async def _load_ledger(state: PaperAgentState) -> EvidenceLedger | None:
    led = state.evidence_ledger
    if led is not None and led.items:
        return led
    raw = await get_json(state, KEY_EVIDENCE_LEDGER)
    if isinstance(raw, dict) and raw.get("items"):
        try:
            return EvidenceLedger.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EvidenceLedger 反序列化失败: %s", exc)
    return None


async def prepare_evidence_bound_writing_bundle(state: PaperAgentState) -> dict[str, Any]:
    """返回可并入 WritingState 的字段；无证据或未启用时返回 {}。"""
    if not _evidence_bound_enabled():
        return {}

    ledger = await _load_ledger(state)
    if ledger is None or not ledger.items:
        append_workflow_error(
            state,
            {"node": "writing_node", "phase": "evidence_bound", "warning": "no_evidence_ledger_fallback_legacy_writing"},
        )
        return {}

    max_items = config.get_int("workflow_v2.evidence_context_max_items_per_section", 16)
    max_chars = config.get_int("workflow_v2.evidence_context_max_chars", 12000)

    meta_by_id = await _paper_metadata_by_id_from_state(state)
    if meta_by_id:
        cmap = CitationMap.from_evidence_ledger_with_metadata(ledger, meta_by_id)
    else:
        cmap = CitationMap.from_evidence_ledger(ledger)
    cmap_json = cmap.to_jsonable()
    state.citation_map = cmap_json  # type: ignore[assignment]
    try:
        await put_json(state, KEY_CITATION_MAP, cmap_json)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写入 KEY_CITATION_MAP 失败（继续写作）: %s", exc)

    sec_blocks: list[str] = []
    bound_payload: list[dict[str, Any]] = []
    global_fallback = ""

    plan = state.plan
    if plan and plan.writing_tasks:
        for wt in plan.writing_tasks:
            ctx = build_section_evidence_context(
                wt.section_title,
                wt.section_goal,
                ledger,
                cmap,
                max_items=max_items,
            )
            sec_blocks.append(ctx.context_text)
            bound_payload.append(
                {
                    "section_title": wt.section_title,
                    "section_goal": wt.section_goal,
                    "context_text": ctx.context_text,
                }
            )
    else:
        gctx = build_global_evidence_context(ledger, cmap, max_items=max_items + 8, max_chars=max_chars)
        global_fallback = gctx.context_text
        bound_payload.append(
            {"section_title": "(global)", "section_goal": None, "context_text": global_fallback}
        )

    total = sum(len(s) for s in sec_blocks)
    while sec_blocks and total > max_chars:
        removed = sec_blocks.pop()
        if bound_payload:
            bound_payload.pop()
        total -= len(removed)

    instr = build_citation_writing_instruction(cmap)

    try:
        await put_json(state, KEY_EVIDENCE_BOUND_SECTIONS, bound_payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写入 KEY_EVIDENCE_BOUND_SECTIONS 失败: %s", exc)

    state.config = dict(state.config or {})
    state.config["evidence_bound_sections_count"] = len(bound_payload)

    return {
        "citation_marker_instruction": instr,
        "evidence_bound_block": global_fallback,
        "section_evidence_blocks": sec_blocks,
    }
