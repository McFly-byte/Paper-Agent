"""单次 run 审计包导出：脱敏、截断、结合 tmp state store。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.core.config import config
from src.core.state_models import BackToFrontData, ExecutionState, PaperAgentState
from src.observability.redaction import redact_mapping, redact_value
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 与 run_tmp_state_store 保持一致（避免模块顶层 import knowledge → chromadb）
KEY_SEARCH_RESULTS = "workflow_search_results"
KEY_FILTERED_PAPERS = "workflow_filtered_papers"
KEY_EXTRACTED_DATA = "workflow_extracted_data"
KEY_READING_SUCCESSFUL_PAPERS = "workflow_reading_successful_papers"
KEY_EVIDENCE_LEDGER = "workflow_evidence_ledger"
KEY_CITATION_MAP = "workflow_citation_map"
KEY_FAITHFULNESS_REVIEW = "workflow_faithfulness_review"
KEY_REPORT_CITATION_VALIDATION = "workflow_report_citation_validation"
KEY_EVIDENCE_BOUND_SECTIONS = "workflow_evidence_bound_sections"
KEY_ANALYSE_RESULTS = "workflow_analyse_results"
KEY_WRITTED_SECTIONS = "workflow_writted_sections"


async def _default_get_json(state: PaperAgentState, blob_key: str) -> Any | None:
    from src.services.run_tmp_state_store import get_json as _g

    return await _g(state, blob_key)


async def _default_get_text(state: PaperAgentState, blob_key: str) -> str | None:
    from src.services.run_tmp_state_store import get_text as _t

    return await _t(state, blob_key)


# 测试可 monkeypatch 这两个符号，避免拉起重依赖
get_json = _default_get_json
get_text = _default_get_text

TMP_KEYS: list[tuple[str, str]] = [
    ("search_results", KEY_SEARCH_RESULTS),
    ("filtered_papers", KEY_FILTERED_PAPERS),
    ("reading_successful_papers", KEY_READING_SUCCESSFUL_PAPERS),
    ("extracted_data", KEY_EXTRACTED_DATA),
    ("evidence_ledger", KEY_EVIDENCE_LEDGER),
    ("analyse_results_text", KEY_ANALYSE_RESULTS),
    ("writted_sections", KEY_WRITTED_SECTIONS),
    ("citation_map", KEY_CITATION_MAP),
    ("evidence_bound_sections", KEY_EVIDENCE_BOUND_SECTIONS),
    ("faithfulness_review", KEY_FAITHFULNESS_REVIEW),
    ("report_citation_validation", KEY_REPORT_CITATION_VALIDATION),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_base_dir(raw: str | None) -> Path:
    p = Path(raw or "artifacts/run_audits")
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _na(reason: str = "") -> dict[str, Any]:
    return {"status": "not_available", "reason": reason or "missing"}


def _dump_json(path: Path, obj: Any, *, redact: bool, max_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = redact_mapping(obj, max_text_chars=max_chars) if redact else obj
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _dump_text(path: Path, text: str, *, redact: bool, max_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = text or ""
    if redact:
        s = str(redact_value(s, max_text_chars=max_chars))
    elif len(s) > max_chars:
        s = s[:max_chars] + "\n...[truncated]"
    path.write_text(s, encoding="utf-8")


def _dump_jsonl(path: Path, lines: list[dict[str, Any]], *, redact: bool, max_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for row in lines:
        row2 = redact_mapping(row, max_text_chars=max_chars) if redact else row
        parts.append(json.dumps(row2, ensure_ascii=False))
    path.write_text("\n".join(parts) + ("\n" if parts else ""), encoding="utf-8")


def _search_results_summary(results: list[Any] | None, max_papers: int, max_chars: int, redact: bool) -> Any:
    if not results:
        return _na("no_search_results")
    out: list[dict[str, Any]] = []
    for i, p in enumerate(results[:max_papers]):
        if not isinstance(p, dict):
            continue
        reasons = p.get("reasons") or p.get("filter_reasons")
        if isinstance(reasons, list):
            reasons = [str(x)[:200] for x in reasons[:5]]
        elif reasons is not None:
            reasons = str(reasons)[:400]
        item = {
            "title": str(p.get("title") or "")[:400],
            "paper_id": str(p.get("paper_id") or p.get("id") or "")[:200],
            "published": p.get("published") or p.get("published_date"),
            "url": p.get("url") or p.get("link"),
            "score": p.get("score") or p.get("relevance_score"),
            "reasons": reasons,
        }
        out.append(item)
    obj: dict[str, Any] = {"count": len(results), "sample": out}
    return redact_mapping(obj, max_text_chars=max_chars) if redact else obj


def _papers_summary(cands: list[Any] | None, max_papers: int, max_chars: int, redact: bool) -> Any:
    if not cands:
        return _na("no_papers")
    rows: list[dict[str, Any]] = []
    for c in cands[:max_papers]:
        if hasattr(c, "model_dump"):
            d = c.model_dump(mode="json")
        elif isinstance(c, dict):
            d = dict(c)
        else:
            d = {"repr": str(c)[:300]}
        rows.append(
            {
                "paper_id": str(d.get("paper_id", ""))[:120],
                "title": str(d.get("title", ""))[:300],
                "published": d.get("published"),
                "url": d.get("url"),
                "source": d.get("source"),
                "abstract_excerpt": str(d.get("abstract") or "")[:400],
            }
        )
    obj = {"count": len(cands), "sample": rows}
    return redact_mapping(obj, max_text_chars=max_chars) if redact else obj


def _evidence_sample(
    ledger_obj: Any,
    max_items: int,
    max_chars: int,
    redact: bool,
    include_claims: bool,
) -> Any:
    if ledger_obj is None:
        return _na("no_evidence_ledger")
    items: list[Any] = []
    if isinstance(ledger_obj, dict):
        items = ledger_obj.get("items") or []
    elif hasattr(ledger_obj, "model_dump"):
        items = getattr(ledger_obj, "items", []) or []
    if not items:
        return _na("empty_evidence_items")
    slim: list[dict[str, Any]] = []
    for it in items[:max_items]:
        if hasattr(it, "model_dump"):
            d = it.model_dump(mode="json")
        elif isinstance(it, dict):
            d = dict(it)
        else:
            continue
        row = {
            "evidence_id": d.get("evidence_id"),
            "paper_id": d.get("paper_id"),
            "title": str(d.get("title") or "")[:300],
            "section_type": d.get("section_type"),
            "confidence": d.get("confidence"),
        }
        if include_claims:
            row["claim"] = str(d.get("claim") or "")[: max_chars // 2]
            ot = d.get("original_text")
            if ot:
                row["original_text"] = str(ot)[: max_chars // 4]
        slim.append(row)
    obj = {"total": len(items), "exported": len(slim), "items": slim}
    return redact_mapping(obj, max_text_chars=max_chars) if redact else obj


def _evidence_ledger_summary(ledger_obj: Any, max_chars: int, redact: bool) -> Any:
    if ledger_obj is None:
        return _na("no_evidence_ledger")
    if hasattr(ledger_obj, "summary"):
        try:
            summ = ledger_obj.summary()
        except Exception:  # noqa: BLE001
            summ = {"error": "summary_failed"}
    elif isinstance(ledger_obj, dict) and isinstance(ledger_obj.get("items"), list):
        n = len(ledger_obj["items"])
        summ = {"total_items": n}
    else:
        summ = _na("unknown_ledger_shape")
    if isinstance(summ, dict) and redact:
        return redact_mapping(summ, max_text_chars=max_chars)
    return summ


def _citation_ref_count(cm: Any) -> int:
    if cm is None:
        return 0
    if isinstance(cm, dict):
        refs = cm.get("refs")
        if isinstance(refs, list):
            return len(refs)
    if hasattr(cm, "refs"):
        try:
            return len(getattr(cm, "refs", []) or [])
        except Exception:  # noqa: BLE001
            return 0
    return 0


def _written_sections_md(
    sections: list[Any] | None,
    *,
    include_body: bool,
    max_chars: int,
    redact: bool,
) -> str:
    if not sections:
        return "# not_available\n\n无章节内容。\n"
    lines: list[str] = []
    for i, sec in enumerate(sections):
        body = sec if isinstance(sec, str) else str(sec)
        title = f"## Section {i + 1}"
        if include_body:
            chunk = body[:max_chars] if len(body) > max_chars else body
            if redact:
                chunk = str(redact_value(chunk, max_text_chars=max_chars))
            lines.append(f"{title}\n\n{chunk}\n")
        else:
            lines.append(f"{title}\n\n_length_chars={len(body)}\n")
    return "\n".join(lines)


def _extracted_data_compact(extracted: Any, max_papers: int, max_chars: int, redact: bool) -> Any:
    if extracted is None:
        return _na("no_extracted_data")
    papers_raw: list[Any] = []
    if hasattr(extracted, "papers"):
        papers_raw = list(extracted.papers or [])
    elif isinstance(extracted, dict) and isinstance(extracted.get("papers"), list):
        papers_raw = extracted["papers"]
    if not papers_raw:
        return _na("empty_extracted_papers")
    rows: list[dict[str, Any]] = []
    for p in papers_raw[:max_papers]:
        if hasattr(p, "model_dump"):
            d = p.model_dump(mode="json")
        elif isinstance(p, dict):
            d = dict(p)
        else:
            continue
        km = d.get("key_methodology") or {}
        if isinstance(km, dict):
            km = {"name": str(km.get("name", ""))[:120], "novelty_excerpt": str(km.get("novelty", ""))[:200]}
        rows.append(
            {
                "paper_id": str(d.get("paper_id", ""))[:120],
                "core_problem_excerpt": str(d.get("core_problem", ""))[:max_chars],
                "key_methodology": km,
                "contributions_count": len(d.get("contributions") or []) if isinstance(d.get("contributions"), list) else 0,
                "main_results_excerpt": str(d.get("main_results", ""))[:400],
            }
        )
    obj = {"total_papers": len(papers_raw), "sample": rows}
    return redact_mapping(obj, max_text_chars=max_chars) if redact else obj


def _minimal_timeline(state: PaperAgentState) -> list[dict[str, Any]]:
    evs = state.trace_events or []
    if evs:
        return list(evs)
    return [
        {
            "synthetic": True,
            "note": "trace_events_empty",
            "run_id": state.run_id,
            "current_step": str(state.current_step),
            "user_request_excerpt": (state.user_request or "")[:240],
        }
    ]


def _faithfulness_verdict(fr: Any) -> str | None:
    if fr is None:
        return None
    if isinstance(fr, dict):
        return fr.get("verdict")
    return None


def _report_citation_verdict(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("verdict")
    return None


class RunAuditManifest(BaseModel):
    run_id: str
    user_request: str
    created_at: str
    workflow_version: str = "v2"
    artifacts: list[str]
    summary: dict[str, Any]


class RunAuditExporter:
    """将 PaperAgentState + tmp store 导出为可审计目录。"""

    def __init__(
        self,
        base_dir: str,
        *,
        max_text_chars: int = 4000,
        max_papers: int = 20,
        max_evidence_items: int = 120,
        include_full_report: bool = True,
        include_written_sections: bool = True,
        include_evidence_claims: bool = True,
        redact_secrets: bool = True,
    ) -> None:
        self.base_path = _resolve_base_dir(base_dir)
        self.max_text_chars = max_text_chars
        self.max_papers = max_papers
        self.max_evidence_items = max_evidence_items
        self.include_full_report = include_full_report
        self.include_written_sections = include_written_sections
        self.include_evidence_claims = include_evidence_claims
        self.redact_secrets = redact_secrets
        self.load_notes: list[str] = []

    async def _load_tmp(self, state: PaperAgentState) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for label, key in TMP_KEYS:
            try:
                if key == KEY_ANALYSE_RESULTS:
                    txt = await get_text(state, key)
                    out[label] = txt
                else:
                    out[label] = await get_json(state, key)
            except Exception as exc:  # noqa: BLE001
                msg = f"{label}:{type(exc).__name__}:{exc}"
                self.load_notes.append(msg)
                logger.warning("[run_audit] tmp read failed %s", msg[:500])
                out[label] = None
        return out

    def _state_summary(self, state: PaperAgentState, tmp: dict[str, Any]) -> dict[str, Any]:
        sr = state.search_results
        if not sr:
            sr = tmp.get("search_results")
        sr = sr if isinstance(sr, list) else []
        fp = state.filtered_papers or tmp.get("filtered_papers") or []
        if isinstance(fp, list):
            fpc = len(fp)
        else:
            fpc = 0
        ed = state.extracted_data
        if ed is None and tmp.get("extracted_data"):
            ed = tmp["extracted_data"]
        extracted_n = 0
        if ed is not None:
            if hasattr(ed, "papers"):
                extracted_n = len(ed.papers or [])
            elif isinstance(ed, dict) and isinstance(ed.get("papers"), list):
                extracted_n = len(ed["papers"])
        el = state.evidence_ledger or tmp.get("evidence_ledger")
        ev_n = 0
        if el is not None:
            if isinstance(el, dict):
                _items = el.get("items")
                ev_n = len(_items) if isinstance(_items, list) else 0
            else:
                seq = getattr(el, "items", None)
                ev_n = len(seq) if isinstance(seq, list) else 0
        cm = state.citation_map or tmp.get("citation_map")
        ws = state.writted_sections or tmp.get("writted_sections")
        if not isinstance(ws, list):
            ws = []
        report = state.report_markdown or ""
        fr = state.faithfulness_review or tmp.get("faithfulness_review")
        rcv = (state.config or {}).get("report_citation_validation") or tmp.get("report_citation_validation")
        return {
            "run_id": state.run_id,
            "current_step": str(state.current_step),
            "max_papers": state.max_papers,
            "has_plan": state.plan is not None,
            "search_result_count": len(sr),
            "filtered_paper_count": fpc,
            "extracted_paper_count": extracted_n,
            "evidence_item_count": ev_n,
            "citation_ref_count": _citation_ref_count(cm),
            "written_section_count": len(ws),
            "report_length": len(report or ""),
            "workflow_error_count": len(state.workflow_errors or []),
            "boundary_check_keys": list((state.boundary_checks or {}).keys()),
            "faithfulness_verdict": _faithfulness_verdict(fr),
            "report_citation_verdict": _report_citation_verdict(rcv),
        }

    def _build_audit_notes_md(
        self,
        state: PaperAgentState,
        summary: dict[str, Any],
        manifest_rel: list[str],
    ) -> str:
        warnings: list[str] = []
        if summary.get("workflow_error_count", 0):
            warnings.append(f"workflow_errors: {summary['workflow_error_count']}")
        if summary.get("faithfulness_verdict") == "fail":
            warnings.append("faithfulness_verdict=fail")
        if summary.get("report_citation_verdict") == "fail":
            warnings.append("report_citation_verdict=fail")
        if self.load_notes:
            warnings.append(f"tmp_store_read_issues: {len(self.load_notes)}")
        checks = [
            "research brief 与 user_request 对齐",
            "background_context 关键词与检索意图",
            "plan / plan_review 可执行性",
            "search_queries 与 search_results.summary",
            "paper_filter_report 是否误杀",
            "extracted_data 字段完整性",
            "evidence_ledger 规模与 claim 覆盖",
            "analysis_results 信息增益",
            "citation_map 与 written_sections 中 [Cn] 使用",
            "faithfulness_review 与 report_citation_validation",
        ]
        warn_lines = [f"- {w}" for w in warnings] if warnings else ["- (none)"]
        tmp_lines = [f"- {n}" for n in self.load_notes] if self.load_notes else ["- (all ok)"]
        lines = [
            "# Run audit notes",
            "",
            "## 产物清单",
            "",
            *[f"- `{p}`" for p in manifest_rel],
            "",
            "## 关键计数",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Warnings",
            "",
            *warn_lines,
            "",
            "## Tmp store 读取",
            "",
            *tmp_lines,
            "",
            "## 建议人工检查点",
            "",
            *[f"- {c}" for c in checks],
            "",
        ]
        return "\n".join(lines)

    async def export(self, state: PaperAgentState, *, run_dir: Path | None = None) -> Path:
        self.load_notes.clear()
        run_path = Path(run_dir) if run_dir is not None else self.base_path / state.run_id
        run_path.mkdir(parents=True, exist_ok=True)
        run_dir = run_path
        rd = self.redact_secrets
        mc = self.max_text_chars

        tmp = await self._load_tmp(state)

        search_results = state.search_results
        if not search_results:
            search_results = tmp.get("search_results")
        if not isinstance(search_results, list):
            search_results = []

        filtered = state.filtered_papers
        if not filtered:
            loaded_fp = tmp.get("filtered_papers")
            if isinstance(loaded_fp, list):
                filtered = loaded_fp

        candidates = state.paper_candidates or []

        extracted = state.extracted_data
        if extracted is None:
            extracted = tmp.get("extracted_data")

        ledger = state.evidence_ledger or tmp.get("evidence_ledger")

        analyse_text = state.analyse_results
        if not analyse_text:
            analyse_text = tmp.get("analyse_results_text") or ""

        sections = state.writted_sections or tmp.get("writted_sections")
        if not isinstance(sections, list):
            sections = []

        citation_map = state.citation_map or tmp.get("citation_map")
        ebs = tmp.get("evidence_bound_sections")
        if ebs is None and state.config:
            ebs = (state.config or {}).get("evidence_bound_sections")

        fr = state.faithfulness_review or tmp.get("faithfulness_review")
        rcv = (state.config or {}).get("report_citation_validation") or tmp.get("report_citation_validation")

        gate = (state.config or {}).get("search_gate_context") or {}
        search_queries_obj = {
            "querys": gate.get("querys"),
            "start_date": gate.get("start_date"),
            "end_date": gate.get("end_date"),
        }

        # --- timeline ---
        timeline = _minimal_timeline(state)
        _dump_jsonl(run_dir / "timeline.jsonl", timeline, redact=rd, max_chars=mc)

        # --- state / errors / boundary ---
        summ = self._state_summary(state, tmp)
        _dump_json(run_dir / "state_summary.json", summ, redact=rd, max_chars=mc)
        _dump_json(
            run_dir / "workflow_errors.json",
            {"items": state.workflow_errors or []},
            redact=rd,
            max_chars=mc,
        )
        _dump_json(run_dir / "boundary_checks.json", state.boundary_checks or {}, redact=rd, max_chars=mc)

        # coordinator
        if state.brief is not None:
            _dump_json(run_dir / "coordinator" / "brief.json", state.brief.model_dump(mode="json"), redact=rd, max_chars=mc)
        else:
            _dump_json(run_dir / "coordinator" / "brief.json", _na("no_brief"), redact=False, max_chars=mc)

        # background
        if state.background_context is not None:
            _dump_json(
                run_dir / "background" / "background_context.json",
                state.background_context.model_dump(mode="json"),
                redact=rd,
                max_chars=mc,
            )
        else:
            _dump_json(run_dir / "background" / "background_context.json", _na("no_background"), redact=False, max_chars=mc)

        # planner
        if state.plan is not None:
            _dump_json(run_dir / "planner" / "plan.json", state.plan.model_dump(mode="json"), redact=rd, max_chars=mc)
        else:
            _dump_json(run_dir / "planner" / "plan.json", _na("no_plan"), redact=False, max_chars=mc)
        if state.plan_review is not None:
            _dump_json(
                run_dir / "planner" / "plan_review.json",
                state.plan_review.model_dump(mode="json"),
                redact=rd,
                max_chars=mc,
            )
        else:
            _dump_json(run_dir / "planner" / "plan_review.json", _na("no_plan_review"), redact=False, max_chars=mc)

        # search
        _dump_json(
            run_dir / "search" / "search_results.summary.json",
            _search_results_summary(search_results, self.max_papers, mc, rd),
            redact=False,
            max_chars=mc,
        )
        _dump_json(
            run_dir / "search" / "search_queries.json",
            search_queries_obj if any(search_queries_obj.values()) else _na("no_search_gate_context"),
            redact=rd,
            max_chars=mc,
        )

        # paper_filter
        _dump_json(
            run_dir / "paper_filter" / "paper_candidates.summary.json",
            _papers_summary(candidates, self.max_papers, mc, rd),
            redact=False,
            max_chars=mc,
        )
        _dump_json(
            run_dir / "paper_filter" / "filtered_papers.summary.json",
            _papers_summary(filtered if isinstance(filtered, list) else [], self.max_papers, mc, rd),
            redact=False,
            max_chars=mc,
        )
        pfr = (state.config or {}).get("paper_filter_report")
        _dump_json(
            run_dir / "paper_filter" / "filter_report.json",
            pfr if pfr is not None else _na("no_paper_filter_report"),
            redact=rd,
            max_chars=mc,
        )

        # reading
        readings = state.paper_readings or []
        if not readings and tmp.get("reading_successful_papers"):
            readings = tmp["reading_successful_papers"]
            if not isinstance(readings, list):
                readings = []
        _dump_json(
            run_dir / "reading" / "paper_readings.summary.json",
            {"count": len(readings), "sample": readings[: self.max_papers]},
            redact=rd,
            max_chars=mc,
        )
        _dump_json(
            run_dir / "reading" / "extracted_data.summary.json",
            _extracted_data_compact(extracted, self.max_papers, mc, rd),
            redact=False,
            max_chars=mc,
        )

        # evidence
        _dump_json(
            run_dir / "evidence" / "evidence_ledger.summary.json",
            _evidence_ledger_summary(ledger, mc, rd),
            redact=False,
            max_chars=mc,
        )
        _dump_json(
            run_dir / "evidence" / "evidence_items.sample.json",
            _evidence_sample(
                ledger,
                self.max_evidence_items,
                mc,
                rd,
                self.include_evidence_claims,
            ),
            redact=False,
            max_chars=mc,
        )

        # analysis
        _dump_text(
            run_dir / "analysis" / "analysis_results.md",
            analyse_text if isinstance(analyse_text, str) and analyse_text.strip() else "# not_available\n",
            redact=rd,
            max_chars=mc * 4,
        )

        # writing
        outline = state.outline or ""
        _dump_text(run_dir / "writing" / "outline.md", outline or "# not_available\n", redact=rd, max_chars=mc * 2)
        _dump_text(
            run_dir / "writing" / "written_sections.md",
            _written_sections_md(
                sections,
                include_body=self.include_written_sections,
                max_chars=mc * 8,
                redact=rd,
            ),
            redact=False,
            max_chars=10**9,
        )
        cm_obj = citation_map if citation_map is not None else _na("no_citation_map")
        _dump_json(run_dir / "writing" / "citation_map.json", cm_obj, redact=rd, max_chars=mc)
        ebs_obj = ebs if ebs is not None else _na("no_evidence_bound_sections")
        _dump_json(run_dir / "writing" / "evidence_bound_sections.json", ebs_obj, redact=rd, max_chars=mc)

        # review
        _dump_json(
            run_dir / "review" / "faithfulness_review.json",
            fr if fr is not None else _na("no_faithfulness_review"),
            redact=rd,
            max_chars=mc,
        )
        _dump_json(
            run_dir / "review" / "report_citation_validation.json",
            rcv if rcv is not None else _na("no_report_citation_validation"),
            redact=rd,
            max_chars=mc,
        )

        # report
        if self.include_full_report and (state.report_markdown or "").strip():
            _dump_text(run_dir / "report" / "final_report.md", state.report_markdown or "", redact=rd, max_chars=mc * 20)
        else:
            _dump_text(
                run_dir / "report" / "final_report.md",
                "# not_available\n\n（run_audit_include_full_report=false 或无 report_markdown）\n",
                redact=rd,
                max_chars=mc,
            )

        rel_paths = [
            "manifest.json",
            "timeline.jsonl",
            "state_summary.json",
            "workflow_errors.json",
            "boundary_checks.json",
            "coordinator/brief.json",
            "background/background_context.json",
            "planner/plan.json",
            "planner/plan_review.json",
            "search/search_results.summary.json",
            "search/search_queries.json",
            "paper_filter/paper_candidates.summary.json",
            "paper_filter/filtered_papers.summary.json",
            "paper_filter/filter_report.json",
            "reading/extracted_data.summary.json",
            "reading/paper_readings.summary.json",
            "evidence/evidence_ledger.summary.json",
            "evidence/evidence_items.sample.json",
            "analysis/analysis_results.md",
            "writing/outline.md",
            "writing/written_sections.md",
            "writing/citation_map.json",
            "writing/evidence_bound_sections.json",
            "review/faithfulness_review.json",
            "review/report_citation_validation.json",
            "report/final_report.md",
            "audit_notes.md",
        ]
        notes = self._build_audit_notes_md(state, summ, rel_paths)
        _dump_text(run_dir / "audit_notes.md", notes, redact=rd, max_chars=mc * 4)

        manifest = RunAuditManifest(
            run_id=state.run_id,
            user_request=(state.user_request or "")[:2000],
            created_at=_utc_now_iso(),
            artifacts=rel_paths,
            summary=summ,
        )
        _dump_json(run_dir / "manifest.json", manifest.model_dump(mode="json"), redact=rd, max_chars=mc)

        logger.info("[run_audit] exported run_id=%s path=%s", state.run_id, run_dir)
        return run_dir


async def export_run_audit_from_state(state: PaperAgentState) -> Path:
    """按 system_params workflow_v2.* 从全局 config 读取参数并导出。"""
    base = str(config.get("workflow_v2.run_audit_dir", "artifacts/run_audits") or "artifacts/run_audits")
    exporter = RunAuditExporter(
        base,
        max_text_chars=config.get_int("workflow_v2.run_audit_max_text_chars", 4000),
        max_papers=config.get_int("workflow_v2.run_audit_max_papers", 20),
        max_evidence_items=config.get_int("workflow_v2.run_audit_max_evidence_items", 120),
        include_full_report=config.get_bool("workflow_v2.run_audit_include_full_report", True),
        include_written_sections=config.get_bool("workflow_v2.run_audit_include_written_sections", True),
        include_evidence_claims=config.get_bool("workflow_v2.run_audit_include_evidence_claims", True),
        redact_secrets=config.get_bool("workflow_v2.run_audit_redact_secrets", True),
    )
    return await exporter.export(state)


async def maybe_export_run_audit_after_run(
    state_queue: Any,
    state: PaperAgentState,
) -> None:
    """工作流结束后调用：受 enable_run_audit_dump 控制，失败不影响主流程。"""
    if not config.get_bool("workflow_v2.enable_run_audit_dump", False):
        return
    try:
        path = await export_run_audit_from_state(state)
        if state.config is None:
            state.config = {}
        state.config["run_audit_path"] = str(path)
        step = (
            ExecutionState.COMPLETED
            if state.current_step != ExecutionState.FAILED
            else ExecutionState.FAILED
        )
        await state_queue.put(
            BackToFrontData(
                step=step.value,
                state="audit_exported",
                data={"path": str(path), "run_id": state.run_id},
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[run_audit] export failed: %s", exc)
        from src.runtime.trace_utils import append_workflow_error

        append_workflow_error(
            state,
            {"node": "run_audit_export", "error": f"{type(exc).__name__}: {exc}"[:4000]},
        )
