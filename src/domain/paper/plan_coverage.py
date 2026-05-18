"""Plan coverage diagnostics for search / analysis / writing alignment.

The report is deterministic and intentionally lightweight: it explains which
plan tasks have visible support in filter results, analysis text, written
sections, and evidence metadata without calling an LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agents.planner.models import ResearchPlan


TaskKind = Literal["search", "analysis", "writing"]


class PlanTaskCoverage(BaseModel):
    task_type: TaskKind
    task_index: int
    task_name: str
    covered: bool = False
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    representative_papers: list[dict[str, Any]] = Field(default_factory=list)
    evidence_count: int = 0
    notes: list[str] = Field(default_factory=list)


class PlanCoverageReport(BaseModel):
    search_tasks: list[PlanTaskCoverage] = Field(default_factory=list)
    analysis_tasks: list[PlanTaskCoverage] = Field(default_factory=list)
    writing_tasks: list[PlanTaskCoverage] = Field(default_factory=list)
    search_task_coverage_ratio: float = 0.0
    analysis_task_coverage_ratio: float = 0.0
    writing_task_coverage_ratio: float = 0.0
    missing_plan_dimensions: dict[str, list[str]] = Field(default_factory=dict)
    representative_papers: list[dict[str, Any]] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Plan Coverage Summary",
            "",
            f"- search_task_coverage_ratio: {self.search_task_coverage_ratio:.2f}",
            f"- analysis_task_coverage_ratio: {self.analysis_task_coverage_ratio:.2f}",
            f"- writing_task_coverage_ratio: {self.writing_task_coverage_ratio:.2f}",
            "",
        ]
        for title, rows in (
            ("Search Tasks", self.search_tasks),
            ("Analysis Tasks", self.analysis_tasks),
            ("Writing Tasks", self.writing_tasks),
        ):
            lines.append(f"## {title}")
            if not rows:
                lines.append("")
                lines.append("- not_available")
                lines.append("")
                continue
            for row in rows:
                status = "covered" if row.covered else "missing"
                reps = ", ".join(
                    str(p.get("paper_id") or p.get("title") or "") for p in row.representative_papers[:3]
                )
                lines.append(
                    f"- [{status}] {row.task_index}: {row.task_name} "
                    f"(score={row.coverage_score:.2f}, evidence={row.evidence_count}"
                    f"{', reps=' + reps if reps else ''})"
                )
            lines.append("")
        return "\n".join(lines).strip() + "\n"


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "agent",
    "agents",
    "agentic",
    "analysis",
    "benchmark",
    "benchmarks",
    "comparison",
    "evaluation",
    "framework",
    "large",
    "language",
    "llm",
    "llms",
    "model",
    "models",
    "paper",
    "papers",
    "research",
    "section",
    "system",
    "systems",
    "task",
    "tasks",
    "with",
}


def _tokens(text: str) -> set[str]:
    out = {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}
    return {t for t in out if t not in _STOPWORDS}


def _coerce_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _paper_meta(row: Any) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        d = row.model_dump(mode="json")
    elif isinstance(row, dict):
        d = row
    else:
        return {}
    return {
        "paper_id": d.get("paper_id") or d.get("id") or d.get("arxiv_id"),
        "title": d.get("title"),
        "url": d.get("url") or d.get("pdf_url"),
        "published": d.get("published") or d.get("published_date"),
    }


def _coverage_ratio(rows: list[PlanTaskCoverage]) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r.covered) / len(rows), 4)


def _evidence_items(ledger: Any) -> list[dict[str, Any]]:
    if ledger is None:
        return []
    if hasattr(ledger, "items"):
        return [
            it.model_dump(mode="json") if hasattr(it, "model_dump") else dict(it)
            for it in (getattr(ledger, "items", []) or [])
            if isinstance(it, dict) or hasattr(it, "model_dump")
        ]
    if isinstance(ledger, dict) and isinstance(ledger.get("items"), list):
        return [x for x in ledger["items"] if isinstance(x, dict)]
    return []


def _representatives_from_evidence(items: list[dict[str, Any]], task_tokens: set[str], limit: int = 5) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        blob = " ".join(str(item.get(k) or "") for k in ("title", "claim", "supports_argument", "section_type"))
        hits = task_tokens & _tokens(blob)
        if hits:
            scored.append((len(hits), _paper_meta(item)))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, meta in scored:
        key = str(meta.get("paper_id") or meta.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(meta)
        if len(out) >= limit:
            break
    return out


def _build_search_coverage(plan: ResearchPlan, filter_report: dict[str, Any]) -> list[PlanTaskCoverage]:
    summaries = filter_report.get("task_coverage_summary")
    by_idx: dict[int, dict[str, Any]] = {}
    if isinstance(summaries, list):
        for row in summaries:
            if isinstance(row, dict):
                try:
                    by_idx[int(row.get("task_index", -1))] = row
                except (TypeError, ValueError):
                    continue

    rows: list[PlanTaskCoverage] = []
    for i, task in enumerate(plan.search_tasks or []):
        row = by_idx.get(i, {})
        reps = row.get("covered_papers") if isinstance(row.get("covered_papers"), list) else []
        count = int(row.get("covered_paper_count") or len(reps or []))
        rows.append(
            PlanTaskCoverage(
                task_type="search",
                task_index=i,
                task_name=task.query,
                covered=count > 0,
                coverage_score=1.0 if count > 0 else 0.0,
                matched_terms=[str(x) for x in (row.get("core_terms") or [])[:12]],
                representative_papers=[_paper_meta(x) for x in reps[:5] if isinstance(x, dict)],
                evidence_count=count,
                notes=[] if count > 0 else ["no_filtered_paper_matched_task"],
            )
        )
    return rows


def _build_analysis_coverage(
    plan: ResearchPlan,
    analyse_results: Any,
    evidence_items: list[dict[str, Any]],
) -> list[PlanTaskCoverage]:
    raw = analyse_results if isinstance(analyse_results, str) else json.dumps(analyse_results or {}, ensure_ascii=False)
    text_tokens = _tokens(raw)
    rows: list[PlanTaskCoverage] = []
    for i, task in enumerate(plan.analysis_tasks or []):
        task_text = " ".join(
            [
                task.name,
                task.analysis_type,
                task.description or "",
                " ".join(task.target_dimensions or []),
            ]
        )
        task_tokens = _tokens(task_text)
        hits = sorted(task_tokens & text_tokens)
        score = min(1.0, len(hits) / max(2, min(len(task_tokens), 8))) if task_tokens else 0.0
        reps = _representatives_from_evidence(evidence_items, task_tokens)
        covered = score >= 0.25 or bool(reps and hits)
        rows.append(
            PlanTaskCoverage(
                task_type="analysis",
                task_index=i,
                task_name=task.name,
                covered=covered,
                coverage_score=round(score, 4),
                matched_terms=hits[:12],
                representative_papers=reps,
                evidence_count=len(reps),
                notes=[] if covered else ["analysis_text_does_not_cover_task_terms"],
            )
        )
    return rows


def _build_writing_coverage(
    plan: ResearchPlan,
    written_sections: list[Any] | None,
    evidence_items: list[dict[str, Any]],
) -> list[PlanTaskCoverage]:
    sections = [str(x.get("content") if isinstance(x, dict) else x or "") for x in (written_sections or [])]
    all_text_tokens = _tokens("\n".join(sections))
    rows: list[PlanTaskCoverage] = []
    for i, task in enumerate(plan.writing_tasks or []):
        section_text = sections[i] if i < len(sections) else ""
        task_text = " ".join(
            [
                task.section_title,
                task.section_goal,
                " ".join(task.required_evidence_types or []),
            ]
        )
        task_tokens = _tokens(task_text)
        section_tokens = _tokens(section_text)
        hits = sorted((task_tokens & section_tokens) or (task_tokens & all_text_tokens))
        index_covered = bool(section_text.strip())
        score = 0.5 if index_covered else 0.0
        if task_tokens:
            score += 0.5 * min(1.0, len(hits) / max(2, min(len(task_tokens), 8)))
        reps = _representatives_from_evidence(evidence_items, task_tokens)
        rows.append(
            PlanTaskCoverage(
                task_type="writing",
                task_index=i,
                task_name=task.section_title,
                covered=score >= 0.5,
                coverage_score=round(min(1.0, score), 4),
                matched_terms=hits[:12],
                representative_papers=reps,
                evidence_count=len(reps),
                notes=[] if index_covered else ["missing_written_section_at_plan_index"],
            )
        )
    return rows


def build_plan_coverage_report(
    *,
    plan: ResearchPlan | None,
    filter_report: Any = None,
    analyse_results: Any = None,
    written_sections: list[Any] | None = None,
    evidence_ledger: Any = None,
) -> PlanCoverageReport:
    if plan is None:
        return PlanCoverageReport(
            missing_plan_dimensions={
                "search_tasks": ["no_plan"],
                "analysis_tasks": ["no_plan"],
                "writing_tasks": ["no_plan"],
            }
        )

    fr = _coerce_dict(filter_report)
    items = _evidence_items(evidence_ledger)
    search_rows = _build_search_coverage(plan, fr)
    analysis_rows = _build_analysis_coverage(plan, analyse_results, items)
    writing_rows = _build_writing_coverage(plan, written_sections, items)

    missing = {
        "search_tasks": [r.task_name for r in search_rows if not r.covered],
        "analysis_tasks": [r.task_name for r in analysis_rows if not r.covered],
        "writing_tasks": [r.task_name for r in writing_rows if not r.covered],
    }
    reps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*search_rows, *analysis_rows, *writing_rows]:
        for meta in row.representative_papers:
            key = str(meta.get("paper_id") or meta.get("title") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            reps.append(meta)
            if len(reps) >= 10:
                break
        if len(reps) >= 10:
            break

    return PlanCoverageReport(
        search_tasks=search_rows,
        analysis_tasks=analysis_rows,
        writing_tasks=writing_rows,
        search_task_coverage_ratio=_coverage_ratio(search_rows),
        analysis_task_coverage_ratio=_coverage_ratio(analysis_rows),
        writing_task_coverage_ratio=_coverage_ratio(writing_rows),
        missing_plan_dimensions=missing,
        representative_papers=reps,
    )
