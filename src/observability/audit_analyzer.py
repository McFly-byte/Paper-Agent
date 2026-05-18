"""Static checks for exported run audit directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = [
    "manifest.json",
    "audit_notes.md",
    "state_summary.json",
    "boundary_checks.json",
    "timeline.jsonl",
    "workflow_errors.json",
    "planner/plan.json",
    "paper_filter/filter_report.json",
    "reading/extracted_data.summary.json",
    "reading/paper_readings.summary.json",
    "evidence/evidence_ledger.summary.json",
    "writing/citation_map.json",
    "writing/written_sections.md",
    "review/faithfulness_review.json",
    "review/report_citation_validation.json",
    "report/final_report.md",
]

MAIN_NODE_NAMES = [
    "coordinator_node",
    "background_investigation_node",
    "planner_node",
    "plan_review_node",
    "search_node",
    "paper_filter_node",
    "reading_node",
    "evidence_index_node",
    "analyse_node",
    "writing_node",
    "faithfulness_review_node",
    "report_node",
]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _issue(code: str, message: str, *, severity: str = "warning", path: str | None = None) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "path": path}


def _all_scores_match_all_tasks(filter_report: dict[str, Any]) -> bool:
    scores = filter_report.get("scores")
    if not isinstance(scores, list) or not scores:
        return False
    task_summary = filter_report.get("task_coverage_summary")
    task_count = len(task_summary) if isinstance(task_summary, list) and task_summary else 0
    if task_count <= 1:
        cov = filter_report.get("plan_task_coverage")
        if isinstance(cov, dict):
            task_count = len(cov)
    if task_count <= 1:
        return False
    checked = 0
    overbroad = 0
    for row in scores:
        if not isinstance(row, dict):
            continue
        matched = row.get("matched_plan_tasks")
        if not isinstance(matched, list):
            continue
        checked += 1
        if len(set(matched)) >= task_count:
            overbroad += 1
    return checked >= 3 and (overbroad / checked) >= 0.8


def analyze_run_audit(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    issues: list[dict[str, Any]] = []

    for rel in REQUIRED_ARTIFACTS:
        if not (root / rel).is_file():
            issues.append(
                _issue(
                    "missing_required_artifact",
                    f"required audit artifact missing: {rel}",
                    severity="error",
                    path=rel,
                )
            )

    state_summary = _load_json(root / "state_summary.json")
    if isinstance(state_summary, dict):
        if state_summary.get("evidence_chain_status") != "ok":
            ev_n = int(state_summary.get("evidence_item_count") or 0)
            ex_n = int(state_summary.get("extracted_paper_count") or 0)
            if ex_n > 0 and ev_n <= 0:
                issues.append(
                    _issue(
                        "evidence_chain_broken",
                        "extracted papers exist but evidence ledger is empty",
                        severity="error",
                        path="state_summary.json",
                    )
                )
        if state_summary.get("citation_chain_status") != "ok":
            ev_n = int(state_summary.get("evidence_item_count") or 0)
            cite_n = int(state_summary.get("citation_ref_count") or 0)
            if ev_n > 0 and cite_n <= 0:
                issues.append(
                    _issue(
                        "citation_chain_broken",
                        "evidence items exist but CitationMap has no refs",
                        severity="error",
                        path="state_summary.json",
                    )
                )

    filter_report = _load_json(root / "paper_filter" / "filter_report.json")
    if isinstance(filter_report, dict) and _all_scores_match_all_tasks(filter_report):
        issues.append(
            _issue(
                "paper_filter_plan_match_overbroad",
                "all scored papers match all plan tasks; plan-aware filtering is not discriminative",
                severity="warning",
                path="paper_filter/filter_report.json",
            )
        )

    extracted = _load_json(root / "reading" / "extracted_data.summary.json")
    if isinstance(extracted, dict):
        samples = extracted.get("sample")
        if isinstance(samples, list):
            missing_meta = [
                i
                for i, row in enumerate(samples)
                if isinstance(row, dict)
                and not (row.get("paper_id") and row.get("title") and row.get("url"))
            ]
            if missing_meta:
                issues.append(
                    _issue(
                        "extracted_summary_missing_metadata",
                        f"extracted_data.summary sample rows missing paper_id/title/url at indexes {missing_meta[:8]}",
                        severity="warning",
                        path="reading/extracted_data.summary.json",
                    )
                )

    timeline = _load_jsonl(root / "timeline.jsonl")
    seen_nodes = {str(x.get("node_name") or "") for x in timeline}
    missing_nodes = [node for node in MAIN_NODE_NAMES if node not in seen_nodes]
    if missing_nodes:
        issues.append(
            _issue(
                "timeline_missing_main_nodes",
                f"timeline missing main workflow nodes: {missing_nodes}",
                severity="warning",
                path="timeline.jsonl",
            )
        )

    error_count = sum(1 for x in issues if x.get("severity") == "error")
    warning_count = sum(1 for x in issues if x.get("severity") == "warning")
    return {
        "run_dir": str(root),
        "passed": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
    }
