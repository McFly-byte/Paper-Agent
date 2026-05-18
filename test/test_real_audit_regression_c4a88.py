from __future__ import annotations

import json
from pathlib import Path

from src.observability.audit_analyzer import analyze_run_audit


RUN_DIR = Path("examples/run_audits/c4a88a32-c825-47f8-ac7f-eef811229c4f")


def test_c4a88_evidence_and_citation_chain_are_ok():
    state = json.loads((RUN_DIR / "state_summary.json").read_text(encoding="utf-8"))
    assert state["evidence_chain_status"] == "ok"
    assert state["citation_chain_status"] == "ok"
    assert state["evidence_item_count"] > 0
    assert state["citation_ref_count"] > 0


def test_c4a88_audit_analyzer_flags_known_gaps():
    result = analyze_run_audit(RUN_DIR)
    codes = {issue["code"] for issue in result["issues"]}
    assert "missing_required_artifact" in codes
    assert "paper_filter_plan_match_overbroad" in codes
    assert "extracted_summary_missing_metadata" in codes


def test_c4a88_timeline_has_main_nodes():
    rows = [
        json.loads(line)
        for line in (RUN_DIR / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen = {row.get("node_name") for row in rows}
    for node in [
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
    ]:
        assert node in seen
