from __future__ import annotations

from src.agents.planner.models import AnalysisTask, ResearchPlan, SearchTask, WritingTask
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger
from src.domain.paper.plan_coverage import build_plan_coverage_report


def test_plan_coverage_report_counts_search_analysis_and_writing():
    plan = ResearchPlan(
        title="t",
        search_tasks=[SearchTask(query="tool calling API routing"), SearchTask(query="safety benchmark")],
        reading_tasks=[],
        analysis_tasks=[
            AnalysisTask(
                name="Tool calling comparison",
                analysis_type="method_comparison",
                target_dimensions=["api routing", "tool selection"],
            ),
            AnalysisTask(
                name="Safety benchmark limitations",
                analysis_type="limitation_summary",
                target_dimensions=["safety benchmark", "robustness"],
            ),
        ],
        writing_tasks=[
            WritingTask(section_title="Tool Calling", section_goal="Compare API routing"),
            WritingTask(section_title="Safety Benchmarks", section_goal="Summarize robustness"),
        ],
    )
    filter_report = {
        "task_coverage_summary": [
            {
                "task_index": 0,
                "core_terms": ["tool", "calling", "api", "routing"],
                "covered_paper_count": 1,
                "covered_papers": [{"paper_id": "p1", "title": "Tool Paper"}],
            },
            {
                "task_index": 1,
                "core_terms": ["safety", "benchmark"],
                "covered_paper_count": 1,
                "covered_papers": [{"paper_id": "p2", "title": "Safety Paper"}],
            },
        ]
    }
    ledger = EvidenceLedger(
        items=[
            EvidenceItem(
                evidence_id="e1",
                paper_id="p1",
                title="Tool Paper",
                claim="API routing and tool selection are evaluated.",
                supports_argument="method",
                section_type="method",
                confidence=0.8,
            ),
            EvidenceItem(
                evidence_id="e2",
                paper_id="p2",
                title="Safety Paper",
                claim="Safety benchmark robustness remains limited.",
                supports_argument="limitation",
                section_type="limitation",
                confidence=0.8,
            ),
        ]
    )

    report = build_plan_coverage_report(
        plan=plan,
        filter_report=filter_report,
        analyse_results="The analysis compares API routing, tool selection, safety benchmark robustness.",
        written_sections=[
            "Tool Calling section discusses API routing.",
            "Safety Benchmarks section summarizes robustness.",
        ],
        evidence_ledger=ledger,
    )

    assert report.search_task_coverage_ratio == 1.0
    assert report.analysis_task_coverage_ratio == 1.0
    assert report.writing_task_coverage_ratio == 1.0
    assert report.representative_papers
    assert "Plan Coverage Summary" in report.to_markdown()
