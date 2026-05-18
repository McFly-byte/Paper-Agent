from __future__ import annotations

from src.agents.planner.models import ResearchPlan, SearchTask
from src.domain.paper.filter import filter_candidates, normalize_paper_candidates


def test_plan_tasks_are_matched_specifically_not_all_tasks():
    plan = ResearchPlan(
        title="LLM agents",
        search_tasks=[
            SearchTask(
                query="LLM agent tool calling function calling API routing",
                inclusion_criteria=["structured tool function calling and API routing"],
                top_k=10,
            ),
            SearchTask(
                query="LLM agent safety alignment benchmark tool misuse robustness",
                inclusion_criteria=["safety benchmark for tool misuse and robustness"],
                top_k=10,
            ),
        ],
        reading_tasks=[],
        analysis_tasks=[],
        writing_tasks=[],
    )
    cands = normalize_paper_candidates(
        [
            {
                "paper_id": "tool-1",
                "title": "Function Calling and API Routing for Tool-Using LLM Agents",
                "abstract": "We study parameter parsing, tool selection, and API routing recovery.",
                "published": "2025",
                "pdf_url": "http://x/tool.pdf",
            },
            {
                "paper_id": "safe-1",
                "title": "Agent Safety Benchmark for Tool Misuse and Robustness",
                "abstract": "A dynamic safety alignment benchmark covers permission escalation and unsafe tool use.",
                "published": "2025",
                "pdf_url": "http://x/safe.pdf",
            },
        ]
    )

    result = filter_candidates(cands, plan, None, 10, {}, min_score=0.01)
    scores = {s.paper_id: s for s in result.scores}

    assert scores["tool-1"].matched_plan_tasks == [0]
    assert scores["safe-1"].matched_plan_tasks == [1]
    assert result.plan_task_coverage == {"0": 1, "1": 1}
    assert all(len(s.matched_plan_tasks) < len(plan.search_tasks) for s in result.scores)
    assert result.task_coverage_summary[0]["covered_papers"][0]["paper_id"] == "tool-1"
