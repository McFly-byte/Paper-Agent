"""规划链与 search adapter 最小单测（不调用真实 LLM）。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.planner.models import PlanReviewResult, ResearchPlan, SearchTask
from src.agents.planner.defaults import default_research_plan
from src.agents.search_plan_context import inject_search_plan_hints
from src.core.state_models import NodeError, PaperAgentState
from src.runtime.state import BackgroundContext, ResearchBrief


def test_inject_search_plan_hints_when_no_plan():
    st = PaperAgentState(run_id="1", user_request="topic", error=NodeError(), config={})
    inject_search_plan_hints(st)
    assert "search_plan_hints" not in st.config


def test_inject_search_plan_hints_when_approved():
    st = PaperAgentState(
        run_id="1",
        user_request="topic",
        error=NodeError(),
        config={},
        plan=ResearchPlan(
            title="t",
            search_tasks=[
                SearchTask(query="all:gnn", source="arxiv", top_k=10),
                SearchTask(query="all:transformer", source="arxiv"),
            ],
            reading_tasks=[],
            analysis_tasks=[],
            writing_tasks=[],
        ),
        plan_approved=True,
    )
    inject_search_plan_hints(st)
    assert "search_plan_hints" in st.config
    assert "all:gnn" in st.config["search_plan_hints"]


def test_default_research_plan_nonempty_tasks():
    brief = ResearchBrief(original_query="q", clarified_topic="LLM survey")
    bg = BackgroundContext(topic="t", suggested_search_queries=["rag", "lora"])
    plan = default_research_plan(brief, bg)
    assert plan.search_tasks
    assert plan.reading_tasks
    assert plan.analysis_tasks
    assert plan.writing_tasks


def test_normalize_plan_dict_llm_shape():
    """LLM 常见偏差：缺 title、criteria 为字符串、reading 用 description、analysis 用 task_id。"""
    from src.agents.planner.plan_normalization import normalize_plan_dict

    brief = ResearchBrief(original_query="u", clarified_topic="主题A")
    raw = {
        "thought": "计划说明第一行作为标题回退",
        "search_tasks": [
            {"query": "q1", "inclusion_criteria": "准则一段", "exclusion_criteria": "排除一段"},
        ],
        "reading_tasks": [{"task_id": "r1", "description": "阅读重点描述"}],
        "analysis_tasks": [{"task_id": "a1", "description": "分析说明"}],
        "writing_tasks": [{"section_title": "引言", "section_goal": "定义问题"}],
        "evaluation_criteria": "单条标准",
    }
    plan = ResearchPlan.model_validate(normalize_plan_dict(raw, brief))
    assert plan.title
    assert plan.search_tasks[0].inclusion_criteria == ["准则一段"]
    assert "阅读重点" in plan.reading_tasks[0].focus
    assert plan.analysis_tasks[0].name
    assert plan.analysis_tasks[0].analysis_type == "clustering"
    assert plan.evaluation_criteria == ["单条标准"]


def test_plan_review_no_plan_does_not_crash():
    import asyncio

    from src.agents.planner.review_node import plan_review_node
    from src.core.state_models import PaperRunContext, State

    st: State = {
        "value": PaperAgentState(
            run_id="r",
            user_request="u",
            error=NodeError(),
            plan=None,
        )
    }
    ctx = PaperRunContext(state_queue=asyncio.Queue(), user_proxy=MagicMock())
    rt = MagicMock()
    rt.context = ctx

    out = asyncio.run(plan_review_node(st, rt))
    assert out["value"].plan_approved is True
    assert out["value"].plan_review is not None


def test_coordinator_uses_heuristic_on_llm_failure():
    import asyncio

    from src.agents.coordinator import node as cn

    st: State = {
        "value": PaperAgentState(
            run_id="r",
            user_request="diffusion",
            error=NodeError(),
            trace_events=[],
        )
    }
    ctx = MagicMock()
    ctx.context = MagicMock(state_queue=asyncio.Queue(), user_proxy=MagicMock())

    async def _run():
        with patch("src.agents.coordinator.node._get_coordinator_agent") as mock_get:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=RuntimeError("no api"))
            mock_get.return_value = mock_agent
            return await cn.coordinator_node(st, ctx)

    out = asyncio.run(_run())
    assert out["value"].brief is not None
    assert out["value"].brief.original_query == "diffusion"
