"""v2 运行时与领域模型基础单测（无外部网络）。"""

from __future__ import annotations

from src.agents.planner.models import (
    AnalysisTask,
    ReadingTask,
    ResearchPlan,
    SearchTask,
    WritingTask,
)
from src.core.state_models import NodeError, PaperAgentState
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger
from src.domain.paper.models import PaperCandidate
from src.runtime.events import TraceEvent
from src.runtime.state import AgentRunState, BackgroundContext, ResearchBrief
from src.agents.planner.models import PlanReviewResult


def test_research_brief_defaults():
    b = ResearchBrief(original_query="x", clarified_topic="y")
    assert b.task_type == "survey"
    assert b.language == "zh"


def test_trace_event_serialization():
    te = TraceEvent(
        run_id="r1",
        event_type="TEST",
        node_name="n",
        status="finished",
    )
    d = te.to_state_dict()
    assert d["run_id"] == "r1"
    assert d["status"] == "finished"


def test_evidence_ledger_helpers():
    led = EvidenceLedger()
    e = EvidenceItem(
        evidence_id="e1",
        paper_id="p1",
        title="T",
        claim="c",
        confidence=0.5,
    )
    led.add_item(e)
    assert led.by_paper_id("p1")[0].evidence_id == "e1"
    assert "e1" in led.to_citation_context()


def test_paper_agent_state_has_v2_fields():
    st = PaperAgentState(
        run_id="rid",
        user_request="hello",
        error=NodeError(),
        brief=ResearchBrief(original_query="hello", clarified_topic="hello"),
    )
    assert st.brief is not None
    assert isinstance(st.trace_events, list)


def test_agent_run_state_from_paper_agent_state():
    st = PaperAgentState(
        run_id="r",
        user_request="q",
        error=NodeError(),
        plan=ResearchPlan(
            title="p",
            search_tasks=[SearchTask(query="all:attention", source="arxiv")],
            reading_tasks=[
                ReadingTask(focus="f", required_fields=["core_problem"], extraction_schema_name=None)
            ],
            analysis_tasks=[
                AnalysisTask(name="a", analysis_type="clustering", target_dimensions=["m"])
            ],
            writing_tasks=[
                WritingTask(
                    section_title="s",
                    section_goal="g",
                    required_evidence_types=["abstract"],
                )
            ],
        ),
        plan_review=PlanReviewResult(approved=True, auto_approved=True),
        plan_approved=True,
    )
    snap = AgentRunState.from_paper_agent_state(st)
    assert snap.run_id == "r"
    assert snap.plan is not None


def test_paper_candidate_raw():
    pc = PaperCandidate(paper_id="1", title="t", raw={"k": 1})
    assert pc.raw["k"] == 1


def test_background_context():
    bg = BackgroundContext(topic="t", expanded_keywords=["a", "b"])
    assert len(bg.expanded_keywords) == 2
