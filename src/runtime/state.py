"""v2 运行域模型：ResearchBrief / BackgroundContext / AgentRunState（与 LangGraph 并存，经 model_dump 可序列化）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agents.planner.models import PlanReviewResult, ResearchPlan
from src.domain.paper.evidence import EvidenceLedger


TaskType = Literal[
    "survey",
    "related_work",
    "method_comparison",
    "benchmark_review",
    "reading_report",
]
Audience = Literal[
    "group_meeting",
    "thesis",
    "paper_related_work",
    "technical_report",
    "interview_project_summary",
]
ReportLang = Literal["zh", "en"]


class ResearchBrief(BaseModel):
    """coordinator_node 输出：澄清调研意图，不触发检索。"""

    original_query: str
    clarified_topic: str
    domain: str | None = None
    task_type: TaskType = "survey"
    target_audience: Audience = "technical_report"
    time_range: tuple[str | None, str | None] = (None, None)
    language: ReportLang = "zh"
    clarification_needed: bool = False
    clarification_questions: list[str] = Field(default_factory=list)


class BackgroundContext(BaseModel):
    """background_investigation_node 输出：关键词与检索建议（第一阶段可无真实联网）。"""

    topic: str
    expanded_keywords: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    initial_findings: list[str] = Field(default_factory=list)
    suggested_search_queries: list[str] = Field(default_factory=list)
    notes: str | None = None


class AgentRunState(BaseModel):
    """统一快照模型：描述一次 run 在 v2 流水线中的关键字段（可与 PaperAgentState 字段对齐）。"""

    run_id: str
    user_query: str
    brief: ResearchBrief | None = None
    background_context: BackgroundContext | None = None
    plan: ResearchPlan | None = None
    plan_review: PlanReviewResult | None = None
    plan_approved: bool = False
    paper_candidates: list[Any] = Field(default_factory=list, description="PaperCandidate 列表，暂用 Any 避免环导入")
    filtered_papers: list[Any] = Field(default_factory=list)
    paper_readings: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ledger: EvidenceLedger | None = None
    analysis_results: dict[str, Any] = Field(default_factory=dict)
    draft_sections: list[dict[str, Any]] = Field(default_factory=list)
    review_results: list[dict[str, Any]] = Field(default_factory=list)
    final_report: str | None = None
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_paper_agent_state(cls, state: Any) -> AgentRunState:
        """从 PaperAgentState 组装快照（duck typing，避免硬依赖导入顺序）。"""
        trace = list(getattr(state, "trace_events", None) or [])
        errs = list(getattr(state, "workflow_errors", None) or [])
        return cls(
            run_id=str(getattr(state, "run_id", "") or ""),
            user_query=str(getattr(state, "user_request", "") or ""),
            brief=getattr(state, "brief", None),
            background_context=getattr(state, "background_context", None),
            plan=getattr(state, "plan", None),
            plan_review=getattr(state, "plan_review", None),
            plan_approved=bool(getattr(state, "plan_approved", False)),
            paper_candidates=list(getattr(state, "paper_candidates", None) or []),
            filtered_papers=list(getattr(state, "filtered_papers", None) or []),
            paper_readings=list(getattr(state, "paper_readings", None) or []),
            evidence_ledger=getattr(state, "evidence_ledger", None),
            analysis_results=(
                {"analyse_results": ar}
                if (ar := getattr(state, "analyse_results", None)) is not None
                else {}
            ),
            draft_sections=list(getattr(state, "writted_sections", None) or []),
            review_results=[],
            final_report=getattr(state, "report_markdown", None),
            trace_events=trace,
            errors=errs,
        )
