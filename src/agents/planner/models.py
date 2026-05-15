"""研究计划相关 Pydantic 模型（planner / plan_review 输出）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchTask(BaseModel):
    query: str
    source: Literal["arxiv", "local_pdf", "web", "vector_db"] = "arxiv"
    time_range: tuple[str | None, str | None] = (None, None)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    top_k: int = 50


class ReadingTask(BaseModel):
    focus: str
    required_fields: list[str] = Field(default_factory=list)
    extraction_schema_name: str | None = None


class AnalysisTask(BaseModel):
    name: str
    analysis_type: Literal[
        "clustering",
        "method_comparison",
        "trend_analysis",
        "limitation_summary",
        "metric_comparison",
    ]
    target_dimensions: list[str] = Field(default_factory=list)
    description: str | None = None


class WritingTask(BaseModel):
    section_title: str
    section_goal: str
    required_evidence_types: list[str] = Field(default_factory=list)
    expected_length: int | None = None


class ResearchPlan(BaseModel):
    title: str
    thought: str = ""
    has_enough_context: bool = True
    search_tasks: list[SearchTask] = Field(default_factory=list)
    reading_tasks: list[ReadingTask] = Field(default_factory=list)
    analysis_tasks: list[AnalysisTask] = Field(default_factory=list)
    writing_tasks: list[WritingTask] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)


class PlanReviewResult(BaseModel):
    approved: bool = False
    modified_plan: ResearchPlan | None = None
    user_feedback: str | None = None
    auto_approved: bool = False
