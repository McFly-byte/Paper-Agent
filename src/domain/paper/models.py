"""论文候选统一模型（检索 / 过滤阶段）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PaperCandidate(BaseModel):
    paper_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    source: str = "arxiv"
    url: str | None = None
    pdf_url: str | None = None
    categories: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
