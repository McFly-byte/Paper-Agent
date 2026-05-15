"""证据条目与账本（第一阶段模型 + 预留写作/忠实度审查接口）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SectionType = Literal[
    "abstract",
    "introduction",
    "method",
    "experiment",
    "result",
    "limitation",
    "conclusion",
    "unknown",
]


class EvidenceItem(BaseModel):
    evidence_id: str
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    section_type: SectionType = "unknown"
    claim: str
    original_text: str | None = None
    page_or_section: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supports_argument: str | None = None


class EvidenceLedger(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)

    def by_paper_id(self, paper_id: str) -> list[EvidenceItem]:
        return [x for x in self.items if x.paper_id == paper_id]

    def add_item(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def to_citation_context(self, max_items: int = 32) -> str:
        """拼成可供写作/审查引用的短文本（后续可换模板引擎）。"""
        lines: list[str] = []
        for it in self.items[:max_items]:
            who = ", ".join(it.authors[:3]) if it.authors else "Unknown"
            yr = f" ({it.year})" if it.year else ""
            lines.append(
                f"- [{it.evidence_id}] {it.title}{yr} | {who} | {it.section_type}: {it.claim}"
            )
        return "\n".join(lines) if lines else "(empty evidence ledger)"
