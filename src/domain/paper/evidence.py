"""证据条目与账本（第一阶段模型 + 预留写作/忠实度审查接口）。"""

from __future__ import annotations

from typing import Any, Literal

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

    def by_argument(self, argument: str) -> list[EvidenceItem]:
        if not argument:
            return []
        a = argument.strip().lower()
        return [
            x
            for x in self.items
            if (x.supports_argument or "").strip().lower() == a
        ]

    def by_section_type(self, section_type: str) -> list[EvidenceItem]:
        if not section_type:
            return []
        return [x for x in self.items if x.section_type == section_type]

    def search_claims(self, keyword: str) -> list[EvidenceItem]:
        if not keyword:
            return []
        k = keyword.strip().lower()
        return [x for x in self.items if k in (x.claim or "").lower()]

    def summary(self) -> dict[str, Any]:
        by_sec: dict[str, int] = {}
        by_arg: dict[str, int] = {}
        for it in self.items:
            by_sec[it.section_type] = by_sec.get(it.section_type, 0) + 1
            ak = it.supports_argument or "unknown"
            by_arg[ak] = by_arg.get(ak, 0) + 1
        confs = [it.confidence for it in self.items]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return {
            "total_items": len(self.items),
            "section_counts": by_sec,
            "argument_counts": by_arg,
            "avg_confidence": round(avg_conf, 4),
        }

    def add_item(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def to_writer_context(self, max_items: int = 64) -> str:
        """面向写作代理的紧凑证据块（非 PDF 原文定位）。"""
        lines: list[str] = []
        for it in self.items[:max_items]:
            arg = f"[{it.supports_argument}] " if it.supports_argument else ""
            lines.append(
                f"- {arg}[{it.evidence_id}] ({it.section_type}, conf={it.confidence:.2f}) {it.claim}"
            )
        return "\n".join(lines) if lines else "(empty evidence ledger)"

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

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
