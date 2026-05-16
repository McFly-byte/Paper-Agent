"""CitationMap：报告内 [C1] 与 EvidenceItem / 论文元数据的映射（不编造 DOI/页码）。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from src.domain.paper.evidence import EvidenceItem, EvidenceLedger


class CitationRef(BaseModel):
    citation_id: str = Field(description="如 C1、C2")
    evidence_id: str = Field(description="该条引用绑定的代表 evidence_id（同 paper 首条）")
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str | None = None
    source: str | None = None
    source_label: str | None = Field(default=None, description="如 arXiv")
    inline_marker: str = Field(default="", description="如 [C1]")


class CitationMap(BaseModel):
    refs: list[CitationRef] = Field(default_factory=list)
    evidence_id_to_citation_id: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_evidence_ledger(cls, ledger: EvidenceLedger) -> CitationMap:
        """按 paper_id 聚合：同一论文共享一个 citation_id；保留 evidence_id → citation 映射。"""
        if not ledger.items:
            return cls(refs=[], evidence_id_to_citation_id={})

        paper_order: list[str] = []
        seen_paper: set[str] = set()
        for it in ledger.items:
            pid = (it.paper_id or "").strip() or "_unknown"
            if pid not in seen_paper:
                seen_paper.add(pid)
                paper_order.append(pid)

        paper_to_cid: dict[str, str] = {}
        ev_map: dict[str, str] = {}
        refs: list[CitationRef] = []

        for i, pid in enumerate(paper_order, start=1):
            cid = f"C{i}"
            paper_to_cid[pid] = cid
            rep: EvidenceItem | None = None
            for it in ledger.items:
                p = (it.paper_id or "").strip() or "_unknown"
                if p == pid:
                    rep = it
                    break
            if rep is None:
                continue
            for it in ledger.items:
                p = (it.paper_id or "").strip() or "_unknown"
                if p == pid:
                    ev_map[it.evidence_id] = cid

            src_lbl, src = _infer_source(rep)
            refs.append(
                CitationRef(
                    citation_id=cid,
                    evidence_id=rep.evidence_id,
                    paper_id=rep.paper_id,
                    title=rep.title or "Untitled",
                    authors=list(rep.authors or []),
                    year=rep.year,
                    url=None,
                    source=src,
                    source_label=src_lbl,
                    inline_marker=f"[{cid}]",
                )
            )

        return cls(refs=refs, evidence_id_to_citation_id=ev_map)

    def get_ref_by_evidence_id(self, evidence_id: str) -> CitationRef | None:
        cid = self.evidence_id_to_citation_id.get(evidence_id)
        if not cid:
            return None
        for r in self.refs:
            if r.citation_id == cid:
                return r
        return None

    def get_marker(self, evidence_id: str) -> str:
        cid = self.evidence_id_to_citation_id.get(evidence_id or "")
        return f"[{cid}]" if cid else ""

    def citation_ids(self) -> set[str]:
        return {r.citation_id for r in self.refs}

    def to_markdown_references(self) -> str:
        lines: list[str] = []
        for r in self.refs:
            auth = ", ".join(r.authors[:12]) if r.authors else "Unknown authors"
            yr = f" ({r.year})" if r.year else ""
            tail_parts: list[str] = []
            if r.source_label:
                tail_parts.append(r.source_label)
            elif r.source:
                tail_parts.append(r.source)
            if r.url:
                tail_parts.append(r.url)
            tail = ". ".join(tail_parts) if tail_parts else ""
            line = f"{r.inline_marker} {auth}.{yr} {r.title}."
            if tail:
                line = f"{line} {tail}"
            lines.append(line.strip())
        return "\n".join(lines) if lines else ""

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _infer_source(it: EvidenceItem) -> tuple[str | None, str | None]:
    pid = (it.paper_id or "").lower()
    if "arxiv" in pid or pid.startswith("cs.") or re.search(r"\d{4}\.\d{4,5}", pid):
        return "arXiv", "arxiv"
    return None, None
