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


def _infer_source(it: EvidenceItem) -> tuple[str | None, str | None]:
    pid = (it.paper_id or "").lower()
    if "arxiv" in pid or pid.startswith("cs.") or re.search(r"\d{4}\.\d{4,5}", pid):
        return "arXiv", "arxiv"
    return None, None


def _merge_paper_level_fields(rep: EvidenceItem, meta: dict[str, Any] | None) -> tuple[str, list[str], int | None, str | None, str | None, str | None]:
    """合并 EvidenceItem 与外部论文元数据（不编造缺失字段）。"""
    meta = meta or {}
    raw = meta.get("raw") if isinstance(meta.get("raw"), dict) else {}

    title = (meta.get("title") or rep.title or "").strip() or "Untitled"

    authors: list[str]
    if isinstance(meta.get("authors"), list) and meta["authors"]:
        authors = [str(a) for a in meta["authors"]]
    else:
        authors = list(rep.authors or [])

    year: int | None = None
    if meta.get("year") is not None:
        try:
            year = int(meta["year"])
        except (TypeError, ValueError):
            year = rep.year
    else:
        year = rep.year

    url = (
        meta.get("url")
        or meta.get("pdf_url")
        or raw.get("url")
        or raw.get("pdf_url")
        or getattr(rep, "url", None)
        or getattr(rep, "pdf_url", None)
    )
    if isinstance(url, str):
        url = url.strip() or None
    else:
        url = None

    source = meta.get("source") or raw.get("source") or getattr(rep, "source", None)
    if isinstance(source, str):
        source = source.strip() or None
    else:
        source = None

    source_label = meta.get("source_label") or raw.get("source_label") or getattr(rep, "source_label", None)
    if isinstance(source_label, str):
        source_label = source_label.strip() or None
    else:
        source_label = None

    if not source_label and not source:
        sl, s = _infer_source(rep)
        source_label = source_label or sl
        source = source or s

    return title, authors, year, url, source, source_label


class CitationMap(BaseModel):
    refs: list[CitationRef] = Field(default_factory=list)
    evidence_id_to_citation_id: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_evidence_ledger(cls, ledger: EvidenceLedger) -> CitationMap:
        """按 paper_id 聚合：同一论文共享一个 citation_id；保留 evidence_id → citation 映射。"""
        return cls.from_evidence_ledger_with_metadata(ledger, None)

    @classmethod
    def from_evidence_ledger_with_metadata(
        cls,
        ledger: EvidenceLedger,
        paper_metadata_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> CitationMap:
        """在 ``from_evidence_ledger`` 基础上合并 ``paper_id → 元数据``（优先 metadata，再 EvidenceItem）。"""
        if not ledger.items:
            return cls(refs=[], evidence_id_to_citation_id={})

        meta_by_id = paper_metadata_by_id or {}

        paper_order: list[str] = []
        seen_paper: set[str] = set()
        for it in ledger.items:
            pid = (it.paper_id or "").strip() or "_unknown"
            if pid not in seen_paper:
                seen_paper.add(pid)
                paper_order.append(pid)

        ev_map: dict[str, str] = {}
        refs: list[CitationRef] = []

        for i, pid in enumerate(paper_order, start=1):
            cid = f"C{i}"
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

            pm = meta_by_id.get(pid) or meta_by_id.get(rep.paper_id) or {}
            title, authors, year, url, source, source_label = _merge_paper_level_fields(rep, pm if isinstance(pm, dict) else {})

            refs.append(
                CitationRef(
                    citation_id=cid,
                    evidence_id=rep.evidence_id,
                    paper_id=rep.paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    url=url,
                    source=source,
                    source_label=source_label,
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
