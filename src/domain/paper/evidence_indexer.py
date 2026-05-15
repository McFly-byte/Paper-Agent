"""从结构化阅读结果构建 EvidenceLedger（无伪造页码 / 无 PDF 原文定位）。"""

from __future__ import annotations

import re
from typing import Any

from src.domain.paper.evidence import EvidenceItem, EvidenceLedger, SectionType

_MIN_CLAIM_LEN = 12


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _parse_year_from_published(published: str | None) -> int | None:
    if not published:
        return None
    m = re.search(r"(19|20)\d{2}", published)
    if not m:
        return None
    try:
        y = int(m.group(0))
        return y if 1900 <= y <= 2100 else None
    except ValueError:
        return None


def _slug_evidence_component(s: str) -> str:
    t = re.sub(r"[^\w.\-]+", "_", (s or "").strip(), flags=re.UNICODE)
    return (t[:80] or "x").strip("_") or "x"


def make_evidence_id(paper_id: str, field_name: str, index: int) -> str:
    pid = _slug_evidence_component(paper_id)
    fname = _slug_evidence_component(field_name)
    return f"{pid}:{fname}:{index}"


def _meta_from_paper_dict(meta: dict[str, Any], index: int) -> tuple[str, str, list[str], int | None]:
    pid = _safe_str(meta.get("paper_id")) or _safe_str(meta.get("arxiv_id")) or _safe_str(meta.get("id"))
    if not pid:
        pid = f"idx_{index}"
    title = _safe_str(meta.get("title")) or "Untitled Paper"
    authors_raw = meta.get("authors") or meta.get("author") or []
    if isinstance(authors_raw, str):
        authors = [a.strip() for a in re.split(r"[,;，、]", authors_raw) if a.strip()]
    elif isinstance(authors_raw, list):
        authors = [str(a).strip() for a in authors_raw if str(a).strip()]
    else:
        authors = []
    year = _parse_year_from_published(_safe_str(meta.get("published")) or _safe_str(meta.get("year")))
    return pid, title, authors, year


def _append_item(
    ledger: EvidenceLedger,
    *,
    paper_id: str,
    title: str,
    authors: list[str],
    year: int | None,
    section_type: SectionType,
    supports_argument: str,
    claim: str,
    page_or_section: str,
    confidence: float,
) -> None:
    text = _safe_str(claim)
    if len(text) < _MIN_CLAIM_LEN:
        return
    idx = sum(1 for it in ledger.items if it.paper_id == paper_id and it.supports_argument == supports_argument)
    eid = make_evidence_id(paper_id, supports_argument, idx)
    ledger.add_item(
        EvidenceItem(
            evidence_id=eid,
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            section_type=section_type,
            claim=text,
            original_text=None,
            page_or_section=page_or_section,
            confidence=confidence,
            supports_argument=supports_argument,
        )
    )


def _methodology_claim(km: Any) -> str:
    if km is None:
        return ""
    if isinstance(km, dict):
        name = _safe_str(km.get("name"))
        principle = _safe_str(km.get("principle"))
        novelty = _safe_str(km.get("novelty"))
    else:
        name = _safe_str(getattr(km, "name", None))
        principle = _safe_str(getattr(km, "principle", None))
        novelty = _safe_str(getattr(km, "novelty", None))
    parts = []
    if name:
        parts.append(name)
    if principle:
        parts.append(principle)
    base = ": ".join(parts) if parts else ""
    if novelty:
        base = f"{base}. Novelty: {novelty}" if base else f"Novelty: {novelty}"
    return base.strip()


def build_evidence_ledger(
    extracted_papers: list[dict[str, Any]],
    paper_metadatas: list[dict[str, Any]],
    run_id: str | None = None,
) -> EvidenceLedger:
    """按索引将抽取结果与 successful_papers 元数据对齐。"""
    ledger = EvidenceLedger(items=[])
    _ = run_id  # 预留扩展（如命名空间），当前不参与 evidence_id

    n = min(len(extracted_papers), len(paper_metadatas))
    for i in range(n):
        ex = extracted_papers[i] or {}
        meta = paper_metadatas[i] if isinstance(paper_metadatas[i], dict) else {}
        paper_id, title, authors, year = _meta_from_paper_dict(meta, i)

        core = _safe_str(ex.get("core_problem"))
        _append_item(
            ledger,
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            section_type="introduction",
            supports_argument="problem",
            claim=core,
            page_or_section="structured_reading.core_problem",
            confidence=0.85 if len(core) >= _MIN_CLAIM_LEN else 0.0,
        )

        km_claim = _methodology_claim(ex.get("key_methodology"))
        _append_item(
            ledger,
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            section_type="method",
            supports_argument="method",
            claim=km_claim,
            page_or_section="structured_reading.key_methodology",
            confidence=0.85 if len(km_claim) >= _MIN_CLAIM_LEN else 0.0,
        )

        datasets = ex.get("datasets_used") or []
        if isinstance(datasets, list) and datasets:
            ds_text = "Datasets used: " + "; ".join(_safe_str(d) for d in datasets if _safe_str(d))
            _append_item(
                ledger,
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                section_type="experiment",
                supports_argument="dataset",
                claim=ds_text,
                page_or_section="structured_reading.datasets_used",
                confidence=0.75 if len(ds_text) >= _MIN_CLAIM_LEN else 0.0,
            )

        metrics = ex.get("evaluation_metrics") or []
        if isinstance(metrics, list) and metrics:
            m_text = "Evaluation metrics: " + "; ".join(_safe_str(m) for m in metrics if _safe_str(m))
            _append_item(
                ledger,
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                section_type="experiment",
                supports_argument="metric",
                claim=m_text,
                page_or_section="structured_reading.evaluation_metrics",
                confidence=0.75 if len(m_text) >= _MIN_CLAIM_LEN else 0.0,
            )

        results = _safe_str(ex.get("main_results"))
        _append_item(
            ledger,
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            section_type="result",
            supports_argument="result",
            claim=results,
            page_or_section="structured_reading.main_results",
            confidence=0.85 if len(results) >= _MIN_CLAIM_LEN else 0.0,
        )

        lim = _safe_str(ex.get("limitations"))
        _append_item(
            ledger,
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            section_type="limitation",
            supports_argument="limitation",
            claim=lim,
            page_or_section="structured_reading.limitations",
            confidence=0.65 if len(lim) >= _MIN_CLAIM_LEN else 0.0,
        )

        contribs = ex.get("contributions") or []
        if isinstance(contribs, list):
            for j, c in enumerate(contribs):
                ct = _safe_str(c)
                if len(ct) < _MIN_CLAIM_LEN:
                    continue
                eid = make_evidence_id(paper_id, "contribution", j)
                ledger.add_item(
                    EvidenceItem(
                        evidence_id=eid,
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        year=year,
                        section_type="conclusion",
                        claim=ct,
                        original_text=None,
                        page_or_section=f"structured_reading.contributions[{j}]",
                        confidence=0.75,
                        supports_argument="contribution",
                    )
                )

    return ledger


def extracted_papers_to_reading_snapshots(
    extracted_papers: list[dict[str, Any]],
    paper_metadatas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snaps: list[dict[str, Any]] = []
    n = min(len(extracted_papers), len(paper_metadatas))
    for i in range(n):
        snaps.append(
            {
                "paper": paper_metadatas[i],
                "extracted": extracted_papers[i],
            }
        )
    return snaps
