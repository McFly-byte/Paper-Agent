"""Report-level citation validation 单测（无 LLM）。"""

from __future__ import annotations

from src.domain.paper.citation import CitationMap
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger
from src.domain.paper.report_citation_validation import validate_report_citations


def _map_one() -> CitationMap:
    return CitationMap.from_evidence_ledger(
        EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="e1",
                    paper_id="p1",
                    title="T",
                    authors=["A"],
                    year=2020,
                    section_type="result",
                    claim="Claim one is long enough for evidence builder tests.",
                    supports_argument="result",
                    confidence=0.9,
                ),
            ]
        )
    )


def test_valid_report_pass_or_warning_no_unknown():
    cmap = _map_one()
    report = "# R\n\nHello [C1] world.\n\n## References\n\n" + cmap.to_markdown_references()
    r = validate_report_citations(report, cmap)
    assert r.unknown_markers == []
    assert r.verdict in ("pass", "warning")


def test_unknown_marker_fail():
    cmap = _map_one()
    r = validate_report_citations("Bad [C999] here.\n\n## References\n\nx", cmap)
    assert r.verdict == "fail"
    assert "[C999]" in r.unknown_markers


def test_missing_references_warning():
    cmap = _map_one()
    r = validate_report_citations("Only body [C1] no section.", cmap)
    assert r.missing_references_section is True
    assert r.verdict == "warning"


def test_no_markers_in_body_while_map_nonempty_warning():
    cmap = _map_one()
    ref = cmap.to_markdown_references()
    report = "# Title\n\n正文无任何编号。\n\n## References\n\n" + ref
    r = validate_report_citations(report, cmap)
    assert r.verdict == "warning"


def test_unused_allowed_warning_not_fail():
    cmap = CitationMap.from_evidence_ledger(
        EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="e1",
                    paper_id="p1",
                    title="T1",
                    authors=["A"],
                    year=2020,
                    section_type="result",
                    claim="Claim one is long enough for evidence builder tests.",
                    supports_argument="result",
                    confidence=0.9,
                ),
                EvidenceItem(
                    evidence_id="e2",
                    paper_id="p2",
                    title="T2",
                    authors=["B"],
                    year=2021,
                    section_type="result",
                    claim="Claim two is long enough for evidence builder tests.",
                    supports_argument="result",
                    confidence=0.9,
                ),
            ]
        )
    )
    report = "Body [C1] only.\n\n## References\n\n" + cmap.to_markdown_references()
    r = validate_report_citations(report, cmap, allow_unused_references=True)
    assert "[C2]" in r.unused_references
    assert r.verdict == "warning"


def test_unused_not_allowed_fail():
    cmap = CitationMap.from_evidence_ledger(
        EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="e1",
                    paper_id="p1",
                    title="T1",
                    authors=["A"],
                    year=2020,
                    section_type="result",
                    claim="Claim one is long enough for evidence builder tests.",
                    supports_argument="result",
                    confidence=0.9,
                ),
                EvidenceItem(
                    evidence_id="e2",
                    paper_id="p2",
                    title="T2",
                    authors=["B"],
                    year=2021,
                    section_type="result",
                    claim="Claim two is long enough for evidence builder tests.",
                    supports_argument="result",
                    confidence=0.9,
                ),
            ]
        )
    )
    report = "Body [C1] only.\n\n## References\n\n" + cmap.to_markdown_references()
    r = validate_report_citations(report, cmap, allow_unused_references=False)
    assert r.verdict == "fail"


def test_missing_citation_map_returns_warning_not_empty_pass():
    r = validate_report_citations("# R\n\nBody [C1].", None)
    assert r.verdict == "warning"
    assert r.issues
    assert r.summary
