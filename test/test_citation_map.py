"""CitationMap 单测。"""

from __future__ import annotations

from src.domain.paper.citation import CitationMap
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger


def _ledger_two_same_paper() -> EvidenceLedger:
    return EvidenceLedger(
        items=[
            EvidenceItem(
                evidence_id="p1:method:0",
                paper_id="paper-1",
                title="Title One",
                authors=["A"],
                year=2023,
                section_type="method",
                claim="Method claim one is long enough for tests here.",
                supports_argument="method",
                confidence=0.8,
            ),
            EvidenceItem(
                evidence_id="p1:result:0",
                paper_id="paper-1",
                title="Title One",
                authors=["A"],
                year=2023,
                section_type="result",
                claim="Result claim two is also long enough for tests.",
                supports_argument="result",
                confidence=0.8,
            ),
            EvidenceItem(
                evidence_id="p2:problem:0",
                paper_id="paper-2",
                title="Title Two",
                authors=["B"],
                year=None,
                section_type="introduction",
                claim="Another paper problem statement long enough.",
                supports_argument="problem",
                confidence=0.7,
            ),
        ]
    )


def test_from_evidence_ledger_groups_by_paper():
    cmap = CitationMap.from_evidence_ledger(_ledger_two_same_paper())
    assert len(cmap.refs) == 2
    assert cmap.get_marker("p1:method:0") == "[C1]"
    assert cmap.get_marker("p1:result:0") == "[C1]"
    assert cmap.get_marker("p2:problem:0") == "[C2]"


def test_to_markdown_references_no_fake_fields():
    cmap = CitationMap.from_evidence_ledger(_ledger_two_same_paper())
    md = cmap.to_markdown_references()
    assert "[C1]" in md
    assert "DOI" not in md
    assert "page" not in md.lower()


def test_empty_ledger():
    cmap = CitationMap.from_evidence_ledger(EvidenceLedger(items=[]))
    assert cmap.refs == []
    assert cmap.get_marker("x") == ""
