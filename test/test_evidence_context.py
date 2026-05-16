"""SectionEvidenceContext 单测。"""

from __future__ import annotations

from src.domain.paper.citation import CitationMap
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger
from src.domain.paper.evidence_context import build_section_evidence_context, build_global_evidence_context


def _ledger() -> EvidenceLedger:
    return EvidenceLedger(
        items=[
            EvidenceItem(
                evidence_id="e1",
                paper_id="p1",
                title="T",
                authors=[],
                year=2022,
                section_type="method",
                claim="Method detail is sufficiently long for threshold.",
                supports_argument="method",
                confidence=0.9,
            ),
            EvidenceItem(
                evidence_id="e2",
                paper_id="p1",
                title="T",
                authors=[],
                year=2022,
                section_type="experiment",
                claim="Dataset benchmark numbers are included here long.",
                supports_argument="dataset",
                confidence=0.8,
            ),
            EvidenceItem(
                evidence_id="e3",
                paper_id="p1",
                title="T",
                authors=[],
                year=2022,
                section_type="limitation",
                claim="Limitation text is long enough for the unit test.",
                supports_argument="limitation",
                confidence=0.6,
            ),
        ]
    )


def test_method_section_prefers_method():
    led = _ledger()
    cm = CitationMap.from_evidence_ledger(led)
    ctx = build_section_evidence_context("Approach and Method", "技术路线", led, cm, max_items=4)
    assert "[C1]" in ctx.context_text or "e1" in ctx.context_text
    assert ctx.evidence_items[0].supports_argument == "method"


def test_dataset_section_prefers_dataset():
    led = _ledger()
    cm = CitationMap.from_evidence_ledger(led)
    ctx = build_section_evidence_context("Benchmark evaluation", "数据集与指标", led, cm, max_items=4)
    args = [x.supports_argument for x in ctx.evidence_items]
    assert "dataset" in args or "metric" in args or "experiment" in args


def test_limitation_section():
    led = _ledger()
    cm = CitationMap.from_evidence_ledger(led)
    ctx = build_section_evidence_context("Limitations", "局限与挑战", led, cm, max_items=4)
    assert any(x.supports_argument == "limitation" for x in ctx.evidence_items)


def test_max_items():
    led = _ledger()
    cm = CitationMap.from_evidence_ledger(led)
    ctx = build_section_evidence_context("Mixed", None, led, cm, max_items=1)
    assert len(ctx.evidence_items) <= 1


def test_global_truncation():
    led = _ledger()
    cm = CitationMap.from_evidence_ledger(led)
    g = build_global_evidence_context(led, cm, max_items=10, max_chars=80)
    assert len(g.context_text) <= 80
