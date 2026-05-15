"""证据索引构建单测。"""

from __future__ import annotations

from src.domain.paper.evidence_indexer import build_evidence_ledger
from src.domain.paper.evidence import EvidenceLedger


def _sample_extracted() -> list[dict]:
    return [
        {
            "core_problem": "Autonomous driving needs better scene understanding under occlusion scenarios.",
            "key_methodology": {
                "name": "SparseAttention",
                "principle": "Prunes redundant tokens in attention maps.",
                "novelty": "First applied to lidar BEV encoders.",
            },
            "datasets_used": ["nuScenes", "KITTI"],
            "evaluation_metrics": ["mAP", "NDS"],
            "main_results": "mAP improves from 0.42 to 0.51 on nuScenes validation split.",
            "limitations": "Higher compute on very long sequences remains challenging for deployment.",
            "contributions": [
                "Proposes a sparse attention variant for BEV transformers.",
                "Shows consistent gains across two driving datasets.",
            ],
        }
    ]


def _sample_meta() -> list[dict]:
    return [
        {
            "paper_id": "arxiv:2301.00001",
            "title": "Sparse BEV Attention",
            "authors": ["Alice", "Bob"],
            "published": "2023-05-01",
        }
    ]


def test_build_evidence_ledger_multiple_items():
    ledger = build_evidence_ledger(_sample_extracted(), _sample_meta(), run_id="r1")
    assert len(ledger.items) >= 6
    kinds = {(it.section_type, it.supports_argument) for it in ledger.items}
    assert ("introduction", "problem") in kinds
    assert ("method", "method") in kinds
    assert ("experiment", "dataset") in kinds
    assert ("result", "result") in kinds
    assert ("limitation", "limitation") in kinds
    assert ("conclusion", "contribution") in kinds


def test_metadata_year_and_authors():
    ledger = build_evidence_ledger(_sample_extracted(), _sample_meta())
    for it in ledger.items:
        assert it.paper_id
        assert it.title
        assert it.authors
        assert it.year == 2023


def test_short_fields_do_not_create_noise_items():
    ledger = build_evidence_ledger(
        [{"core_problem": "short", "main_results": "x", "limitations": "", "contributions": []}],
        [{"paper_id": "p", "title": "T", "authors": []}],
    )
    assert len(ledger.items) == 0


def test_evidence_ledger_summary_and_citation():
    ledger = build_evidence_ledger(_sample_extracted(), _sample_meta())
    s = ledger.summary()
    assert s["total_items"] == len(ledger.items)
    assert "section_counts" in s
    ctx = ledger.to_citation_context(max_items=50)
    assert "Sparse BEV" in ctx or len(ctx) > 0
    roundtrip = EvidenceLedger.model_validate(ledger.to_jsonable())
    assert len(roundtrip.items) == len(ledger.items)


def test_evidence_ledger_helpers():
    ledger = build_evidence_ledger(_sample_extracted(), _sample_meta())
    assert ledger.by_argument("problem")
    assert ledger.by_section_type("introduction")
    assert ledger.search_claims("nuScenes")
    wc = ledger.to_writer_context(max_items=10)
    assert "[" in wc
