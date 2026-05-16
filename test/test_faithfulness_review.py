"""Faithfulness 规则审查单测。"""

from __future__ import annotations

from src.domain.paper.citation import CitationMap
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger
from src.domain.paper.faithfulness import analyze_section_text, build_faithfulness_report


def _tiny_map() -> CitationMap:
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
                    claim="Claim is long enough for builder.",
                    supports_argument="result",
                    confidence=0.9,
                ),
            ]
        )
    )


def test_unknown_marker_fail():
    cmap = _tiny_map()
    r = analyze_section_text("S1", "We improve accuracy [C999] greatly.", cmap)
    assert r.verdict == "fail"
    assert any(i.issue_type == "unknown_marker" for i in r.citation_issues)


def test_long_section_no_marker_warning():
    cmap = _tiny_map()
    body = "Lorem ipsum dolor sit amet. " * 40
    r = analyze_section_text("S1", body, cmap, long_section_chars=200)
    assert r.verdict == "warning"


def test_valid_marker_pass():
    cmap = _tiny_map()
    r = analyze_section_text("S1", "We improve accuracy [C1] on the task.", cmap, long_section_chars=2000)
    assert r.verdict == "pass"


def test_chinese_factual_without_marker():
    cmap = _tiny_map()
    # 使用 \\u 转义避免部分环境下源文件编码损坏导致中文匹配失败
    r = analyze_section_text(
        "S1",
        "\u5b9e\u9a8c\u8868\u660e\u8be5\u65b9\u6cd5\u975e\u5e38\u6709\u6548\u3002",
        cmap,
        long_section_chars=5000,
    )
    assert r.unsupported_claims


def test_empty_sections():
    rep = build_faithfulness_report([], _tiny_map())
    assert rep.verdict == "warning"


def test_global_unused_refs():
    cmap = _tiny_map()
    rep = build_faithfulness_report(["short"], cmap, long_section_chars=5000)
    assert rep.global_citation_issues
