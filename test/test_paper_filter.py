"""论文过滤领域逻辑单测（无 LLM）。"""

from __future__ import annotations

from src.agents.planner.models import ResearchPlan, SearchTask
from src.domain.paper.filter import (
    deduplicate_candidates,
    filter_candidates,
    normalize_paper_candidate,
    normalize_paper_candidates,
)
from src.domain.paper.models import PaperCandidate
from src.runtime.state import BackgroundContext


def test_normalize_paper_candidate_missing_fields():
    c = normalize_paper_candidate({})
    assert c.title == "Untitled Paper"
    assert c.paper_id.startswith("hash:")


def test_normalize_paper_candidate_stable_id_from_arxiv_url():
    raw = {"title": "T", "url": "https://arxiv.org/abs/2301.00001v2"}
    c = normalize_paper_candidate(raw)
    assert "2301.00001" in c.paper_id.lower()


def test_normalize_paper_candidate_authors_variants():
    c1 = normalize_paper_candidate({"title": "x", "authors": ["A", "B"]})
    assert c1.authors == ["A", "B"]
    c2 = normalize_paper_candidate({"title": "x", "author": "A, B"})
    assert len(c2.authors) == 2


def test_deduplicate_by_title_and_url():
    a = normalize_paper_candidate({"title": "Hello World", "abstract": "x" * 50, "paper_id": "1"})
    b = normalize_paper_candidate(
        {"title": "hello   world!!", "abstract": "short", "paper_id": "2", "pdf_url": "http://x/y.pdf"}
    )
    out, stats = deduplicate_candidates([a, b])
    assert len(out) == 1
    assert stats.duplicates_removed == 1
    # 更长摘要优先
    assert len(out[0].abstract or "") >= 50


def test_filter_candidates_keyword_and_exclusion():
    plan = ResearchPlan(
        title="p",
        search_tasks=[
            SearchTask(
                query="transformer attention",
                inclusion_criteria=["benchmark"],
                exclusion_criteria=["survey only"],
                top_k=10,
            )
        ],
        reading_tasks=[],
        analysis_tasks=[],
        writing_tasks=[],
    )
    cands = [
        normalize_paper_candidate(
            {
                "title": "Transformer benchmark on X",
                "abstract": "We present attention mechanism and benchmark results.",
                "paper_id": "a1",
                "published": "2023",
                "pdf_url": "http://p/a.pdf",
            }
        ),
        normalize_paper_candidate(
            {
                "title": "survey only overview",
                "abstract": "This survey only paper discusses unrelated topics without benchmark.",
                "paper_id": "a2",
            }
        ),
    ]
    r = filter_candidates(
        cands,
        plan,
        None,
        max_papers=50,
        config={},
        user_request="",
        min_score=0.05,
    )
    assert r.output_count >= 1
    assert r.scores
    low = next(s for s in r.scores if s.paper_id == "a2")
    assert low.exclusion_penalty > 0


def test_filter_candidates_empty_input():
    r = filter_candidates([], None, None, 5, {}, user_request="hi")
    assert r.output_count == 0
    assert "empty_input" in r.warnings


def test_filter_candidates_all_low_score_fallback():
    plan = ResearchPlan(
        title="p",
        search_tasks=[SearchTask(query="zzzznonexistenttokenqqqq", top_k=5)],
        reading_tasks=[],
        analysis_tasks=[],
        writing_tasks=[],
    )
    cands = normalize_paper_candidates(
        [
            {"title": "A", "abstract": "nothing", "paper_id": "p1"},
            {"title": "B", "abstract": "nothing", "paper_id": "p2"},
        ]
    )
    r = filter_candidates(
        cands,
        plan,
        BackgroundContext(topic="t", expanded_keywords=[]),
        max_papers=50,
        config={},
        user_request="",
        min_score=0.99,
        fallback_keep_top_k=True,
    )
    assert r.output_count >= 1
    assert any("fallback" in w for w in r.warnings)
