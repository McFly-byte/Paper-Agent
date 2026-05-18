from __future__ import annotations

from src.observability.run_audit import _extracted_data_compact


def test_extracted_summary_backfills_metadata_by_index():
    extracted = {
        "papers": [
            {
                "core_problem": "A long enough problem statement.",
                "key_methodology": {"name": "M", "novelty": "N"},
                "contributions": ["c1", "c2"],
                "main_results": "Strong result.",
            }
        ]
    }
    readings = [
        {
            "paper": {
                "paper_id": "2501.00001v1",
                "title": "Paper Title",
                "url": "http://arxiv.org/abs/2501.00001v1",
                "published": "2025-01-01",
            }
        }
    ]

    summary = _extracted_data_compact(
        extracted,
        max_papers=5,
        max_chars=100,
        redact=False,
        metadata_sources=[readings],
    )

    row = summary["sample"][0]
    assert row["paper_id"] == "2501.00001v1"
    assert row["title"] == "Paper Title"
    assert row["url"] == "http://arxiv.org/abs/2501.00001v1"
    assert row["published"] == "2025-01-01"
