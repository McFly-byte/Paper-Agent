from __future__ import annotations

from src.domain.paper.retrieval_mode import diagnose_retrieval_mode


def test_zero_rag_calls_with_zero_llamaindex_nodes_is_evidence_context_only():
    d = diagnose_retrieval_mode(
        config={"rag_backend": "llamaindex"},
        llamaindex_node_count=0,
        kb_write_count=10,
        rag_retrieval_logs=[],
        evidence_bound_sections_count=3,
    )
    assert d.retrieval_mode == "evidence_context_only"
    assert "llamaindex_configured_but_no_ingested_nodes" in d.reasons
    assert "writing_rag_call_count_is_zero" in d.reasons


def test_llamaindex_requires_ingestion_and_positive_writing_call():
    d = diagnose_retrieval_mode(
        config={"rag_backend": "llamaindex"},
        llamaindex_node_count=12,
        rag_retrieval_logs=[
            {"rag_backend": "llamaindex", "retrieved_count": 4, "final_context_count": 2}
        ],
    )
    assert d.retrieval_mode == "llamaindex"


def test_legacy_tmp_store_when_legacy_tool_call_positive():
    d = diagnose_retrieval_mode(
        config={"rag_backend": "legacy"},
        kb_write_count=3,
        rag_retrieval_logs=[
            {"rag_backend": "legacy", "retrieved_count": 2, "final_context_count": 2}
        ],
    )
    assert d.retrieval_mode == "legacy_tmp_store"
