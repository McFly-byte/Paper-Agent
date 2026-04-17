from __future__ import annotations

from typing import Any

from src.core.config import config


def _lower(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def get_rag_backend() -> str:
    """legacy | llamaindex；环境变量 RAG_BACKEND 或 system_params rag.backend。"""
    v = config.get("RAG_BACKEND") or config.get("rag.backend") or "legacy"
    x = _lower(v)
    return x if x in ("legacy", "llamaindex") else "legacy"


def llamaindex_ingestion_enabled() -> bool:
    return config.get_bool("LLAMAINDEX_ENABLE", False) or config.get_bool(
        "rag.llamaindex.enable", False
    )


def llamaindex_top_k() -> int:
    k = config.get_int("LLAMAINDEX_TOP_K", 0) or config.get_int("rag.llamaindex.top_k", 5)
    return max(1, min(k, 50))


def llamaindex_query_mode() -> str:
    m = _lower(config.get("LLAMAINDEX_QUERY_MODE") or config.get("rag.llamaindex.query_mode"))
    return m if m in ("plain", "hyde") else "plain"


def llamaindex_enable_rerank() -> bool:
    return config.get_bool("LLAMAINDEX_ENABLE_RERANK", False) or config.get_bool(
        "rag.llamaindex.enable_rerank", False
    )


_RERANK_MODES = frozenset({"tfidf_fuse", "cross_encoder", "cohere", "voyage", "none"})


def llamaindex_rerank_mode() -> str:
    m = _lower(config.get("LLAMAINDEX_RERANK_MODE") or config.get("rag.llamaindex.rerank.mode"))
    return m if m in _RERANK_MODES else "tfidf_fuse"


def llamaindex_rerank_fuse_alpha() -> float:
    a = config.get_float("rag.llamaindex.rerank.fuse_alpha", 0.7)
    return max(0.0, min(1.0, a))


def llamaindex_cross_encoder_model() -> str:
    return str(
        config.get("LLAMAINDEX_CROSS_ENCODER_MODEL")
        or config.get("rag.llamaindex.rerank.cross_encoder_model")
        or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ).strip()


def llamaindex_rerank_cohere_model() -> str:
    return str(
        config.get("rag.llamaindex.rerank.cohere_model") or "rerank-multilingual-v3.0"
    ).strip()


def llamaindex_rerank_voyage_model() -> str:
    return str(config.get("rag.llamaindex.rerank.voyage_model") or "rerank-2").strip()


def llamaindex_rerank_cohere_api_key() -> str:
    return str(
        config.get("COHERE_API_KEY") or config.get("rag.llamaindex.rerank.cohere_api_key") or ""
    ).strip()


def llamaindex_rerank_voyage_api_key() -> str:
    return str(
        config.get("VOYAGE_API_KEY") or config.get("rag.llamaindex.rerank.voyage_api_key") or ""
    ).strip()


def llamaindex_recall_top_k() -> int:
    """0 表示由 effective_recall 按 final top_k 自动放大。"""
    v = config.get_int("LLAMAINDEX_RECALL_TOP_K", 0) or config.get_int("rag.llamaindex.recall_top_k", 0)
    return max(0, min(v, 50))


def llamaindex_expand_recall_pool() -> bool:
    """仅在启用 rerank 时放大向量召回；multi-query 本身已提高覆盖面。"""
    return llamaindex_enable_rerank()


def llamaindex_effective_recall_top_k(final_k: int) -> int:
    if not llamaindex_expand_recall_pool():
        return max(1, final_k)
    r = llamaindex_recall_top_k()
    if r <= 0:
        return min(50, max(final_k * 4, max(15, final_k * 3)))
    return min(50, max(r, final_k))


def llamaindex_should_apply_rerank() -> bool:
    if not llamaindex_enable_rerank():
        return False
    return llamaindex_rerank_mode() != "none"


def llamaindex_multi_query_enabled() -> bool:
    return config.get_bool("LLAMAINDEX_MULTI_QUERY", False) or config.get_bool(
        "rag.llamaindex.multi_query.enabled", False
    )


def llamaindex_multi_query_max() -> int:
    return max(2, min(config.get_int("rag.llamaindex.multi_query.max_queries", 6), 12))


def llamaindex_hyde_primary_only() -> bool:
    return config.get_bool("rag.llamaindex.hyde_primary_only", True)


def llamaindex_section_intent_enabled() -> bool:
    return config.get_bool("rag.llamaindex.section_intent.enabled", True)


def llamaindex_section_classifier() -> str:
    """keywords：规则关键词；llm：轻量 Chat 分类（失败回退关键词）。"""
    m = _lower(
        config.get("LLAMAINDEX_SECTION_CLASSIFIER")
        or config.get("rag.llamaindex.section_intent.classifier")
    )
    return m if m in ("keywords", "llm") else "keywords"


def llamaindex_section_intent_llm_client_type() -> str:
    return str(
        config.get("rag.llamaindex.section_intent.llm_client_type") or "rag-section-intent-model"
    ).strip()


def llamaindex_section_filter_mode() -> str:
    m = _lower(config.get("rag.llamaindex.section_intent.filter_mode"))
    return m if m in ("off", "soft", "hard") else "soft"


def llamaindex_section_boost() -> float:
    return max(0.0, min(0.35, config.get_float("rag.llamaindex.section_intent.boost", 0.12)))


def llamaindex_enable_metadata_filters() -> bool:
    return config.get_bool("LLAMAINDEX_ENABLE_METADATA_FILTERS", False) or config.get_bool(
        "rag.llamaindex.enable_metadata_filters", False
    )


def llamaindex_use_semantic_splitter() -> bool:
    return config.get_bool("rag.llamaindex.use_semantic_splitter", False)


def llamaindex_chunk_size() -> int:
    return max(128, config.get_int("rag.llamaindex.chunk_size", 512))


def llamaindex_chunk_overlap() -> int:
    return max(0, config.get_int("rag.llamaindex.chunk_overlap", 64))
