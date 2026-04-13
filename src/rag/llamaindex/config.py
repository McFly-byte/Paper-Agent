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
