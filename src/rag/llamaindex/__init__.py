"""LlamaIndex RAG Phase 1（延迟导入，避免仅加载 config 时拉全量依赖）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["LlamaIndexRAGService", "get_llamaindex_rag_service"]

if TYPE_CHECKING:
    from src.rag.llamaindex.service import LlamaIndexRAGService as LlamaIndexRAGService


def __getattr__(name: str) -> Any:
    if name == "LlamaIndexRAGService":
        from src.rag.llamaindex.service import LlamaIndexRAGService as cls

        return cls
    if name == "get_llamaindex_rag_service":
        from src.rag.llamaindex.service import get_llamaindex_rag_service as fn

        return fn
    raise AttributeError(name)
