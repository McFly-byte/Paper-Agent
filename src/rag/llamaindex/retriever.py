from __future__ import annotations

import asyncio
from typing import Any

from llama_index.core import VectorStoreIndex
from src.core.llm_infra.invoke import chat_completion_text_routed
from src.rag.llamaindex import config as li_cfg
from src.rag.llamaindex.types import RetrievalHit, RetrievalResponse
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _meta_match(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    for k, v in filters.items():
        if str(meta.get(k)) != str(v):
            return False
    return True


async def _hyde_transform(query: str) -> str | None:
    try:
        prompt = (
            "请根据用户检索问题，写一段简短、通顺的英文或中文假想论文段落（不必真实），"
            "用于向量检索扩展；只输出段落正文，不要标题或解释。\n\n问题：\n"
            f"{query}"
        )
        text = await chat_completion_text_routed(
            "rag-generation-model",
            prompt,
            node_name="rag_hyde",
            temperature=0.2,
            source="rag_hyde",
        )
        t = (text or "").strip()
        return t or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("HyDE 变换失败，将使用原查询: %s", exc)
        return None


class LlamaIndexRetriever:
    """向量召回 + 简单 metadata 后过滤；rerank 在 service 层统一完成。"""

    def __init__(self, index: VectorStoreIndex) -> None:
        self._index = index

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filters: dict[str, Any] | None = None,
        query_mode: str | None = None,
    ) -> RetrievalResponse:
        mode = (query_mode or li_cfg.llamaindex_query_mode()).lower()
        if mode not in ("plain", "hyde"):
            mode = "plain"
        transformed: str | None = None
        qtext = query
        if mode == "hyde":
            hyp = await _hyde_transform(query)
            if hyp:
                qtext = hyp
                transformed = hyp
            else:
                mode = "plain"

        retriever = self._index.as_retriever(similarity_top_k=max(1, top_k))
        ar = getattr(retriever, "aretrieve", None)
        if callable(ar):
            nodes = await ar(qtext)
        else:
            nodes = await asyncio.to_thread(retriever.retrieve, qtext)
        hits: list[RetrievalHit] = []
        use_filter = bool(metadata_filters) and li_cfg.llamaindex_enable_metadata_filters()

        for nws in nodes:
            n = nws.node
            meta = dict(n.metadata or {})
            if use_filter and metadata_filters and not _meta_match(meta, metadata_filters):
                continue
            score = float(getattr(nws, "score", None) or 0.0)
            text = getattr(n, "text", None) or ""
            if not text and hasattr(n, "get_content"):
                try:
                    text = n.get_content(metadata_mode="none")  # type: ignore[call-arg]
                except TypeError:
                    text = n.get_content()
            text = text or ""
            title = str(meta.get("title") or "")
            paper_id = str(meta.get("paper_id") or "")
            chunk_id = str(meta.get("chunk_id") or getattr(n, "id_", "") or "")
            source = str(meta.get("source_type") or "llamaindex_chroma")
            hits.append(
                RetrievalHit(
                    text=text,
                    score=score,
                    title=title,
                    paper_id=paper_id,
                    source=source,
                    chunk_id=chunk_id,
                    metadata=meta,
                )
            )

        return RetrievalResponse(hits=hits, transformed_query=transformed, query_mode=mode)
