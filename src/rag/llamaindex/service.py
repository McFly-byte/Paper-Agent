from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from src.knowledge.knowledge import knowledge_base
from src.rag.llamaindex import config as li_cfg
from src.rag.llamaindex.ingestion import (
    build_documents_from_extracted_pairs,
    embedding_settings_for_db,
    sentence_chunk_documents,
)
from src.rag.llamaindex.retriever import LlamaIndexRetriever
from src.rag.llamaindex.trace import (
    trace_format_context,
    trace_ingest_extracted_papers,
    trace_query_for_writing,
)
from src.rag.llamaindex.types import RetrievalHit, RetrievalResponse
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_LI_COLLECTION_SUFFIX = "__paper_agent_li_v1"


def li_collection_name(db_id: str) -> str:
    return f"{db_id}{_LI_COLLECTION_SUFFIX}"


def _api_base_for_llamaindex_openai(base_url: str) -> str | None:
    b = (base_url or "").strip().replace("/embeddings", "").rstrip("/")
    if not b:
        return None
    if not b.endswith("/v1"):
        return f"{b}/v1"
    return b


class LlamaIndexRAGService:
    """LlamaIndex RAG 统一入口：ingest / query；异常内部消化并降级。"""

    def __init__(self) -> None:
        self._index_by_db: dict[str, Any] = {}

    def _try_imports(
        self,
    ) -> tuple[Any, Any, Any, Any, Any, Any] | None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            from llama_index.core import StorageContext, VectorStoreIndex
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore

            return (
                chromadb,
                OpenAIEmbeddingFunction,
                StorageContext,
                VectorStoreIndex,
                OpenAIEmbedding,
                ChromaVectorStore,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LlamaIndex / Chroma 依赖导入失败，RAG 适配层不可用: %s", exc)
            return None

    def _chromadb_embedding_fn(self, emd: dict[str, Any], OpenAIEmbeddingFunction: Any) -> Any:
        base = (emd.get("base_url") or "").replace("/embeddings", "")
        return OpenAIEmbeddingFunction(
            model_name=emd["model"],
            api_key=emd["api_key"],
            api_base=base,
        )

    def _openai_embed_model(self, emd: dict[str, Any], OpenAIEmbedding: Any) -> Any:
        api_base = _api_base_for_llamaindex_openai(emd.get("base_url") or "")
        kwargs: dict[str, Any] = {
            "model": emd["model"],
            "api_key": emd["api_key"],
        }
        if api_base:
            kwargs["api_base"] = api_base
        return OpenAIEmbedding(**kwargs)

    def _ingest_sync(
        self,
        packs: tuple[Any, Any, Any, Any, Any, Any],
        db_id: str,
        run_id: str,
        paper_meta_list: Sequence[dict[str, Any]],
        extracted_list: Sequence[dict[str, Any]],
    ) -> tuple[int, Any | None]:
        chromadb, OpenAIEmbeddingFunction, StorageContext, VectorStoreIndex, OpenAIEmbedding, ChromaVectorStore = packs
        try:
            kb = knowledge_base.get_kb(db_id)
            if getattr(kb, "kb_type", None) != "chroma":
                logger.warning("LlamaIndex ingest 跳过：非 Chroma 知识库 db_id=%s", db_id)
                return 0, None
            chroma_path = getattr(kb, "chroma_db_path", None)
            if not chroma_path:
                return 0, None
            embed_info = (kb.databases_meta.get(db_id) or {}).get("embed_info") or {}
            emd = embedding_settings_for_db(embed_info if isinstance(embed_info, dict) else {})
            emb_fn = self._chromadb_embedding_fn(emd, OpenAIEmbeddingFunction)
            li_name = li_collection_name(db_id)
            client = chromadb.PersistentClient(path=chroma_path)
            try:
                client.delete_collection(li_name)
            except Exception:  # noqa: BLE001
                pass
            collection = client.create_collection(name=li_name, embedding_function=emb_fn)
            pairs = list(zip(paper_meta_list, extracted_list))
            if len(pairs) != len(paper_meta_list) or len(pairs) != len(extracted_list):
                logger.warning(
                    "ingest_extracted_papers: 论文与抽取条数不一致 (%s vs %s)，已按较短一侧 zip",
                    len(paper_meta_list),
                    len(extracted_list),
                )
            docs = build_documents_from_extracted_pairs(pairs, run_id=run_id)
            embed_model = self._openai_embed_model(emd, OpenAIEmbedding)
            final_docs = sentence_chunk_documents(docs, embed_model)
            if not final_docs:
                logger.info("LlamaIndex ingest：无节点可写入")
                return 0, None
            vs = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vs)
            index = VectorStoreIndex.from_documents(
                final_docs,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=False,
            )
            return len(final_docs), index
        except Exception as exc:  # noqa: BLE001
            logger.exception("LlamaIndex ingest_extracted_papers 失败（已降级）: %s", exc)
            return 0, None

    async def ingest_extracted_papers(
        self,
        *,
        db_id: str,
        run_id: str,
        paper_meta_list: Sequence[dict[str, Any]],
        extracted_list: Sequence[dict[str, Any]],
    ) -> int:
        if not li_cfg.llamaindex_ingestion_enabled():
            return 0
        packs = self._try_imports()
        if packs is None:
            return 0

        async def _runner() -> int:
            n, index = await asyncio.to_thread(
                self._ingest_sync, packs, db_id, run_id, paper_meta_list, extracted_list
            )
            if index is not None:
                self._index_by_db[db_id] = index
            return n

        meta = await trace_ingest_extracted_papers(
            rag_backend="llamaindex",
            run_id=run_id,
            db_id=db_id,
            paper_count=len(paper_meta_list),
            runner=_runner,
        )
        return int(meta.get("ingested_nodes") or 0)

    def _load_index(self, db_id: str, packs: tuple[Any, ...]) -> Any | None:
        chromadb, OpenAIEmbeddingFunction, StorageContext, VectorStoreIndex, OpenAIEmbedding, ChromaVectorStore = packs
        try:
            kb = knowledge_base.get_kb(db_id)
            if getattr(kb, "kb_type", None) != "chroma":
                return None
            chroma_path = getattr(kb, "chroma_db_path", None)
            if not chroma_path:
                return None
            embed_info = (kb.databases_meta.get(db_id) or {}).get("embed_info") or {}
            emd = embedding_settings_for_db(embed_info if isinstance(embed_info, dict) else {})
            emb_fn = self._chromadb_embedding_fn(emd, OpenAIEmbeddingFunction)
            client = chromadb.PersistentClient(path=chroma_path)
            li_name = li_collection_name(db_id)
            collection = client.get_collection(name=li_name, embedding_function=emb_fn)
            embed_model = self._openai_embed_model(emd, OpenAIEmbedding)
            vs = ChromaVectorStore(chroma_collection=collection)
            return VectorStoreIndex.from_vector_store(vector_store=vs, embed_model=embed_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LlamaIndex 加载索引失败 db_id=%s: %s", db_id, exc)
            return None

    async def _get_index(self, db_id: str) -> Any | None:
        if db_id in self._index_by_db and self._index_by_db[db_id] is not None:
            return self._index_by_db[db_id]
        packs = self._try_imports()
        if packs is None:
            return None
        idx = await asyncio.to_thread(self._load_index, db_id, packs)
        if idx is not None:
            self._index_by_db[db_id] = idx
        return idx

    async def query(
        self,
        query_text: str,
        *,
        db_id: str,
        top_k: int | None = None,
        metadata_filters: dict[str, Any] | None = None,
        query_mode: str | None = None,
    ) -> RetrievalResponse:
        k = top_k if top_k is not None else li_cfg.llamaindex_top_k()
        idx = await self._get_index(db_id)
        if idx is None:
            return RetrievalResponse(hits=[], transformed_query=None, query_mode=query_mode or "plain")
        retriever = LlamaIndexRetriever(idx, enable_rerank=li_cfg.llamaindex_enable_rerank())
        return await retriever.retrieve(
            query_text,
            top_k=k,
            metadata_filters=metadata_filters,
            query_mode=query_mode,
        )

    async def query_for_writing(
        self,
        queries: list[str],
        *,
        db_id: str,
        top_k: int | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        k = top_k if top_k is not None else li_cfg.llamaindex_top_k()
        mode = li_cfg.llamaindex_query_mode()

        async def _inner() -> dict[str, Any]:
            all_hits: list[RetrievalHit] = []
            tq_last: str | None = None
            nq = 0
            for q in queries:
                if not (q or "").strip():
                    continue
                nq += 1
                resp = await self.query(
                    q.strip(),
                    db_id=db_id,
                    top_k=k,
                    metadata_filters=metadata_filters,
                    query_mode=mode,
                )
                all_hits.extend(resp.hits)
                if resp.transformed_query:
                    tq_last = resp.transformed_query
            all_hits.sort(key=lambda h: h.score, reverse=True)
            seen: set[str] = set()
            deduped: list[RetrievalHit] = []
            for h in all_hits:
                key = h.chunk_id or f"{h.paper_id}:{hash(h.text) % 10**9}"
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(h)
            cap = max(k, k * max(1, nq))
            trimmed = deduped[:cap]
            strings = trace_format_context(
                rag_backend="llamaindex",
                query_mode=mode,
                retrieved_count=len(all_hits),
                final_context_count=len(trimmed),
                runner=lambda: self.format_context_for_writing(trimmed),
            )
            return {
                "hits": trimmed,
                "context_strings": strings,
                "transformed_query": tq_last,
                "query_mode": mode,
                "retrieved_count": len(all_hits),
                "final_context_count": len(trimmed),
            }

        return await trace_query_for_writing(
            rag_backend="llamaindex",
            query_mode=mode,
            queries=queries,
            top_k=k,
            runner=_inner,
        )

    @staticmethod
    def format_context_for_writing(hits: Sequence[RetrievalHit]) -> list[str]:
        """与 legacy 临时库 JSON 串形式对齐，便于 retrieval_agent 消费。"""
        out: list[str] = []
        for h in hits:
            md = {
                **h.metadata,
                "score": h.score,
                "chunk_id": h.chunk_id,
                "paper_id": h.paper_id,
                "title": h.title,
                "source": h.source,
            }
            out.append(json.dumps({"content": h.text, "metadata": md}, ensure_ascii=False, indent=2))
        return out


_service: LlamaIndexRAGService | None = None


def get_llamaindex_rag_service() -> LlamaIndexRAGService:
    global _service
    if _service is None:
        _service = LlamaIndexRAGService()
    return _service
