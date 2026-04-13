import asyncio
from typing import Any, List, Dict

from src.knowledge.knowledge import knowledge_base
from src.utils.log_utils import setup_logger
import traceback
import json
from src.core.config import config
from src.core.run_context import tmp_db_id_var, rag_retrieval_logs_var
from src.rag.llamaindex.config import get_rag_backend, llamaindex_top_k
from src.rag.llamaindex.service import get_llamaindex_rag_service

logger = setup_logger(__name__)

_rag_log_lock = asyncio.Lock()


async def _append_rag_log(entry: dict[str, Any]) -> None:
    async with _rag_log_lock:
        buf = rag_retrieval_logs_var.get()
        if buf is not None:
            buf.append(entry)


async def retrieval_tool(querys: List[str]) -> List[Any]:
    """
    检索工具：从向量数据库查询相关文档。
    支持 RAG_BACKEND=legacy（默认）与 llamaindex（临时库多节点索引）。
    """
    retrieval_results: list[Any] = []
    backend = get_rag_backend()
    top_k_li = llamaindex_top_k()

    try:
        tmp_db_id = tmp_db_id_var.get() or config.get("tmp_db_id")
        transformed_q: str | None = None
        retrieved_count = 0

        if tmp_db_id and backend == "llamaindex":
            try:
                svc = get_llamaindex_rag_service()
                pack = await svc.query_for_writing(
                    list(querys or []),
                    db_id=tmp_db_id,
                    top_k=top_k_li,
                    metadata_filters=None,
                )
                strings = pack.get("context_strings") or []
                retrieval_results.extend(strings)
                transformed_q = pack.get("transformed_query")
                retrieved_count = int(pack.get("retrieved_count") or 0)
                final_n = int(pack.get("final_context_count") or len(strings))
                logger.info(
                    "[retrieval_tool] backend=%s query=%r transformed_query=%r top_k=%s "
                    "retrieved_count=%s final_context_count=%s",
                    backend,
                    querys,
                    transformed_q,
                    top_k_li,
                    retrieved_count,
                    final_n,
                )
                await _append_rag_log(
                    {
                        "rag_backend": backend,
                        "query_mode": pack.get("query_mode"),
                        "query": querys,
                        "transformed_query": transformed_q,
                        "top_k": top_k_li,
                        "retrieved_count": retrieved_count,
                        "final_context_count": final_n,
                    }
                )
            except Exception as li_err:  # noqa: BLE001
                logger.exception(
                    "llamaindex 检索失败，回退 legacy 临时库: %s",
                    li_err,
                )
                backend = "legacy"

        if tmp_db_id and backend == "legacy":
            tmpdb_results = await knowledge_base.aquery(
                querys,
                db_id=tmp_db_id,
                top_k=config.get_int("tmpdb_top_k"),
                similarity_threshold=config.get_float("tmpdb_similarity_threshold"),
            )
            if isinstance(tmpdb_results, list):
                for item in tmpdb_results:
                    md = item.get("metadata") or {}
                    if md.get("state_blob") == "1":
                        continue
                    retrieval_results.append(
                        json.dumps(
                            {"content": item.get("content", ""), "metadata": md},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
            retrieved_count = len([x for x in retrieval_results if x])
            logger.info(
                "[retrieval_tool] backend=%s query=%r transformed_query=%r top_k=%s retrieved_count=%s",
                "legacy",
                querys,
                None,
                config.get_int("tmpdb_top_k"),
                retrieved_count,
            )
            await _append_rag_log(
                {
                    "rag_backend": "legacy",
                    "query_mode": "plain",
                    "query": querys,
                    "transformed_query": None,
                    "top_k": config.get_int("tmpdb_top_k"),
                    "retrieved_count": retrieved_count,
                    "final_context_count": len(retrieval_results),
                }
            )

        db_id = config.get("current_db_id", default=None)
        if db_id is None:
            return retrieval_results

        db_results = await knowledge_base.aquery(
            querys,
            db_id=db_id,
            top_k=config.get_int("top_k"),
            similarity_threshold=config.get_float("similarity_threshold"),
        )
        if isinstance(db_results, list):
            for item in db_results:
                md = item.get("metadata") or {}
                if md.get("state_blob") == "1":
                    continue
                doc = item.get("content") or ""
                src = md.get("source", "")
                retrieval_results.append(doc + (" \n来源文件：" + str(src) if src else ""))

        return retrieval_results
    except Exception as e:
        logger.error(f"测试查询失败 {e}, {traceback.format_exc()}")
        return {"message": f"测试查询失败: {e}", "status": "failed"}
