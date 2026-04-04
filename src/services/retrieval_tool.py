from typing import List, Dict, Any
from src.knowledge.knowledge import knowledge_base
from src.utils.log_utils import setup_logger
import traceback
import json
from src.core.config import config
from src.core.run_context import tmp_db_id_var

logger = setup_logger(__name__)

async def retrieval_tool(querys: List[str]) -> List[List[Dict[str, Any]]]:
    """
    检索工具，从向量数据库中查询相关文档
    
    :param querys: 查询文本列表
    :return: 包含文档的列表
    """
    retrieval_results = []

    try:
        # 从临时知识库中检索文档（优先当前 run 的 ContextVar，兼容旧脚本回退到 config）
        tmp_db_id = tmp_db_id_var.get() or config.get("tmp_db_id")
        if tmp_db_id:
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
