"""单次调研 run 的大字段外置：与临时向量库（tmp_db_id）同库，检索时需过滤 state_blob。"""

from __future__ import annotations

import json
from typing import Any

from src.core.config import config
from src.core.state_models import PaperAgentState
from src.knowledge.knowledge import knowledge_base
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

KEY_SEARCH_RESULTS = "workflow_search_results"
KEY_FILTERED_PAPERS = "workflow_filtered_papers"
KEY_EXTRACTED_DATA = "workflow_extracted_data"
KEY_READING_SUCCESSFUL_PAPERS = "workflow_reading_successful_papers"
KEY_EVIDENCE_LEDGER = "workflow_evidence_ledger"
KEY_ANALYSE_RESULTS = "workflow_analyse_results"
KEY_WRITTED_SECTIONS = "workflow_writted_sections"


async def ensure_run_tmp_kb(current_state: PaperAgentState) -> str:
    """为本 run 创建或复用临时 Chroma 库，写入 config['tmp_db_id']。"""
    cfg = current_state.config or {}
    existing = cfg.get("tmp_db_id")
    if existing:
        return existing

    embedding_dic = config.get("embedding-model")
    embedding_provider = embedding_dic.get("model-provider")
    provider_dic = config.get(embedding_provider)

    embed_info = {
        "name": embedding_dic.get("model"),
        "dimension": embedding_dic.get("dimension"),
        "base_url": provider_dic.get("base_url"),
        "api_key": provider_dic.get("api_key"),
    }
    kb_type = config.get("KB_TYPE")
    database_name = f"调研临时_{current_state.run_id}"
    database_info = await knowledge_base.create_database(
        database_name,
        "单次调研临时库：论文摘要向量 + 工作流大字段外置",
        kb_type=kb_type,
        embed_info=embed_info,
        llm_info=None,
    )
    db_id = database_info["db_id"]
    if current_state.config is None:
        current_state.config = {}
    current_state.config["tmp_db_id"] = db_id
    logger.info("[run_tmp_kb] 已创建临时库 run_id=%s db_id=%s", current_state.run_id, db_id)
    return db_id


def _db_id(state: PaperAgentState) -> str | None:
    return (state.config or {}).get("tmp_db_id")


async def put_json(state: PaperAgentState, blob_key: str, obj: Any) -> None:
    db_id = _db_id(state)
    if not db_id:
        raise ValueError("put_json: tmp_db_id 未初始化，请先 ensure_run_tmp_kb")
    await knowledge_base.put_state_blob(db_id, blob_key, json.dumps(obj, ensure_ascii=False))


async def get_json(state: PaperAgentState, blob_key: str) -> Any | None:
    db_id = _db_id(state)
    if not db_id:
        return None
    raw = await knowledge_base.get_state_blob(db_id, blob_key)
    if raw is None:
        return None
    return json.loads(raw)


async def put_text(state: PaperAgentState, blob_key: str, text: str) -> None:
    db_id = _db_id(state)
    if not db_id:
        raise ValueError("put_text: tmp_db_id 未初始化")
    await knowledge_base.put_state_blob(db_id, blob_key, text or "")


async def get_text(state: PaperAgentState, blob_key: str) -> str | None:
    db_id = _db_id(state)
    if not db_id:
        return None
    return await knowledge_base.get_state_blob(db_id, blob_key)
