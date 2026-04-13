"""每请求 / 每 run 的异步上下文（避免全局 config 在并发下串数据）。"""

from contextvars import ContextVar
from typing import Any

tmp_db_id_var: ContextVar[str | None] = ContextVar("tmp_db_id", default=None)

# 写作子图内由 retrieval_tool 追加，writing_node finally 合并回 PaperAgentState.rag_retrieval_logs
rag_retrieval_logs_var: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "rag_retrieval_logs", default=None
)
