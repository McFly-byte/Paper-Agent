"""每请求 / 每 run 的异步上下文（避免全局 config 在并发下串数据）。"""

from contextvars import ContextVar

tmp_db_id_var: ContextVar[str | None] = ContextVar("tmp_db_id", default=None)
