from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langsmith import traceable

from src.utils.log_utils import setup_logger

_trace_log = setup_logger(__name__)


@traceable(
    run_type="chain",
    name="llamaindex.ingest_extracted_papers",
    tags=["paper-agent", "rag", "llamaindex", "ingest"],
)
async def trace_ingest_extracted_papers(
    *,
    rag_backend: str,
    run_id: str,
    db_id: str,
    paper_count: int,
    runner: Callable[[], Awaitable[int]],
) -> dict[str, Any]:
    """包装入库逻辑，输出节点数供 LangSmith 展示。"""
    n = await runner()
    return {
        "rag_backend": rag_backend,
        "run_id": run_id,
        "db_id": db_id,
        "paper_count": paper_count,
        "ingested_nodes": n,
    }


@traceable(
    run_type="chain",
    name="llamaindex.query_for_writing",
    tags=["paper-agent", "rag", "llamaindex", "retrieve"],
)
async def trace_query_for_writing(
    *,
    rag_backend: str,
    query_mode: str,
    queries: list[str],
    top_k: int,
    section_hint: str | None = None,
    runner: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    if section_hint:
        _trace_log.debug(
            "trace_query_for_writing section_hint_preview=%s", (section_hint or "")[:120]
        )
    return await runner()


@traceable(
    run_type="chain",
    name="llamaindex.format_context",
    tags=["paper-agent", "rag", "llamaindex"],
)
def trace_format_context(
    *,
    rag_backend: str,
    query_mode: str,
    retrieved_count: int,
    final_context_count: int,
    runner: Callable[[], list[str]],
) -> list[str]:
    _ = (rag_backend, query_mode, retrieved_count, final_context_count)
    return runner()
