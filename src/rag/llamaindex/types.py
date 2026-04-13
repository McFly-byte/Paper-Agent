from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalHit:
    """单条检索结果（写作与评估消费）。"""

    text: str
    score: float
    title: str
    paper_id: str
    source: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    """检索器原始输出。"""

    hits: list[RetrievalHit]
    transformed_query: str | None = None
    query_mode: str = "plain"
