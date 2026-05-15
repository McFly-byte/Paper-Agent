"""结构化 Trace 事件（过程可观测；第一阶段主要写入 PaperAgentState.trace_events）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


TraceStatus = Literal["started", "finished", "failed", "skipped"]


class TraceEvent(BaseModel):
    """单条节点级追踪事件，可序列化后进入 SSE / LangSmith 扩展。"""

    run_id: str
    event_type: str = Field(description="业务事件类型，如 BRIEF_GENERATED、PLAN_APPROVED")
    node_name: str
    agent_name: str | None = None
    status: TraceStatus
    input_summary: str | None = None
    output_summary: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO8601 UTC",
    )

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
