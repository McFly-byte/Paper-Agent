"""TraceEvent 写入 PaperAgentState（避免与 state_models 循环导入）。"""

from __future__ import annotations

from typing import Any

from src.runtime.events import TraceEvent


def append_trace_event(state_value: Any, event: TraceEvent, *, enabled: bool) -> None:
    if not enabled:
        return
    lst = getattr(state_value, "trace_events", None)
    if lst is None:
        lst = []
        setattr(state_value, "trace_events", lst)
    lst.append(event.to_state_dict())


def append_workflow_error(state_value: Any, err: dict[str, Any]) -> None:
    lst = getattr(state_value, "workflow_errors", None)
    if lst is None:
        lst = []
        setattr(state_value, "workflow_errors", lst)
    lst.append(err)
