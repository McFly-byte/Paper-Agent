"""主工作流失败后的恢复：每节点最多 2 次带反馈重试（共 3 轮尝试），由 recovery 节点写入 feedback 并清错重路由。"""

from __future__ import annotations

import json
from typing import Any, Final, Optional

from src.core.state_models import NodeError, PaperAgentState

# 逻辑节点键 → (NodeError 字段名, LangGraph 节点名)
NODE_SPEC: Final[dict[str, tuple[str, str]]] = {
    "search": ("search_node_error", "search_node"),
    "paper_filter": ("paper_filter_node_error", "paper_filter_node"),
    "reading": ("reading_node_error", "reading_node"),
    "evidence_index": ("evidence_index_node_error", "evidence_index_node"),
    "analyse": ("analyse_node_error", "analyse_node"),
    "writing": ("writing_node_error", "writing_node"),
    "report": ("report_node_error", "report_node"),
}

# 同一主节点在「已消耗恢复次数」≥ 此值时不再进入恢复，直接终态错误节点（首轮失败记 0 次已消耗）
MAX_RECOVERY_CYCLES_PER_NODE: Final[int] = 2


def _cfg(state: PaperAgentState) -> dict[str, Any]:
    return state.config if isinstance(state.config, dict) else {}


def node_recovery_cycles(cfg: dict[str, Any], node_key: str) -> int:
    raw = (cfg.get("node_recovery_cycles") or {}).get(node_key)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def bump_node_recovery_cycle(state: PaperAgentState, node_key: str) -> int:
    cfg = _cfg(state)
    m = dict(cfg.get("node_recovery_cycles") or {})
    cur = node_recovery_cycles(cfg, node_key)
    nxt = cur + 1
    m[node_key] = nxt
    cfg["node_recovery_cycles"] = m
    state.config = cfg
    return nxt


def should_abort_to_terminal_error(state: PaperAgentState, node_key: str) -> bool:
    """当前失败是否已用尽该节点的恢复配额（不含本次将触发的恢复）。"""
    return node_recovery_cycles(_cfg(state), node_key) >= MAX_RECOVERY_CYCLES_PER_NODE


def set_recovery_target(state: PaperAgentState, node_key: str) -> None:
    if node_key not in NODE_SPEC:
        return
    cfg = _cfg(state)
    cfg["recovery_target"] = node_key
    state.config = cfg


def infer_recovery_target(state: PaperAgentState) -> Optional[str]:
    cfg = _cfg(state)
    t = cfg.get("recovery_target")
    if isinstance(t, str) and t in NODE_SPEC:
        return t
    err = state.error
    if err is None:
        return None
    for key, (attr, _) in NODE_SPEC.items():
        v = getattr(err, attr, None)
        if v is not None and str(v).strip():
            return key
    if err.error and str(err.error).strip():
        return None
    return None


def clear_node_error_field(err: Optional[NodeError], node_key: str) -> None:
    if err is None or node_key not in NODE_SPEC:
        return
    attr, _ = NODE_SPEC[node_key]
    setattr(err, attr, None)


def graph_node_for(node_key: str) -> str:
    return NODE_SPEC[node_key][1]


def error_attr_for(node_key: str) -> str:
    return NODE_SPEC[node_key][0]


def format_recovery_user_block(cfg: dict[str, Any], node_key: str) -> str:
    """注入到用户任务/章节提示中的可读块（已由 recovery LLM 生成）。"""
    fb = (cfg.get("recovery_feedback") or {}).get(node_key)
    if not fb or not str(fb).strip():
        return ""
    n = node_recovery_cycles(cfg, node_key)
    return (
        f"\n\n【错误恢复模块】（第 {n} 次恢复反馈，请严格对照执行）\n"
        f"{str(fb).strip()}\n"
    )


def store_recovery_feedback(state: PaperAgentState, node_key: str, text: str) -> None:
    cfg = _cfg(state)
    m = dict(cfg.get("recovery_feedback") or {})
    m[node_key] = (text or "").strip()[:8000]
    cfg["recovery_feedback"] = m
    state.config = cfg


def clear_recovery_feedback_for(state: PaperAgentState, node_key: str) -> None:
    cfg = _cfg(state)
    m = dict(cfg.get("recovery_feedback") or {})
    m.pop(node_key, None)
    cfg["recovery_feedback"] = m
    state.config = cfg


def boundary_gate_blob(state: PaperAgentState, node_key: str) -> str:
    bc = state.boundary_checks or {}
    ent = bc.get(node_key)
    if not isinstance(ent, dict):
        return ""
    try:
        return json.dumps(ent, ensure_ascii=False)[:6000]
    except (TypeError, ValueError):
        return str(ent)[:6000]
