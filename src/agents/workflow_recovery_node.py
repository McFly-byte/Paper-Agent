"""LangGraph 恢复节点：分析失败原因并写入 recovery_feedback，清空对应 NodeError 后重试原节点。"""

from __future__ import annotations

import json
from typing import Any

from langgraph.runtime import Runtime

from src.core.llm_infra.invoke import chat_completion_text_routed
from src.core.state_models import BackToFrontData, ExecutionState, NodeError, PaperRunContext, State
from src.core.workflow_recovery import (
    NODE_SPEC,
    bump_node_recovery_cycle,
    clear_node_error_field,
    graph_node_for,
    infer_recovery_target,
    boundary_gate_blob,
    store_recovery_feedback,
)
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _error_detail(state: Any, node_key: str) -> str:
    err = getattr(state, "error", None)
    if err is None:
        return ""
    attr = NODE_SPEC[node_key][0]
    return str(getattr(err, attr, "") or "").strip()


async def workflow_recovery_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    state_queue = runtime.context.state_queue
    current = state["value"]
    if current.error is None:
        current.error = NodeError()

    target = infer_recovery_target(current)
    if not target:
        logger.warning("[工作流·恢复] 无法推断失败节点，跳过 LLM 分析")
        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.RECOVERING,
                state="error",
                data="恢复节点未找到失败来源",
            )
        )
        current.config = dict(current.config or {})
        current.config["retry_next_node"] = "handle_error_node"
        return {"value": current}

    detail = _error_detail(current, target)
    gate = boundary_gate_blob(current, target)
    user_req = (current.user_request or "")[:2000]

    analysis_prompt = f"""你是 Paper-Agent 工作流的故障分析助手。根据下列信息，用中文给出**简短、可执行**的修正建议（不超过 12 条要点，每条一行），面向将再次运行的「{target}」业务节点与相关 LLM。

【用户原始需求摘要】
{user_req}

【节点】{target}
【错误/门禁说明】
{detail or "（无文本）"}

【该节点门禁 JSON 摘录】
{gate or "（无）"}

请输出纯文本要点列表，不要 Markdown 围栏，不要 JSON。重点写：如何满足门禁、避免再次触发同类错误。"""

    try:
        fb = await chat_completion_text_routed(
            "search-model",
            analysis_prompt,
            node_name="workflow_recovery",
            temperature=0.2,
            source="workflow_recovery",
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("[工作流·恢复] LLM 分析失败: %s", e)
        fb = (
            f"（自动恢复分析不可用：{e!s}）请结合门禁原因自行修正：\n"
            f"{detail[:1500]}"
        )

    store_recovery_feedback(current, target, fb)
    bump_node_recovery_cycle(current, target)
    clear_node_error_field(current.error, target)

    current.config = dict(current.config or {})
    current.config["retry_next_node"] = graph_node_for(target)
    current.config.pop("recovery_target", None)

    logger.info(
        "[工作流·恢复] 已对节点 %s 写入恢复反馈，下一跳=%s",
        target,
        current.config["retry_next_node"],
    )
    await state_queue.put(
        BackToFrontData(
            step=ExecutionState.RECOVERING,
            state="completed",
            data=json.dumps(
                {"node": target, "feedback_preview": fb[:500]},
                ensure_ascii=False,
            ),
        )
    )
    return {"value": current}
