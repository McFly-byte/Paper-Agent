"""plan_review_node：人工审核占位 + 默认自动通过。"""

from __future__ import annotations

import os

from langgraph.runtime import Runtime

from src.core.config import config
from src.core.state_models import PaperAgentState, PaperRunContext, State
from src.agents.planner.models import PlanReviewResult
from src.runtime.events import TraceEvent
from src.runtime.trace_utils import append_trace_event, append_workflow_error
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _trace_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_TRACE_EVENTS")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_trace_events", True)


def _auto_approve() -> bool:
    raw = os.environ.get("PAPER_AGENT_AUTO_APPROVE_PLAN")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.auto_approve_plan", True)


async def plan_review_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    val = state["value"]
    run_id = val.run_id
    node = "plan_review_node"
    enable_trace = _trace_enabled(val)

    if val.plan is None:
        val.plan_review = PlanReviewResult(
            approved=True,
            modified_plan=None,
            user_feedback=None,
            auto_approved=True,
        )
        val.plan_approved = True
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="PLAN_APPROVED",
                node_name=node,
                status="finished",
                output_summary="no_plan_present_auto_continue",
            ),
            enabled=enable_trace,
        )
        return {"value": val}

    append_trace_event(
        val,
        TraceEvent(
            run_id=run_id,
            event_type="NODE_LIFECYCLE",
            node_name=node,
            agent_name=None,
            status="started",
            input_summary=val.plan.title if val.plan else None,
        ),
        enabled=enable_trace,
    )

    try:
        auto = _auto_approve()
        if auto:
            val.plan_review = PlanReviewResult(
                approved=True,
                modified_plan=None,
                user_feedback=None,
                auto_approved=True,
            )
            val.plan_approved = True
        else:
            # 预留：未来 HITL 在此阻塞等待前端输入；当前阶段仍自动通过以免卡死图
            logger.warning(
                "[plan_review_node] AUTO_APPROVE_PLAN=false，当前阶段仍自动通过（未接 HITL）；"
                "后续可接入 WebUserProxy 与 config['plan_review_pending']。"
            )
            val.plan_review = PlanReviewResult(
                approved=True,
                modified_plan=val.plan,
                user_feedback=None,
                auto_approved=False,
            )
            val.plan_approved = True

        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="PLAN_APPROVED",
                node_name=node,
                status="finished",
                output_summary=f"auto_approved={val.plan_review.auto_approved}",
            ),
            enabled=enable_trace,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("[plan_review_node] 异常，默认视为已批准以继续主流程")
        append_workflow_error(val, {"node": node, "error": f"{type(e).__name__}: {e}"})
        val.plan_review = PlanReviewResult(
            approved=True,
            modified_plan=None,
            user_feedback=None,
            auto_approved=True,
        )
        val.plan_approved = True
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="PLAN_APPROVED",
                node_name=node,
                status="failed",
                error=str(e),
                output_summary="forced_approve_on_error",
            ),
            enabled=enable_trace,
        )

    return {"value": val}
