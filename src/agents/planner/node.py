"""planner_node：ResearchBrief + BackgroundContext → ResearchPlan。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from langgraph.runtime import Runtime

from src.agents.planner.defaults import default_research_plan
from src.agents.planner.models import ResearchPlan
from src.core.config import config
from src.core.model_client import chat_completion_text, create_default_client
from src.core.state_models import PaperAgentState, PaperRunContext, State
from src.runtime.events import TraceEvent
from src.runtime.state import BackgroundContext, ResearchBrief
from src.runtime.trace_utils import append_trace_event, append_workflow_error
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_PLANNER_PROMPT_PATH = Path(__file__).resolve().parent / "prompt.md"
_PLANNER_INSTRUCTIONS = (
    _PLANNER_PROMPT_PATH.read_text(encoding="utf-8")
    if _PLANNER_PROMPT_PATH.exists()
    else "你是学术 Planner，只输出 ResearchPlan JSON。"
)


def _trace_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_TRACE_EVENTS")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_trace_events", True)


def _extract_json_object(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("响应中未找到 JSON 对象")
    return json.loads(m.group(0))


async def planner_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    val = state["value"]
    run_id = val.run_id
    node = "planner_node"
    enable_trace = _trace_enabled(val)

    append_trace_event(
        val,
        TraceEvent(
            run_id=run_id,
            event_type="NODE_LIFECYCLE",
            node_name=node,
            agent_name="planner_llm",
            status="started",
            input_summary=(val.brief.clarified_topic if val.brief else None),
        ),
        enabled=enable_trace,
    )

    brief = val.brief
    if brief is None:
        brief = ResearchBrief(
            original_query=val.user_request,
            clarified_topic=val.user_request,
            task_type="survey",
            target_audience="technical_report",
            language="zh",
        )
        val.brief = brief
    bg = val.background_context

    user_block = (
        f"{_PLANNER_INSTRUCTIONS}\n\n"
        f"ResearchBrief JSON:\n{brief.model_dump_json(indent=2)}\n\n"
        f"BackgroundContext JSON:\n"
        f"{(bg.model_dump_json(indent=2) if bg else 'null')}\n\n"
        "请只输出一个符合 ResearchPlan schema 的 JSON 对象。"
    )

    try:
        client = create_default_client()
        try:
            raw_text = await chat_completion_text(client, user_block, temperature=0.25, source="planner_node")
        finally:
            close = getattr(client, "close", None)
            if close:
                try:
                    await close()
                except Exception:  # noqa: BLE001
                    logger.debug("planner client close 忽略", exc_info=True)

        plan = ResearchPlan.model_validate(_extract_json_object(raw_text))
        if not plan.search_tasks:
            plan.search_tasks = default_research_plan(brief, bg).search_tasks
        if not plan.reading_tasks:
            plan.reading_tasks = default_research_plan(brief, bg).reading_tasks
        if not plan.analysis_tasks:
            plan.analysis_tasks = default_research_plan(brief, bg).analysis_tasks
        if not plan.writing_tasks:
            plan.writing_tasks = default_research_plan(brief, bg).writing_tasks
        val.plan = plan
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="PLAN_GENERATED",
                node_name=node,
                status="finished",
                output_summary=plan.title[:500],
                metrics={"search_tasks": len(plan.search_tasks), "writing_tasks": len(plan.writing_tasks)},
            ),
            enabled=enable_trace,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("[planner_node] 失败，使用默认 ResearchPlan")
        append_workflow_error(val, {"node": node, "error": f"{type(e).__name__}: {e}"})
        val.plan = default_research_plan(brief, bg)
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="PLAN_GENERATED",
                node_name=node,
                status="failed",
                error=str(e),
                output_summary="fallback_default_plan",
            ),
            enabled=enable_trace,
        )

    return {"value": val}
