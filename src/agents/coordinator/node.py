"""coordinator_node：用户 query → ResearchBrief。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from langgraph.runtime import Runtime

from src.core.config import config
from src.core.state_models import PaperAgentState, PaperRunContext, State
from src.runtime.events import TraceEvent
from src.runtime.state import ResearchBrief
from src.runtime.trace_utils import append_trace_event, append_workflow_error
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent / "prompt.md"
_COORDINATOR_SYSTEM = (
    _PROMPT_PATH.read_text(encoding="utf-8")
    if _PROMPT_PATH.exists()
    else "你是学术调研协调器，只输出 ResearchBrief JSON。"
)

_coordinator_agent = None


def _get_coordinator_agent():
    """延迟导入 AutoGen / model_client，避免仅 import 本模块即依赖 autogen_ext。"""
    global _coordinator_agent
    if _coordinator_agent is None:
        from autogen_agentchat.agents import AssistantAgent

        from src.core.model_client import create_default_client

        _coordinator_agent = AssistantAgent(
            name="coordinator_agent",
            model_client=create_default_client(),
            system_message=_COORDINATOR_SYSTEM,
        )
    return _coordinator_agent
def _trace_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_TRACE_EVENTS")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_trace_events", True)


def _heuristic_brief(user_request: str) -> ResearchBrief:
    ur = (user_request or "").strip()
    vague = len(ur) < 10 or ur.count(" ") < 1 and len(ur) < 24
    return ResearchBrief(
        original_query=ur,
        clarified_topic=ur or "未指定主题",
        domain=None,
        task_type="survey",
        target_audience="technical_report",
        time_range=(None, None),
        language="zh",
        clarification_needed=vague,
        clarification_questions=(
            ["请说明：1) 具体子领域或应用场景 2) 时间范围 3) 期望报告深度（组会/论文 related work/技术说明）"]
            if vague
            else []
        ),
    )


def _extract_json_object(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("响应中未找到 JSON 对象")
    return json.loads(m.group(0))


def _normalize_brief_dict(data: dict, user_request: str) -> dict:
    """补齐 LLM 常漏字段，避免 ResearchBrief 校验失败（不修改 schema 语义）。"""
    d = dict(data or {})
    ur = (user_request or "").strip()
    if not (d.get("original_query") or "").strip():
        d["original_query"] = ur
    if not (d.get("clarified_topic") or "").strip():
        d["clarified_topic"] = (d.get("original_query") or ur or "未指定主题").strip()
    tr = d.get("time_range")
    if tr is None:
        d["time_range"] = (None, None)
    elif isinstance(tr, (list, tuple)) and len(tr) >= 2:
        d["time_range"] = (tr[0], tr[1])
    elif isinstance(tr, (list, tuple)) and len(tr) == 1:
        d["time_range"] = (tr[0], None)
    else:
        d["time_range"] = (None, None)
    if not isinstance(d.get("clarification_questions"), list):
        d["clarification_questions"] = []
    return d


async def coordinator_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    val = state["value"]
    run_id = val.run_id
    enable_trace = _trace_enabled(val)
    node = "coordinator_node"

    append_trace_event(
        val,
        TraceEvent(
            run_id=run_id,
            event_type="NODE_LIFECYCLE",
            node_name=node,
            agent_name="coordinator_agent",
            status="started",
            input_summary=(val.user_request or "")[:500] or None,
        ),
        enabled=enable_trace,
    )

    try:
        task = f"用户原始输入：\n{val.user_request}\n\n请输出符合 schema 的 ResearchBrief JSON。"
        result = await _get_coordinator_agent().run(task=task)
        raw = result.messages[-1].content
        brief: ResearchBrief
        if isinstance(raw, ResearchBrief):
            brief = raw
        else:
            brief = ResearchBrief.model_validate(
                _normalize_brief_dict(_extract_json_object(str(raw)), val.user_request)
            )

        val.brief = brief
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="BRIEF_GENERATED",
                node_name=node,
                agent_name="coordinator_agent",
                status="finished",
                output_summary=(brief.clarified_topic or "")[:500],
            ),
            enabled=enable_trace,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[coordinator_node] 失败，使用启发式 ResearchBrief 回退: %s", e)
        append_workflow_error(
            val,
            {"node": node, "error": f"{type(e).__name__}: {e}"},
        )
        val.brief = _heuristic_brief(val.user_request)
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="BRIEF_GENERATED",
                node_name=node,
                agent_name=None,
                status="failed",
                error=str(e),
                output_summary="fallback_heuristic_brief",
            ),
            enabled=enable_trace,
        )

    return {"value": val}
