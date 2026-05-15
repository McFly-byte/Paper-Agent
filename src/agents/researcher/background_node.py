"""background_investigation_node：ResearchBrief → BackgroundContext（可无联网）。"""

from __future__ import annotations

import json
import os
import re

from autogen_agentchat.agents import AssistantAgent
from langgraph.runtime import Runtime

from src.core.config import config
from src.core.model_client import create_default_client
from src.core.state_models import PaperAgentState, PaperRunContext, State
from src.runtime.events import TraceEvent
from src.runtime.state import BackgroundContext, ResearchBrief
from src.runtime.trace_utils import append_trace_event, append_workflow_error
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_bg_client = create_default_client()
_background_system = (
    "你是学术文献调研中的背景调查助手。基于给定的 ResearchBrief（JSON），"
    "输出 BackgroundContext JSON：expanded_keywords、related_terms、initial_findings（短句列表）、"
    "suggested_search_queries（适合 arXiv 的英文关键词或短语，勿包含中文）、notes。"
    "不要检索网络；不要列出具体论文标题；不要编造实验结果。"
)

background_agent = AssistantAgent(
    name="background_investigation_agent",
    model_client=_bg_client,
    system_message=_background_system,
    output_content_type=BackgroundContext,
)


def _trace_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_TRACE_EVENTS")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_trace_events", True)


def _bg_enabled(val: PaperAgentState) -> bool:
    raw = os.environ.get("PAPER_AGENT_ENABLE_BACKGROUND_INVESTIGATION")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes")
    return config.get_bool("workflow_v2.enable_background_investigation", True)


def _rule_based_background(brief: ResearchBrief) -> BackgroundContext:
    topic = (brief.clarified_topic or brief.original_query or "").strip() or "未命名主题"
    tokens = [t for t in re.split(r"[\s,，;；]+", topic) if len(t) >= 2][:10]
    queries = []
    for t in tokens[:3]:
        if not re.search(r"[\u4e00-\u9fff]", t):
            queries.append(t)
    if not queries:
        queries.append("survey review benchmark")
    return BackgroundContext(
        topic=topic,
        expanded_keywords=tokens or [topic[:32]],
        related_terms=[],
        initial_findings=[f"围绕「{topic}」整理检索关键词与检索式建议（规则生成）。"],
        suggested_search_queries=queries,
        notes="phase1_rule_based_fallback",
    )


def _extract_json_object(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("响应中未找到 JSON 对象")
    return json.loads(m.group(0))


async def background_investigation_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    val: PaperAgentState = state["value"]
    run_id = val.run_id
    node = "background_investigation_node"
    enable_trace = _trace_enabled(val)

    append_trace_event(
        val,
        TraceEvent(
            run_id=run_id,
            event_type="NODE_LIFECYCLE",
            node_name=node,
            agent_name="background_investigation_agent",
            status="started",
            input_summary=(val.brief.clarified_topic if val.brief else val.user_request)[:500]
            if val.brief or val.user_request
            else None,
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

    if not _bg_enabled(val):
        val.background_context = _rule_based_background(brief)
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="BACKGROUND_INVESTIGATED",
                node_name=node,
                status="finished",
                output_summary="skipped_llm_rule_based",
            ),
            enabled=enable_trace,
        )
        return {"value": val}

    try:
        task = f"ResearchBrief:\n{brief.model_dump_json(indent=2)}\n\n请输出 BackgroundContext JSON。"
        result = await background_agent.run(task=task)
        raw = result.messages[-1].content
        if isinstance(raw, BackgroundContext):
            val.background_context = raw
        else:
            val.background_context = BackgroundContext.model_validate(_extract_json_object(str(raw)))
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="BACKGROUND_INVESTIGATED",
                node_name=node,
                agent_name="background_investigation_agent",
                status="finished",
                output_summary=",".join(val.background_context.expanded_keywords[:8])[:500],
            ),
            enabled=enable_trace,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("[background_investigation_node] 失败，规则回退")
        append_workflow_error(val, {"node": node, "error": f"{type(e).__name__}: {e}"})
        val.background_context = _rule_based_background(brief)
        append_trace_event(
            val,
            TraceEvent(
                run_id=run_id,
                event_type="BACKGROUND_INVESTIGATED",
                node_name=node,
                status="failed",
                error=str(e),
                output_summary="fallback_rule_based",
            ),
            enabled=enable_trace,
        )

    return {"value": val}
