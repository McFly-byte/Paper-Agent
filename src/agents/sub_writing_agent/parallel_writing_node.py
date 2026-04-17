import json
import re
from typing import Any, Dict, Optional

from langgraph.runtime import Runtime

from src.agents.sub_writing_agent.writing_state_models import (
    WritingState,
    SectionState,
    WritingRunContext,
    ReviewDecision,
)
from src.agents.sub_writing_agent.writing_chatGroup import create_writing_group
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage,StructuredMessage,ModelClientStreamingChunkEvent,ThoughtEvent,ToolCallSummaryMessage,ToolCallExecutionEvent
from autogen_agentchat.base import TaskResult
from src.core.state_models import BackToFrontData,ExecutionState
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from src.utils.log_utils import setup_logger
from src.core.config import config
from src.core.run_context import rag_writing_section_hint_var
from src.utils.llm_api_throttle import get_throttler_for_client_type
import asyncio

logger = setup_logger(__name__)


def _capture_review_for_section(idx: int, content: Any, state: WritingState) -> None:
    """将 review_agent 的结构化结论写入 SectionState，供 writing 门禁统计。"""
    sec = state["writted_sections"][idx]
    rd: Optional[ReviewDecision] = None
    if isinstance(content, ReviewDecision):
        rd = content
    elif isinstance(content, dict):
        try:
            rd = ReviewDecision.model_validate(content)
        except Exception:
            rd = None
    elif isinstance(content, str):
        s = content.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        try:
            rd = ReviewDecision.model_validate_json(s)
        except Exception:
            try:
                rd = ReviewDecision.model_validate(json.loads(s))
            except Exception:
                rd = None
    if rd is None:
        return
    sec.review_verdict = rd.verdict
    sec.review_summary = (rd.summary or "")[:2000]


async def parallel_writing_node(
    state: WritingState, runtime: Runtime[WritingRunContext]
) -> Dict[str, Any]:
    """并行执行所有子任务"""

    state_queue = runtime.context.state_queue
    global_analyse = state["global_analysis"]
    user_request = state["user_request"]
    sections = state["sections"]
    n_sections = len(sections)

    if state.get("writted_sections") is None or len(state["writted_sections"]) != len(sections):
        state["writted_sections"] = [SectionState() for _ in sections]

    async def run_single_subtask(task: Dict):
        nonlocal state
        sec_i = task["index"] + 1
        _wf = task.get("writing_recovery") or ""
        task_prompt = f"""请根据以下内容完成写作任务：
                用户的请求是：{task['user_request']}
                当前写作子任务: {task['section']}
                论文全局分析: {task['global_analyse']}
                {_wf}

                请开始写作：
            """
        is_thinking = False
        cur_source = "user"
        agent_sources = {"writing_agent", "retrieval_agent", "review_agent"}
        sec_hint = str(task.get("section") or "")
        tok_sec = rag_writing_section_hint_var.set(sec_hint or None)
        try:
            logger.info(
                "[工作流·写作] 章节进行中：第 %s/%s（并发槽已占用）",
                sec_i,
                n_sections,
            )
            task_group = create_writing_group()
            await task_group.reset()
            async for chunk in task_group.run_stream(task=task_prompt):
                if isinstance(chunk, TaskResult):
                    continue
                if chunk.source == "user":
                    continue
                src = getattr(chunk, "source", None)
                if src and src != "user" and src != cur_source:
                    cur_source = src
                    ev = getattr(chunk, "type", None) or type(chunk).__name__
                    logger.info(
                        "[工作流·写作] 第 %s/%s 章 · 发言切换 → %s（事件：%s）",
                        sec_i,
                        n_sections,
                        src,
                        ev,
                    )
                    if src in agent_sources:
                        str1, str2, str3 = "=" * 40, src, "=" * 40
                        split_str = str1 + str2 + str3 + "\n"
                        await state_queue.put(
                            BackToFrontData(
                                step=ExecutionState.SECTION_WRITING + "_" + str(task["index"] + 1),
                                state="generating",
                                data=split_str,
                            )
                        )
                if chunk.type == "TextMessage" and chunk.source == "writing_agent":
                    state["writted_sections"][task["index"]].content = chunk.content
                    continue
                if chunk.source == "review_agent":
                    if isinstance(chunk, StructuredMessage):
                        _capture_review_for_section(task["index"], chunk.content, state)
                        continue
                    if chunk.type == "TextMessage":
                        _capture_review_for_section(task["index"], chunk.content, state)
                        continue
                if chunk.type == "ModelClientStreamingChunkEvent":
                    if '<think>' in chunk.content:
                        is_thinking = True
                    elif '</think>' in chunk.content:
                        is_thinking = False
                        continue
                    if not is_thinking:
                        print(chunk.content,end="")
                        await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="generating",data=chunk.content))
                if chunk.type == "ToolCallSummaryMessage":
                    await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="generating",data=chunk.content))

            await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="completed",data=None))
            logger.info("[工作流·写作] 章节完成：第 %s/%s", sec_i, n_sections)
        except Exception as e:
            await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="error",data=f"Section writing failed: {str(e)}"))
        finally:
            rag_writing_section_hint_var.reset(tok_sec)

    _writing_recovery = (state.get("recovery_feedback") or "").strip()

    subtasks = []
    for i in range(len(sections)):
        await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(i+1),state="initializing",data=None))
        subtasks.append({
            "user_request": user_request,
            "global_analyse": global_analyse,
            "section": sections[i],
            "index": i,
            "writing_recovery": _writing_recovery,
        })

    sec_n = config.get_int(
        "llm_remote_rate_limit.max_concurrent_writing_sections",
        config.get_int("llm_remote_rate_limit.max_concurrent_llm_tasks", 2),
    )
    semaphore = asyncio.Semaphore(max(1, sec_n))
    llm_throttle = get_throttler_for_client_type("subwriting-writing-model")

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=10, min=15, max=120),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def write_single_subtask(task):
        return await run_single_subtask(task)

    async def write_with_limit(task):
        async with semaphore:
            if llm_throttle:
                w_est = config.get_int(
                    "llm_remote_rate_limit.writing_section_estimated_tokens", 50000
                )
                await llm_throttle.acquire(1, w_est)
            return await write_single_subtask(task)

    tasks = [write_with_limit(task) for task in subtasks]
    await asyncio.gather(*tasks, return_exceptions=True)

    return state
