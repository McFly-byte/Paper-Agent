from src.agents.sub_writing_agent.writing_state_models import WritingState, SectionState
from typing import Dict, Any
from src.agents.sub_writing_agent.writing_chatGroup import create_writing_group
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage,StructuredMessage,ModelClientStreamingChunkEvent,ThoughtEvent,ToolCallSummaryMessage,ToolCallExecutionEvent
from autogen_agentchat.base import TaskResult
from src.core.state_models import BackToFrontData,ExecutionState
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from src.utils.log_utils import setup_logger
import asyncio
import logging

logger = setup_logger(__name__)


async def parallel_writing_node(state: WritingState) -> Dict[str, Any]:
        """并行执行所有子任务"""
        
        async def run_single_subtask(task: Dict):
            nonlocal state # 声明state为非局部变量
            """执行单个子任务"""            
            task_prompt = f"""请根据以下内容完成写作任务：
                用户的请求是：{task['user_request']}
                当前写作子任务: {task['section']}
                论文全局分析: {task['global_analyse']}

                请开始写作：
            """
            is_thinking = False
            cur_source = "user" # 当前消息来源
            # 过滤掉非 writing_agent 的消息，保持界面整洁
            agent_sources = {"writing_agent", "retrieval_agent"}
            try:
                task_group = create_writing_group() # 创建写作组
                task_group.reset()
                async for chunk in task_group.run_stream(task=task_prompt):  
                    if isinstance(chunk, TaskResult): # 跳过最终封装TaskResult，只处理中间事件流
                        continue
                    if chunk.source == "user": # 跳过用户消息（“谁在发言”，重复的上下文信息）
                        continue
                    if chunk.type == "TextMessage" and chunk.source == "writing_agent":
                        # 写作结果不应覆盖任务列表 sections ，应写入 writted_sections ，与后续汇总逻辑一致，避免结构被污染。
                        # state["sections"][task["index"]] = chunk.content
                        state["writted_sections"][task["index"]].content = chunk.content
                        continue
                    # if cur_source != chunk.source:
                    if cur_source != chunk.source and chunk.source in agent_sources: # agent切换时，向前端发送标记
                        cur_source = chunk.source
                        # self.name未定义，消息来源应来自 chunk.source ，并维持前端展示用的 1-based step 编号，确保显示一致
                        # str1,str2,str3 = "="*40, self.name, "="*40
                        str1,str2,str3 = "="*40, chunk.source, "="*40
                        splitStr = str1+str2+str3+"\n"
                        # await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"]),state="generating",data=splitStr))
                        await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="generating",data=splitStr))
                    if chunk.type == "ModelClientStreamingChunkEvent": # token级流式输出
                        if '<think>' in chunk.content:
                            is_thinking = True
                        elif '</think>' in chunk.content:
                            is_thinking = False
                            continue
                        if not is_thinking:
                            print(chunk.content,end="")
                            # 统一前端显示索引为 1-based，避免分段进度显示错位。
                            # await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"]),state="generating",data=chunk.content))
                            await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="generating",data=chunk.content))
                    if chunk.type == "ToolCallSummaryMessage": # 推送工具调用
                        # await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"]),state="generating",data=chunk.content))
                        await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="generating",data=chunk.content))

                # 补发completed结束撰写部分
                await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="completed",data=None))
            except Exception as e:
                # 此处应该有重试机制
                # 异常上报使用 1-based step，避免 UI 误标分段编号。
                # await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"]),state="error",data=f"Section writing failed: {str(e)}"))
                await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(task["index"] + 1),state="error",data=f"Section writing failed: {str(e)}"))
                # return state
        
        # 并行执行所有子任务
        # 预先初始化写作结果槽位，避免并行写入时访问不存在的索引导致越界。
        state_queue=state["state_queue"]
        global_analyse = state["global_analysis"]
        user_request = state["user_request"]
        sections = state["sections"]
        # 确保writted_sections槽位存在且长度与sections一致
        if state.get("writted_sections") is None or len(state["writted_sections"]) != len(sections):
            # state["writted_sections"] = [None for _ in sections]
            state["writted_sections"] = [SectionState() for _ in sections]
        subtasks = [] # 子任务列表
        for i in range(len(sections)):
            await state_queue.put(BackToFrontData(step=ExecutionState.SECTION_WRITING+"_"+str(i+1),state="initializing",data=None))
            dic = {
                "user_request": user_request,
                "global_analyse": global_analyse,
                "section": sections[i],
                # 列表下标为 0-based，使用 i+1 会导致 list assignment index out of range
                # "index": i+1
                "index": i
            }
            subtasks.append(dic)
        semaphore = asyncio.Semaphore(2)

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
                return await write_single_subtask(task)

        tasks = [write_with_limit(task) for task in subtasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return state
