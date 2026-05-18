import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录（agents目录）
src_parent_dir = os.path.dirname(os.path.dirname(current_dir))  # 向上两级找到 Paper-Agents 目录

# 将路径添加到 Python 搜索路径
sys.path.append(src_parent_dir)


from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Sequence, get_type_hints, TypeAlias
from autogen_agentchat.agents import BaseChatAgent
import asyncio

from starlette.routing import Route
from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from src.agents.reading_agent import ExtractedPapersData,KeyMethodology,ExtractedPaperData
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage,StructuredMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken, RoutedAgent
from src.agents.sub_analyse_agent.cluster_agent import PaperClusterAgent
from src.agents.sub_analyse_agent.deep_analyse_agent import DeepAnalyseAgent
from src.agents.sub_analyse_agent.global_analyse_agent import GlobalanalyseAgent
from src.core.model_client import create_default_client
from src.core.config import config
from src.utils.llm_api_throttle import (
    coarse_token_estimate,
    configured_token_cap,
    get_throttler_for_client_type,
)
from src.core.state_models import BackToFrontData
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt, before_sleep_log
import json
import logging

from langgraph.runtime import Runtime

from src.core.state_models import State, ExecutionState, PaperRunContext
from autogen_core import message_handler
from src.services.run_tmp_state_store import get_json, put_text, KEY_EXTRACTED_DATA, KEY_ANALYSE_RESULTS
from src.core.node_gates import gate_analyse, record_gate
from src.domain.paper.plan_coverage import build_plan_coverage_report
from src.core.workflow_recovery import (
    clear_recovery_feedback_for,
    format_recovery_user_block,
    set_recovery_target,
)

logger = setup_logger(__name__)
# BaseChatAgent
class AnalyseAgent(BaseChatAgent):
    """基于AutoGen框架的论文分析智能体"""
    
    def __init__(
        self,
        name: str = "analyse_agent",
        state_queue: asyncio.Queue | None = None,
        *,
        analysis_recovery_hint: str = "",
    ):
        super().__init__(name, "A simple agent that counts down.")
        """初始化论文分析系列智能体"""
        # 创建聚类智能体
        self.cluster_agent = PaperClusterAgent()
        # 创建深度分析智能体
        self.deep_analyse_agent = DeepAnalyseAgent()
        # 创建全局分析智能体
        self.global_analyse_agent = GlobalanalyseAgent()

        self.model_client = create_default_client()
        self.state_queue = state_queue
        self.analysis_recovery_hint = (analysis_recovery_hint or "").strip()
    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    # @message_handler
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """处理分析消息并返回响应
        
        Args:
            message: 提取的论文数据
            cancellation_token: 取消令牌
            
        Returns:
            Response: 包含分析结果的响应对象
        """
        # Calls the on_messages_stream.
        response: Response | None = None
        stream_message = messages[-1].content
        # async for msg in self.on_messages_stream(stream_message, cancellation_token):
        #     if isinstance(msg, Response):
        #         response = msg

        # 下层还是调on_messages_stream() 等起输出完成后一次性返回
        response = await self.on_messages_stream(stream_message, cancellation_token) 
        assert response is not None
        return response

    # @message_handler
    async def on_messages_stream(self, message: ExtractedPapersData, cancellation_token: CancellationToken) -> Any:
        """流式处理分析消息
        
        Args:
            message: 提取的论文数据
            cancellation_token: 取消令牌
            
        Yields:
            生成分析过程中的事件或消息
            AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]
        """
        # 1. 聚类
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data="正在进行论文聚类分析\n"))
        logger.info("[工作流·分析] 聚类阶段（嵌入 + KMeans 等，可能较慢）…")
        cluster_results = await self.cluster_agent.run(message) # PaperCluster列表
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data=f"论文聚类分析完成，共形成 {len(cluster_results)} 个聚类\n"))

        # 2. 深度分析每个聚类的论文
        deep_analysis_results = []
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data="正在进行论文深度分析\n"))
        sem_n = max(1, config.get_int("llm_remote_rate_limit.max_concurrent_llm_tasks", 2))
        llm_throttle = get_throttler_for_client_type("subanalyse-deep-analyse-model")
        logger.info(
            "[工作流·分析] 深度分析 %s 个聚类（LLM，并发 %s）…",
            len(cluster_results),
            sem_n,
        )
        semaphore = asyncio.Semaphore(sem_n)

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential(multiplier=10, min=15, max=120),
            stop=stop_after_attempt(5),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def analyse_single(cluster):
            return await self.deep_analyse_agent.run(cluster)

        n_cluster = len(cluster_results)

        # 每个 cluster 包装一个协程，并发执行；每占用/释放槽位打日志便于观感
        async def analyse_with_limit(cluster, idx: int):
            async with semaphore:
                logger.info(
                    "[工作流·分析] 深度分析进行中：簇 %s/%s（并发槽已占用）",
                    idx + 1,
                    n_cluster,
                )
                try:
                    if llm_throttle:
                        try:
                            blob = json.dumps(
                                getattr(cluster, "papers", []), ensure_ascii=False
                            )
                        except (TypeError, ValueError):
                            blob = str(cluster)
                        oh = config.get_int(
                            "llm_remote_rate_limit.deep_analyse_overhead_tokens", 4000
                        )
                        est = coarse_token_estimate(
                            blob[:15000], overhead=oh, cap=configured_token_cap()
                        )
                        await llm_throttle.acquire(1, est)
                    out = await analyse_single(cluster)
                    logger.info("[工作流·分析] 深度分析完成：簇 %s/%s", idx + 1, n_cluster)
                    return out
                except Exception:
                    logger.exception("[工作流·分析] 深度分析失败：簇 %s/%s", idx + 1, n_cluster)
                    raise

        deep_analysis_results = await asyncio.gather(
            *[analyse_with_limit(c, i) for i, c in enumerate(cluster_results)]
        )
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data="论文深度分析完成\n"))
        
        # 3. 调用全局分析智能体生成整体分析报告
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data="等待全局分析\n"))
        logger.info("[工作流·分析] 全局综合分析（LLM，非流式；完成后一次性推送）…")
        global_analysis: dict | None = None
        is_thinking = None
        # 不在收到首个成功 Dict 后 break，以免提前关闭异步生成器链；全局侧已改为单次 yield + run() 完成后再产出
        async for chunk in self.global_analyse_agent.run(
            deep_analysis_results,
            recovery_hint=self.analysis_recovery_hint,
        ):
            if isinstance(chunk, Dict):
                if not chunk.get("isSuccess", False):
                    err_text = chunk.get("global_analyse", "Unknown error")
                    await self.state_queue.put(
                        BackToFrontData(
                            step=ExecutionState.ANALYZING,
                            state="error",
                            data=err_text,
                        )
                    )
                    return Response(
                        chat_message=TextMessage(
                            content=json.dumps(
                                {"isSuccess": False, "global_analyse": err_text},
                                ensure_ascii=False,
                                indent=2,
                            ),
                            source=self.name,
                        )
                    )
                global_analysis = chunk
                await self.state_queue.put(
                    BackToFrontData(
                        step=ExecutionState.ANALYZING,
                        state="thinking",
                        data="全局分析正文已生成，正在汇总…\n",
                    )
                )
                continue
            state, is_thinking = handlerChunk(is_thinking, chunk)
            if state is None:
                continue
            await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING, state=state, data=chunk))

        if global_analysis is None:
            err = "全局分析未返回有效结果"
            await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING, state="error", data=err))
            return Response(
                chat_message=TextMessage(
                    content=json.dumps({"isSuccess": False, "global_analyse": err}, ensure_ascii=False, indent=2),
                    source=self.name,
                )
            )

        return Response(
            chat_message=TextMessage(
                content=json.dumps(global_analysis, ensure_ascii=False, indent=2),
                source=self.name,
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

async def analyse_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    """搜索论文节点"""
    try:
        state_queue = runtime.context.state_queue
        current_state = state["value"]
        current_state.current_step = ExecutionState.ANALYZING
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="initializing",data=None))
        extracted_papers = current_state.extracted_data
        if extracted_papers is None or not extracted_papers.papers:
            raw = await get_json(current_state, KEY_EXTRACTED_DATA)
            if raw is not None:
                extracted_papers = ExtractedPapersData.model_validate(raw)
            elif extracted_papers is None:
                extracted_papers = ExtractedPapersData(papers=[])
        n_papers = len(extracted_papers.papers) if extracted_papers and extracted_papers.papers else 0
        logger.info("[工作流·分析] 启动分析子流程，输入论文数：%s", n_papers)

        recovery_block = format_recovery_user_block(current_state.config or {}, "analyse")
        analyse_agent = AnalyseAgent(
            state_queue=state_queue,
            analysis_recovery_hint=recovery_block,
        )
        task = StructuredMessage(content=extracted_papers, source="User") # 把对象转成结构化消息
        # task = TextMessage(content=json.dumps(extracted_papers.model_dump(),ensure_ascii=False), source="User")
        response = await analyse_agent.run(task=task)

        analyse_results = response.messages[-1].content
        
        await put_text(current_state, KEY_ANALYSE_RESULTS, analyse_results)
        current_state.analyse_results = None

        ag = gate_analyse(
            analyse_results,
            input_paper_count=n_papers,
            parsed_paper_titles=[
                str(getattr(p, "core_problem", "") or "")[:160]
                for p in (extracted_papers.papers if extracted_papers else [])
            ],
        )
        try:
            plan_cov = build_plan_coverage_report(
                plan=current_state.plan,
                filter_report=(current_state.config or {}).get("paper_filter_report"),
                analyse_results=analyse_results,
                evidence_ledger=current_state.evidence_ledger,
            )
            ag.metrics["plan_analysis_task_coverage_ratio"] = plan_cov.analysis_task_coverage_ratio
            ag.metrics["missing_plan_dimensions"] = {
                "analysis_tasks": plan_cov.missing_plan_dimensions.get("analysis_tasks", [])
            }
            if plan_cov.representative_papers:
                ag.metrics["representative_papers"] = plan_cov.representative_papers[:8]
            current_state.config = dict(current_state.config or {})
            current_state.config["plan_coverage_report_partial"] = plan_cov.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[工作流·分析] plan coverage 统计失败（忽略）: %s", exc)
        current_state.boundary_checks = record_gate(current_state.boundary_checks, "analyse", ag)
        if not ag.passed:
            detail = "；".join(ag.reasons) if ag.reasons else "分析门禁未通过"
            current_state.error.analyse_node_error = detail
            set_recovery_target(current_state, "analyse")
            await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING, state="error", data=detail))
            return {"value": current_state}

        # 尝试解析 JSON 并只提取 global_analyse 字段发送给前端，避免显示杂乱的 JSON 数据
        display_content = analyse_results
        try:
            data_obj = json.loads(analyse_results) # 解析成JSON对象
            if isinstance(data_obj, dict) and "global_analyse" in data_obj: # 如果是字典且包含global_analyse字段
                 display_content = data_obj["global_analyse"] # 只提取global_analyse字段，用来显示给前端
        except Exception:
            pass # 如果解析失败，就还是保持原样，或者可以改为发送简短提示
             
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="completed",data=display_content))
        
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="completed",data=analyse_results))

        clear_recovery_feedback_for(current_state, "analyse")
        return {"value": current_state}
            
    except Exception as e:
        err_msg = f"Analyse failed: {str(e)}"
        vs = state["value"]
        vs.error.analyse_node_error = err_msg
        set_recovery_target(vs, "analyse")
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="error",data=err_msg))
        return {"value": vs}

def main():
    """主函数"""
    asyncio.run(analyse_node(state))

if __name__ == "__main__":
    pass
    # from src.core.state_models import PaperAgentState,NodeError
    # state_queue = asyncio.Queue()
    # initial_state = PaperAgentState(
    #         user_request="帮我写一篇关于人工智能的调研报告",
    #         max_papers=2,
    #         error=NodeError(),
    #         config={}  # 可以传入各种配置
    #     )
    # state = {"state_queue": state_queue, "value": initial_state}
    # analyse_agent = AnalyseAgent()
    # main()
