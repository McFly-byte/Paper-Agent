# ---------------------------------------------------------------------------
# 编排器：用 LangGraph 定义「检索 → 阅读 → 分析 → 写作 → 报告」的 DAG，并驱动执行
# 状态在节点间通过 State 传递；条件边根据 current_step 与 error 决定下一跳
# ---------------------------------------------------------------------------

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver

from src.core.state_models import PaperAgentState, ExecutionState, NodeError, PaperRunContext
from src.agents.userproxy_agent import WebUserProxyAgent
from src.agents.search_agent import search_node
from src.agents.reading_agent import reading_node
from src.agents.analyse_agent import analyse_node
from src.agents.writing_agent import writing_node
from src.agents.report_agent import report_node
from typing import Dict, Any
from src.core.state_models import BackToFrontData
from src.core.state_models import State
from src.utils.log_utils import setup_logger
from src.evaluation.evaluators import run_evaluation
from src.services.run_tmp_state_store import get_text, get_json, KEY_ANALYSE_RESULTS, KEY_WRITTED_SECTIONS
from langsmith import traceable

import asyncio

logger = setup_logger(__name__)


class PaperAgentOrchestrator:
    """基于 LangGraph 的论文调研工作流编排器：建图 + 条件路由 + 错误节点，run() 时注入初始状态并异步执行图。"""

    def __init__(self, state_queue: asyncio.Queue):
        # 与 main.py 里创建的 asyncio.Queue 是同一个：各节点往这里 put BackToFrontData，前端 SSE 从该 queue 取并推送
        self.state_queue = state_queue
        # 进程内内存 checkpoint：每节点完成后写入，便于调试与后续扩展断点恢复（重启即失效）
        self._checkpointer = InMemorySaver()
        # 在构造时一次性构建并编译图，后续 run() 只做 ainvoke，避免重复建图
        self.graph = self._build_graph()

    async def handle_error_node(self, state: State):
        """错误处理节点：当某业务节点置位了 error 时，条件边会路由到这里。标记为 FAILED 并结束，不抛异常。"""
        # state 即当前图状态，["value"] 是 PaperAgentState；队列与 user_proxy 在 PaperRunContext 中
        current_state = state["value"]
        current_state.current_step = ExecutionState.FAILED
        print(f"Workflow failed at {current_state.current_step}: {current_state.error}")
        # 返回对状态的「更新」：LangGraph 会合并到全局 state，这里只更新 value，保留 queue 等
        return {"value": current_state}

    def condition_handler(self, state: State) -> str:
        """条件路由函数：根据当前步骤与各节点 error 是否为空，决定下一跳是下一个业务节点、END 还是 handle_error_node。"""
        current_state = state["value"]
        err = current_state.error
        current_step = current_state.current_step
        if err.search_node_error is None and current_step == ExecutionState.SEARCHING:
            return "reading_node"
        elif err.reading_node_error is None and current_step == ExecutionState.READING:
            return "analyse_node"
        elif err.analyse_node_error is None and current_step == ExecutionState.ANALYZING:
            return "writing_node"
        elif err.writing_node_error is None and current_step == ExecutionState.WRITING:
            return "report_node"
        elif err.report_node_error is None and current_step == ExecutionState.REPORTING:
            return END  
        else:
            return "handle_error_node"


    def _build_graph(self):
        """构建并编译 LangGraph 工作流：声明状态/配置类型、添加 6 个节点、设置入口与条件边/终点边。"""
        builder = StateGraph(State, context_schema=PaperRunContext)

        builder.add_node("search_node", search_node)
        builder.add_node("reading_node", reading_node)
        builder.add_node("analyse_node", analyse_node)
        builder.add_node("writing_node", writing_node)
        builder.add_node("report_node", report_node)
        builder.add_node("handle_error_node", self.handle_error_node)

        builder.set_entry_point("search_node")

        builder.add_edge(START, "search_node")
        builder.add_conditional_edges("search_node", self.condition_handler)
        builder.add_conditional_edges("reading_node", self.condition_handler)
        builder.add_conditional_edges("analyse_node", self.condition_handler)
        builder.add_conditional_edges("writing_node", self.condition_handler)
        builder.add_conditional_edges("report_node", self.condition_handler)
        builder.add_edge("handle_error_node", END)

        return builder.compile(checkpointer=self._checkpointer)
    

    
    @traceable(
        run_type="chain",
        name="PaperAgent Full Workflow",
        tags=["paper-agent", "langgraph", "evaluation"]
    )
    async def run(
        self,
        user_request: str,
        *,
        run_id: str,
        user_proxy: WebUserProxyAgent,
        max_papers: int = 50,
        knowledge_base_label: str | None = None,
        user_id: str | None = None,
    ):
        """执行完整工作流：构造初始 PaperAgentState，通过 ainvoke 把 state_queue 与初始状态传入图并异步执行。
        集成 LangSmith traceable 追踪，并在结束时执行丰富评估 (analysis/writing/report/rag)。
        """
        logger.info(
            "[工作流][run_id=%s] 开始执行：检索 → 阅读 → 分析 → 写作 → 报告（max_papers=%s）",
            run_id,
            max_papers,
        )
        cfg: dict = {}
        if knowledge_base_label:
            cfg["knowledge_base_label"] = knowledge_base_label
        initial_state = PaperAgentState(
            run_id=run_id,
            user_id=user_id,
            user_request=user_request,
            max_papers=max_papers,
            error=NodeError(),
            config=cfg,
        )

        logger.info("[工作流][run_id=%s] 进入 LangGraph，当前从检索节点启动…", run_id)
        
        # 执行 LangGraph 工作流 (已通过 config 启用 LangSmith tracing)
        final_state = await self.graph.ainvoke(
            {"value": initial_state},
            context=PaperRunContext(state_queue=self.state_queue, user_proxy=user_proxy),
            config={
                "configurable": {"thread_id": run_id},
                # LangSmith 会自动追踪此 run，包含所有子节点和 traceable 函数
            },
        )
        
        # ==================== 丰富评估阶段 (基于 LangSmith 官网教程) ====================
        try:
            current_value = final_state.get("value", initial_state)
            ar = getattr(current_value, "analyse_results", None)
            if not ar:
                ar = await get_text(current_value, KEY_ANALYSE_RESULTS)
            sect = getattr(current_value, "writted_sections", None)
            if not sect:
                loaded = await get_json(current_value, KEY_WRITTED_SECTIONS)
                sect = loaded if isinstance(loaded, list) else None
            outputs = {
                "report_markdown": getattr(current_value, "report_markdown", None),
                "analyse_results": ar,
                "global_analysis": ar,
                "sections": sect,
            }
            
            eval_results = await run_evaluation(
                run_id=run_id,
                state=current_value,
                outputs=outputs,
            )
            
            # 将评估结果写回 state 和前端反馈
            if hasattr(current_value, 'config'):
                current_value.config["evaluation"] = eval_results
                current_value.config["overall_score"] = eval_results.get("overall_score", 0.0)
            
            # 通过 SSE 推送评估结果给前端
            await self.state_queue.put(
                BackToFrontData(
                    step=ExecutionState.COMPLETED,
                    state="evaluation_complete",
                    data={
                        "evaluation": eval_results,
                        "overall_score": eval_results.get("overall_score"),
                        "summary": eval_results.get("summary", "评估完成")
                    }
                )
            )
            
            logger.info(
                "[工作流][run_id=%s] LangSmith 评估完成，整体得分: %.2f", 
                run_id, 
                eval_results.get("overall_score", 0.0)
            )
        except Exception as eval_err:
            logger.warning(f"[工作流][run_id={run_id}] 评估阶段异常 (不影响主流程): {eval_err}")
        
        # 通知前端流程已完全结束（成功或已在 handle_error_node 里标记 FAILED），前端可关闭 SSE 或展示最终状态
        await self.state_queue.put(BackToFrontData(step=ExecutionState.FINISHED, state="finished", data=None))
        
        return final_state

    
# 本地调试
if __name__ == "__main__":
    q = asyncio.Queue()
    _rid = "local-debug-run"
    _proxy = WebUserProxyAgent(f"user_proxy_{_rid}")
    orchestrator = PaperAgentOrchestrator(state_queue=q)
    asyncio.run(
        orchestrator.run(
            "帮我写一篇有关 llm 在无人驾驶方面的调研报告。",
            run_id=_rid,
            user_proxy=_proxy,
        )
    )

    