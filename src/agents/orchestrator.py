# ---------------------------------------------------------------------------
# 编排器：用 LangGraph 定义「检索 → 阅读 → 分析 → 写作 → 报告」的 DAG，并驱动执行
# 状态在节点间通过 State 传递；条件边根据 current_step 与 error 决定下一跳
# ---------------------------------------------------------------------------

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Any

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver

from src.core.state_models import PaperAgentState, ExecutionState, NodeError, PaperRunContext
from src.agents.userproxy_agent import WebUserProxyAgent
from src.agents.coordinator.node import coordinator_node
from src.agents.planner.node import planner_node
from src.agents.planner.review_node import plan_review_node
from src.agents.reading_agent import reading_node
from src.agents.researcher.background_node import background_investigation_node
from src.agents.researcher.paper_filter_node import paper_filter_node
from src.agents.researcher.evidence_index_node import evidence_index_node
from src.agents.search_plan_adapter import search_node_with_plan
from src.agents.analyse_agent import analyse_node
from src.agents.writing_agent import writing_node
from src.agents.report_agent import report_node
from src.agents.workflow_recovery_node import workflow_recovery_node
from src.core.state_models import BackToFrontData, State
from src.core.workflow_recovery import should_abort_to_terminal_error
from src.utils.log_utils import setup_logger
from src.evaluation.evaluators import run_evaluation
from src.services.run_tmp_state_store import get_text, get_json, KEY_ANALYSE_RESULTS, KEY_WRITTED_SECTIONS
from langsmith import traceable

import asyncio

logger = setup_logger(__name__)

_ERR_ATTRS = (
    "search_node_error",
    "paper_filter_node_error",
    "reading_node_error",
    "evidence_index_node_error",
    "analyse_node_error",
    "writing_node_error",
    "report_node_error",
    "error",
)


def _nonempty_error_field(val: Any) -> bool:
    return val is not None and str(val).strip() != ""


def _format_node_errors_for_user(err: NodeError | None) -> str:
    if err is None:
        return "未知错误"
    parts: list[str] = []
    for name in _ERR_ATTRS:
        raw = getattr(err, name, None)
        if not _nonempty_error_field(raw):
            continue
        label = name.replace("_node_error", "").replace("_", " ")
        parts.append(f"[{label}] {str(raw).strip()}")
    return "；".join(parts) if parts else "未知错误"


def _route_after_node(state: State, *, node_key: str, err_attr: str, ok_next: str) -> str:
    """失败时：未用尽恢复次数 → recovery；否则 → handle_error。成功时 → ok_next。"""
    val = state["value"]
    err = val.error
    if err is not None and _nonempty_error_field(getattr(err, err_attr, None)):
        if should_abort_to_terminal_error(val, node_key):
            return "handle_error_node"
        return "workflow_recovery_node"
    return ok_next


def route_after_search(state: State) -> str:
    return _route_after_node(state, node_key="search", err_attr="search_node_error", ok_next="paper_filter_node")


def route_after_paper_filter(state: State) -> str:
    return _route_after_node(
        state, node_key="paper_filter", err_attr="paper_filter_node_error", ok_next="reading_node"
    )


def route_after_reading(state: State) -> str:
    return _route_after_node(state, node_key="reading", err_attr="reading_node_error", ok_next="evidence_index_node")


def route_after_evidence_index(state: State) -> str:
    return _route_after_node(
        state, node_key="evidence_index", err_attr="evidence_index_node_error", ok_next="analyse_node"
    )


def route_after_analyse(state: State) -> str:
    return _route_after_node(state, node_key="analyse", err_attr="analyse_node_error", ok_next="writing_node")


def route_after_writing(state: State) -> str:
    return _route_after_node(state, node_key="writing", err_attr="writing_node_error", ok_next="report_node")


def route_after_report(state: State) -> str:
    return _route_after_node(state, node_key="report", err_attr="report_node_error", ok_next=END)


def route_after_recovery(state: State) -> str:
    nxt = (state["value"].config or {}).get("retry_next_node")
    if isinstance(nxt, str) and nxt.strip():
        return nxt.strip()
    logger.warning("[工作流] recovery 后缺少 retry_next_node，转入 handle_error_node")
    return "handle_error_node"


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
        """错误处理节点：任意主节点已写入非空 error 时由条件边进入。置 FAILED、推送 SSE、终止后续 LangGraph 边。"""
        current_state = state["value"]
        err = current_state.error
        failed_at_step = current_state.current_step
        summary = _format_node_errors_for_user(err)
        logger.error(
            "[工作流] 已进入错误处理节点：失败前步骤=%s，详情=%s",
            failed_at_step,
            summary,
        )
        try:
            await self.state_queue.put(
                BackToFrontData(
                    step=ExecutionState.FAILED,
                    state="failed",
                    data={"failed_at": str(failed_at_step), "message": summary},
                )
            )
        except Exception as qe:  # noqa: BLE001
            logger.warning("错误节点推送 SSE 失败（忽略）: %s", qe)
        current_state.current_step = ExecutionState.FAILED
        return {"value": current_state}

    def _build_graph(self):
        """构建并编译 LangGraph 工作流：声明状态/配置类型、添加 6 个节点、设置入口与条件边/终点边。"""
        builder = StateGraph(State, context_schema=PaperRunContext)

        builder.add_node("coordinator_node", coordinator_node)
        builder.add_node("background_investigation_node", background_investigation_node)
        builder.add_node("planner_node", planner_node)
        builder.add_node("plan_review_node", plan_review_node)
        builder.add_node("search_node", search_node_with_plan)
        builder.add_node("paper_filter_node", paper_filter_node)
        builder.add_node("reading_node", reading_node)
        builder.add_node("evidence_index_node", evidence_index_node)
        builder.add_node("analyse_node", analyse_node)
        builder.add_node("writing_node", writing_node)
        builder.add_node("report_node", report_node)
        builder.add_node("workflow_recovery_node", workflow_recovery_node)
        builder.add_node("handle_error_node", self.handle_error_node)

        builder.add_edge(START, "coordinator_node")
        builder.add_edge("coordinator_node", "background_investigation_node")
        builder.add_edge("background_investigation_node", "planner_node")
        builder.add_edge("planner_node", "plan_review_node")
        builder.add_edge("plan_review_node", "search_node")
        builder.add_conditional_edges("search_node", route_after_search)
        builder.add_conditional_edges("paper_filter_node", route_after_paper_filter)
        builder.add_conditional_edges("reading_node", route_after_reading)
        builder.add_conditional_edges("evidence_index_node", route_after_evidence_index)
        builder.add_conditional_edges("analyse_node", route_after_analyse)
        builder.add_conditional_edges("writing_node", route_after_writing)
        builder.add_conditional_edges("report_node", route_after_report)
        builder.add_conditional_edges("workflow_recovery_node", route_after_recovery)
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

        current_value = final_state.get("value", initial_state)
        if getattr(current_value, "current_step", None) == ExecutionState.FAILED:
            logger.error(
                "[工作流][run_id=%s] 已失败终止，跳过 LangSmith 评估。%s",
                run_id,
                _format_node_errors_for_user(getattr(current_value, "error", None)),
            )
            await self.state_queue.put(
                BackToFrontData(step=ExecutionState.FINISHED, state="finished", data=None)
            )
            if os.environ.get("PAPER_AGENT_EXIT_ON_WORKFLOW_ERROR", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                logger.critical(
                    "[工作流] PAPER_AGENT_EXIT_ON_WORKFLOW_ERROR 已启用，进程退出码 1"
                )
                sys.exit(1)
            return final_state

        # ==================== 丰富评估阶段 (基于 LangSmith 官网教程) ====================
        try:
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

    