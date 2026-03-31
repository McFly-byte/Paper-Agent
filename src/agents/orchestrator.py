# ---------------------------------------------------------------------------
# 编排器：用 LangGraph 定义「检索 → 阅读 → 分析 → 写作 → 报告」的 DAG，并驱动执行
# 状态在节点间通过 State 传递；条件边根据 current_step 与 error 决定下一跳
# ---------------------------------------------------------------------------

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END, START  
from src.core.state_models import PaperAgentState, ExecutionState, NodeError
from src.agents.search_agent import search_node
from src.agents.reading_agent import reading_node
from src.agents.analyse_agent import analyse_node
from src.agents.writing_agent import writing_node
from src.agents.report_agent import report_node
from typing import Dict, Any
from src.core.state_models import BackToFrontData
from src.core.state_models import State, ConfigSchema

import asyncio


class PaperAgentOrchestrator:
    """基于 LangGraph 的论文调研工作流编排器：建图 + 条件路由 + 错误节点，run() 时注入初始状态并异步执行图。"""

    def __init__(self, state_queue: asyncio.Queue):
        # 与 main.py 里创建的 asyncio.Queue 是同一个：各节点往这里 put BackToFrontData，前端 SSE 从该 queue 取并推送
        self.state_queue = state_queue
        # 在构造时一次性构建并编译图，后续 run() 只做 ainvoke，避免重复建图
        self.graph = self._build_graph()

    async def handle_error_node(self, state: State):
        """错误处理节点：当某业务节点置位了 error 时，条件边会路由到这里。标记为 FAILED 并结束，不抛异常。"""
        # state 即当前图状态，["value"] 是 PaperAgentState，["state_queue"] 是 asyncio.Queue
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
        builder = StateGraph(State, context_schema=ConfigSchema) 

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

        return builder.compile()
    

    
    async def run(self, user_request: str, max_papers: int = 50):
        """执行完整工作流：构造初始 PaperAgentState，通过 ainvoke 把 state_queue 与初始状态传入图并异步执行，结束时向 queue 放入 FINISHED。"""
        print("Starting workflow...")
        # 初始状态：只有用户输入、数量上限、空错误；各节点会按顺序填充 search_results、paper_contents、extracted_data、analyse_results、writted_sections、report_markdown
        initial_state = PaperAgentState(
            user_request=user_request,
            max_papers=max_papers,
            error=NodeError(),
            config={},
        )

        # 运行图：传入的 dict 会作为初始 state。ainvoke 会按边与条件依次执行节点，直到 END 或 handle_error_node → END
        await self.graph.ainvoke({"state_queue": self.state_queue, "value": initial_state})
        # 通知前端流程已完全结束（成功或已在 handle_error_node 里标记 FAILED），前端可关闭 SSE 或展示最终状态
        await self.state_queue.put(BackToFrontData(step=ExecutionState.FINISHED, state="finished", data=None))

    
# 本地调试
if __name__ == "__main__":
    q = asyncio.Queue()
    orchestrator = PaperAgentOrchestrator(state_queue=q)
    asyncio.run(orchestrator.run("帮我写一篇有关 llm 在无人驾驶方面的调研报告。"))

    