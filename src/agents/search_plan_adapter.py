"""在旧 search_node 前注入「研究计划 → 检索提示」，并作为 LangGraph 入口包装。"""

from __future__ import annotations

from langgraph.runtime import Runtime

from src.agents.search_agent import search_node
from src.agents.search_plan_context import inject_search_plan_hints
from src.core.state_models import PaperRunContext, State


async def search_node_with_plan(state: State, runtime: Runtime[PaperRunContext]) -> State:
    inject_search_plan_hints(state["value"])
    return await search_node(state, runtime)
