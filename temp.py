# -*- coding: utf-8 -*-
"""
面试题：Chain vs LangGraph 对比示例
从 可控性、可观测性、错误处理 三方面用代码对比两种范式。
运行前请确保：poetry install 已安装依赖；Part 4 需 langgraph（本仓库已包含）。
在项目根目录执行：poetry run python temp.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict


# =============================================================================
# Part 1: 可控性（Control Flow）
# =============================================================================

# ---------- 1.1 Chain 风格：线性 pipeline，分支/循环写在「外部代码」或 prompt 里 ----------

class StepStatus(str, Enum):
    OK = "ok"
    FAIL = "fail"


def chain_step_a(input_text: str) -> Dict[str, Any]:
    """模拟第一步：解析意图。状态通过 dict 在步骤间隐式传递。"""
    return {"intent": "search", "query": input_text.strip(), "step": "a_done"}


def chain_step_b(data: Dict[str, Any]) -> Dict[str, Any]:
    """第二步：根据 intent 做不同事——分支逻辑写在「代码里」，不在图里。"""
    if data.get("intent") == "search":
        data["results"] = ["doc1", "doc2"]
    else:
        data["results"] = []
    data["step"] = "b_done"
    return data


def chain_step_c(data: Dict[str, Any]) -> Dict[str, Any]:
    """第三步：生成。没有「循环」语义，若要重试只能在外层写 while。"""
    data["answer"] = f"Based on {data.get('results', [])}"
    data["step"] = "c_done"
    return data


def run_chain_style():
    """Chain 风格：顺序写死，条件分支在 step_b 内部用 if；循环/重试需在外层手写。"""
    state = {"raw": "帮我找一下 LLM 综述"}
    state = chain_step_a(state["raw"])
    state = chain_step_b(state)
    state = chain_step_c(state)
    print("[Chain 可控性] 线性三步，分支在 step_b 的 if 里:", state)
    # 若要做「step_b 失败则重试 3 次」，只能：
    # for _ in range(3):
    #     state = chain_step_b(state)
    #     if ok: break
    # 控制流散落在业务代码中，图结构不可见。


# ---------- 1.2 Graph 风格：节点 + 条件边，分支与循环在图结构中显式定义 ----------

class GraphState(TypedDict):
    query: str
    intent: str
    results: List[str]
    answer: str
    step: str
    need_retry: bool


def graph_node_parse(state: GraphState) -> Dict[str, Any]:
    return {"query": state["query"][:50], "intent": "search", "step": "parse"}


def graph_node_search(state: GraphState) -> Dict[str, Any]:
    return {"results": ["doc1", "doc2"], "step": "search"}


def graph_node_generate(state: GraphState) -> Dict[str, Any]:
    return {"answer": f"Based on {state['results']}", "step": "generate"}


def route_after_search(state: GraphState) -> str:
    """条件边：根据 state 决定下一跳，控制流在图里。"""
    if state.get("need_retry"):
        return "graph_node_search"  # 循环回 search，实现重试/循环
    if state.get("results"):
        return "graph_node_generate"
    return "__end__"


def run_graph_style_controllability():
    """仅演示「条件边」思想：下一跳由 route_after_search(state) 决定，循环/分支在图定义中。"""
    # 实际执行需 LangGraph 的 StateGraph；这里用伪代码表达「图里怎么定义」
    print("[Graph 可控性] 下一跳由 condition_handler(state) 决定，可返回 search_node 形成环，或 generate 或 END；"
          "分支/循环都在图的 add_conditional_edges 中显式声明。")


# =============================================================================
# Part 2: 可观测性（Observability）
# =============================================================================

# ---------- 2.1 Chain 风格：状态散落在局部变量/dict，无统一结构，难以逐步追踪 ----------

@dataclass
class ChainLikePipeline:
    """模拟 Chain：中间状态在步骤间用局部变量传递，没有「全局 State」类型。"""
    steps: List[Callable] = field(default_factory=list)
    # 没有统一的 state 类型，每步输入输出格式由开发者自己约定，难以序列化/持久化

    def run(self, initial: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {"raw": initial}
        for i, step in enumerate(self.steps):
            data = step(data)  # 每步覆盖 data，若某步抛异常，前面步骤的中间结果要自己 try/except 里保存
        return data


def run_chain_observability():
    """Chain：状态隐式，想「记录每步后的 state」需在每步后手动 append 到 list。"""
    def step1(d: Dict) -> Dict:
        d["a"] = 1
        return d

    def step2(d: Dict) -> Dict:
        d["b"] = 2
        return d

    pipeline = ChainLikePipeline(steps=[step1, step2])
    final = pipeline.run("x")
    print("[Chain 可观测性] 只有最终 final state，中间 step1 后的 state 未显式保留；"
          "若要追踪需在每步后手动 history.append(copy.deepcopy(data))。")


# ---------- 2.2 Graph 风格：显式 State 类型，每步输入/输出为 State 的更新，易追踪与持久化 ----------

# 使用 LangGraph 时，状态是 TypedDict/Pydantic，每个节点 (state) -> partial_update；
# 运行时可配置 checkpointer，每个节点执行完自动持久化 state，便于断点恢复与审计。

def run_graph_observability_note():
    print("[Graph 可观测性] State 为 TypedDict/Annotated，节点返回 partial update；"
          "LangGraph 支持 checkpointer，每步后 state 可序列化、可持久化、可逐节点回放。")


# =============================================================================
# Part 3: 错误处理（Error Handling）
# =============================================================================

# ---------- 3.1 Chain 风格：异常需在「图外」用 try/except 包住，或每步内自己 catch ----------

def chain_step_may_fail(data: Dict[str, Any]) -> Dict[str, Any]:
    if data.get("fail_at") == "step":
        raise ValueError("simulated error")
    data["done"] = True
    return data


def run_chain_error_handling():
    """Chain：错误处理要么外层 try/except，要么每步内部 try 并写 data['error']，逻辑分散。"""
    data: Dict[str, Any] = {"fail_at": "step"}
    try:
        data = chain_step_may_fail(data)
    except ValueError as e:
        data["error"] = str(e)
        data["done"] = False
    print("[Chain 错误处理] 依赖外部 try/except 或每步内 if error；无法在「图」里声明错误节点。", data)


# ---------- 3.2 Graph 风格：图中专设错误处理节点，条件边根据 state.error 路由到该节点 ----------

# 见本项目 src/agents/orchestrator.py：
# - condition_handler 检查 state["value"].error 各字段，若有则 return "handle_error_node"
# - handle_error_node 将 current_step 设为 FAILED，然后 add_edge(handle_error_node, END)
# 错误处理是图的一部分，而不是外层 try/except。

def run_graph_error_handling_note():
    print("[Graph 错误处理] 在图中 add_node('handle_error_node', handle_error_node)，"
          "condition_handler 根据 state.value.error 返回 'handle_error_node'，"
          "add_edge('handle_error_node', END)；错误路径在图结构中显式存在。")


# =============================================================================
# Part 4: 使用真实 LangGraph 的完整小示例（可控性 + 可观测性 + 错误处理）
# =============================================================================

def run_langgraph_concrete_example():
    """用 LangGraph 写一个最小可运行图：显式 State、条件边、错误节点。"""
    try:
        from langgraph.graph import StateGraph, END, START
    except ImportError:
        print("请先安装: poetry add langgraph，再运行本示例。")
        return

    class State(TypedDict):
        value: int
        error: Optional[str]

    def node_a(state: State) -> Dict[str, Any]:
        return {"value": state["value"] + 1}

    def node_b(state: State) -> Dict[str, Any]:
        if state["value"] > 5:
            return {"error": "value too large"}  # 模拟错误，由条件边路由到 error_node
        return {"value": state["value"] * 2}

    def error_node(state: State) -> Dict[str, Any]:
        return {"value": -1, "error": state.get("error") or "unknown"}

    def route(state: State) -> str:
        if state.get("error"):
            return "error_node"
        if state["value"] >= 8:
            return END
        return "node_b"

    builder = StateGraph(State)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("error_node", error_node)
    builder.set_entry_point("node_a")
    builder.add_edge(START, "node_a")
    builder.add_conditional_edges("node_a", route)
    builder.add_conditional_edges("node_b", route)
    builder.add_edge("error_node", END)

    graph = builder.compile()

    # 正常路径：0 -> node_a(1) -> node_b(2) -> node_b(4) -> node_b(8) -> END
    result_ok = graph.invoke({"value": 0, "error": None})
    print("[LangGraph 正常] 显式 State + 条件边，最终 state:", result_ok)

    # 错误路径：初始 5 -> node_a(6) -> node_b 置位 error -> error_node -> END
    result_err = graph.invoke({"value": 5, "error": None})
    print("[LangGraph 错误] 经 error_node 收尾，最终 state:", result_err)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("========== 1. 可控性 ==========")
    run_chain_style()
    run_graph_style_controllability()

    print("\n========== 2. 可观测性 ==========")
    run_chain_observability()
    run_graph_observability_note()

    print("\n========== 3. 错误处理 ==========")
    run_chain_error_handling()
    run_graph_error_handling_note()

    print("\n========== 4. LangGraph 最小可运行示例 ==========")
    run_langgraph_concrete_example()
