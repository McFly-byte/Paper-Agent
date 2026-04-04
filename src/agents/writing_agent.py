import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# from typing import Dict, Any
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
# from sqlalchemy.sql.functions import current_date
from src.agents.sub_writing_agent.writing_state_models import WritingState, WritingRunContext
from src.core.state_models import State, PaperRunContext
# from src.agents.sub_writing_agent import writing_director_agent, parallel_writing_node
from src.agents.sub_writing_agent.parallel_writing_node import parallel_writing_node
from src.agents.sub_writing_agent.writing_director_agent import writing_director_node
# from src.agents.sub_writing_agent.writing_agent import section_writing_node
from src.agents.sub_writing_agent.writing_agent import create_writing_agent
# from src.agents.sub_writing_agent.retrieval_agent import retrieval_node
from src.agents.sub_writing_agent.retrieval_agent import create_retrieval_agent
from src.core.state_models import ExecutionState
# from src.core.state_models import BackToFrontData
# from src.utils.tool_utils import handlerChunk
from src.utils.log_utils import setup_logger
from src.core.run_context import tmp_db_id_var
from src.services.run_tmp_state_store import get_text, put_json, KEY_ANALYSE_RESULTS, KEY_WRITTED_SECTIONS

logger = setup_logger(__name__)

class WritingWorkflow:
    def __init__(self):
        self.workflow = self.build_workflow()
        
    def build_workflow(self):
        """构建LangGraph工作流"""
        builder = StateGraph(WritingState, context_schema=WritingRunContext)

        # 添加节点
        builder.add_node("writing_director_node", writing_director_node)
        builder.add_node("parallel_writing_node", parallel_writing_node)

        # 设置入口点
        builder.set_entry_point("writing_director_node")

        # 添加边
        builder.add_edge("writing_director_node", "parallel_writing_node")
        builder.add_edge("parallel_writing_node", END)

        # 编译图（进程内内存 checkpoint，与主编排器一致）
        graph = builder.compile(checkpointer=InMemorySaver())
    
        return graph
    
async def writing_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    """调度子工作流写作"""
    state_queue = runtime.context.state_queue
    try:
        current_state = state["value"]
        current_state.current_step = ExecutionState.WRITING
        # await state_queue.put(BackToFrontData(step=ExecutionState.WRITING,state="initializing",data=None))
        ga = current_state.analyse_results
        if ga is None:
            ga = await get_text(current_state, KEY_ANALYSE_RESULTS) or ""
        writing_state: WritingState = {
            "user_request": current_state.user_request,
            "global_analysis": ga,
            "sections": [],
            "writted_sections": [],
            "current_section_index": -1,
            "retrieved_docs": [],
        }
        writingWorkFlow = WritingWorkflow()
        logger.info("[工作流·写作] 启动写作子图：大纲 → 多章节并行撰写（LLM 密集，耗时较长）…")
        tmp_id = (current_state.config or {}).get("tmp_db_id")
        tok = tmp_db_id_var.set(tmp_id) if tmp_id else None
        try:
            writing_state = await writingWorkFlow.workflow.ainvoke(
                writing_state,
                context=WritingRunContext(state_queue=state_queue),
                config={
                    "configurable": {
                        "thread_id": f"{current_state.run_id}-writing",
                    }
                },
            )
        finally:
            if tok is not None:
                tmp_db_id_var.reset(tok)
        logger.info(f"writing_state: {writing_state}")
        section_texts = [
            (section.content or "").strip()
            for section in writing_state["writted_sections"]
        ]
        await put_json(current_state, KEY_WRITTED_SECTIONS, section_texts)
        current_state.writted_sections = []
        # await state_queue.put(BackToFrontData(step=ExecutionState.WRITING,state="completed",data=writing_state["writted_sections"]))
        return {"value": current_state}
        
    except Exception as e:
        state["value"].error.writing_node_error = f"Writing failed: {str(e)}"
        # await state_queue.put(BackToFrontData(step=ExecutionState.WRITING,state="error",data=str(e)))
        return {"value": state["value"]}

async def main():
    import asyncio as _asyncio

    global_analysis = """
    全局分析结果LangGraph 是一个基于图状态机架构的框架，专为编排复杂、有状态的 AI 智能体（Agent）工作流而设计。它通过引入“循环”概念，克服了传统链式结构无法处理循环和持续对话的局限，非常适合构建多步骤推理、工具调用和多智能体协作系统。其核心优势在于提供了极高的灵活性和清晰的状态管理，是开发高级AI应用的关键工具
    """
    q = _asyncio.Queue()
    writing_state: WritingState = {
        "user_request": "本地调试写作子图",
        "global_analysis": global_analysis,
        "sections": [],
        "writted_sections": [],
        "current_section_index": -1,
        "retrieved_docs": [],
    }
    writingWorkFlow = WritingWorkflow()
    result = await writingWorkFlow.workflow.ainvoke(
        writing_state,
        context=WritingRunContext(state_queue=q),
        config={"configurable": {"thread_id": "local-writing-debug"}},
    )
    # result是WritingState，而WritingState本质上就是一个字典
    print("result:")
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())