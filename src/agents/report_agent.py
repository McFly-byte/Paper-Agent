from autogen_agentchat.agents import AssistantAgent

from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from langgraph.runtime import Runtime

from src.core.state_models import State, ExecutionState, PaperRunContext
from src.core.prompts import report_agent_prompt
from src.core.state_models import BackToFrontData
from autogen_agentchat.base import TaskResult

from src.core.model_client import create_report_model_client
from src.services.report_history_store import append_completed
from src.services.run_tmp_state_store import get_json, KEY_WRITTED_SECTIONS
from src.evaluation.evaluators import run_evaluation

logger = setup_logger(__name__)


model_client = create_report_model_client()


report_agent = AssistantAgent(
    name="report_agent",
    model_client=model_client,
    system_message=report_agent_prompt,
    model_client_stream=True
)

async def report_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    """报告生成节点"""
    state_queue = runtime.context.state_queue
    try:
        current_state = state["value"]
        current_state.current_step = ExecutionState.REPORTING
        await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state="initializing",data=None))
        sections = current_state.writted_sections
        if not sections:
            loaded = await get_json(current_state, KEY_WRITTED_SECTIONS)
            if isinstance(loaded, list):
                sections = loaded
        sections_text = "\n".join(sections) if sections else "无章节内容提供"
    
        prompt = f"""
        请将以下提供的章节内容组装成一份完整的调研报告，并以Markdown格式输出。

        【章节内容开始】
        {sections_text}
        【章节内容结束】

        【输出要求】
        1. 使用Markdown格式进行排版（标题、列表、加粗等）
        2. 自动补充必要的过渡语句使报告连贯
        3. 保持专业学术风格
        4. 直接输出完整报告，无需解释过程

        【额外说明】
        请确保章节逻辑顺序合理，如有需要可调整章节排列。
        """
        logger.info("[工作流·报告] 流式组装最终 Markdown 报告（LLM）…")
        is_thinking = None
        is_First = True
        report_text_captured = False
        async for chunk in report_agent.run_stream(task = prompt):
            if is_First:
                is_First = False
                continue
            if not isinstance(chunk, TaskResult):
                if chunk.type == "ThoughtEvent":
                    continue
                if chunk.type == "TextMessage":
                    current_state.report_markdown = chunk.content
                    report_text_captured = True
                    continue
                if report_text_captured:
                    continue

                state,is_thinking = handlerChunk(is_thinking,chunk.content)
                if state is None:
                    continue
                await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state=state,data=chunk.content))

        kb_label = current_state.config.get("knowledge_base_label")
        
        # 在报告生成完成后立即执行 LangSmith 评估（丰富评估）
        eval_results = {}
        try:
            # 同时传入 state 和 outputs，让评估函数能从多个来源提取数据
            eval_results = await run_evaluation(
                run_id=current_state.run_id,
                state=current_state,
                outputs={
                    "report_markdown": current_state.report_markdown,
                    "sections": getattr(current_state, 'writted_sections', None),
                },
            )
            if hasattr(current_state, 'config') and isinstance(current_state.config, dict):
                current_state.config["evaluation"] = eval_results
                current_state.config["overall_score"] = eval_results.get("overall_score", 0.0)
            logger.info("[报告节点] LangSmith 评估完成，得分: %.2f", eval_results.get("overall_score", 0.0))
        except Exception as e:
            logger.warning("报告评估阶段轻微异常: %s", e)
            eval_results = {"overall_score": 0.0, "summary": f"评估异常: {str(e)[:80]}"}

        saved_id = await append_completed(
            current_state.report_markdown or "",
            current_state.user_request,
            knowledge_base=kb_label,
        )
        if saved_id:
            current_state.config["last_saved_report_id"] = saved_id
            # 更新历史记录中的评估信息
            # (append_completed 内部可进一步增强，此处先简单记录)

        await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state="completed",data=None))
        
        # 推送最终评估结果给前端
        if eval_results:
            await state_queue.put(
                BackToFrontData(
                    step=ExecutionState.COMPLETED,
                    state="evaluation_complete",
                    data={"evaluation": eval_results, "overall_score": eval_results.get("overall_score")}
                )
            )
        
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Report failed: {str(e)}"
        state["value"].error.report_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state="error",data=err_msg))
        return {"value": state["value"]}