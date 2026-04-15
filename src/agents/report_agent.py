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
from src.core.node_gates import gate_report, record_gate
from src.core.workflow_recovery import (
    clear_recovery_feedback_for,
    format_recovery_user_block,
    set_recovery_target,
)

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
        _rep_rec = format_recovery_user_block(current_state.config or {}, "report")

        prompt = f"""
        请将以下提供的章节内容组装成一份完整的调研报告，并以Markdown格式输出。
        {_rep_rec}

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
                    # 流式下可能多次 TextMessage：保留最长正文，避免仅末段 delta 参与门禁覆盖校验
                    new_text = str(getattr(chunk, "content", "") or "")
                    prev = str(getattr(current_state, "report_markdown", "") or "")
                    if len(new_text) >= len(prev):
                        current_state.report_markdown = new_text
                    report_text_captured = True
                    continue
                if report_text_captured:
                    continue

                state,is_thinking = handlerChunk(is_thinking,chunk.content)
                if state is None:
                    continue
                await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state=state,data=chunk.content))

        kb_label = current_state.config.get("knowledge_base_label")

        repg = gate_report(
            report_markdown=current_state.report_markdown or "",
            section_snippets=list(sections or []),
        )
        current_state.boundary_checks = record_gate(current_state.boundary_checks, "report", repg)
        if not repg.passed:
            detail = "；".join(repg.reasons) if repg.reasons else "报告门禁未通过"
            current_state.error.report_node_error = detail
            set_recovery_target(current_state, "report")
            await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING, state="error", data=detail))
            return {"value": current_state}

        saved_id = await append_completed(
            current_state.report_markdown or "",
            current_state.user_request,
            knowledge_base=kb_label,
        )
        if saved_id:
            current_state.config["last_saved_report_id"] = saved_id
            # 更新历史记录中的评估信息
            # (append_completed 内部可进一步增强，此处先简单记录)

        clear_recovery_feedback_for(current_state, "report")
        await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state="completed",data=None))

        return {"value": current_state}

    except Exception as e:
        err_msg = f"Report failed: {str(e)}"
        vs = state["value"]
        vs.error.report_node_error = err_msg
        set_recovery_target(vs, "report")
        await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state="error",data=err_msg))
        return {"value": vs}