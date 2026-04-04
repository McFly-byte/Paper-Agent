from autogen_agentchat.agents import AssistantAgent
from src.core.prompts import writing_director_agent_prompt
from langgraph.runtime import Runtime

from src.agents.sub_writing_agent.writing_state_models import WritingState, WritingRunContext
from src.core.state_models import BackToFrontData
from src.core.state_models import ExecutionState

from typing import Dict, Any, List

from src.utils.log_utils import setup_logger
from src.core.model_client import create_subwriting_writing_director_model_client


logger = setup_logger(__name__)


model_client = create_subwriting_writing_director_model_client()


writing_director_agent = AssistantAgent(
    name="writing_director_agent",
    description="一个写作主管，你只负责拆分写作任务，并返回小节列表。",
    model_client=model_client,
    system_message=writing_director_agent_prompt,
    # 非流式：避免 run_stream 内 break/中断触发 GeneratorExit，与 OTel span 清理冲突（见 global_analyse_agent）
    model_client_stream=False,
)

def parse_outline(outline_str: str) -> List[str]:
    """
    解析大纲字符串，提取每个带编号的小节。

    支持常见编号形式：1. / 2. / 1.1 / 2.3.1 等（行首，可带 markdown 标题或列表符）。
    旧实现仅用 (\\d+\\.\\d+|\\d+)\\s，无法匹配「1. 标题」(\\d+ 后紧跟的是点而非空格)，会导致
    sections 为空、并行写作不执行、最终报告只有「无章节内容提供」骨架。
    """
    import re

    text = outline_str.strip()
    if not text:
        return []

    # 行首小节编号，捕获组 1 为编号本体
    pattern = re.compile(
        r"(?:^|\n)\s*(?:[-*]\s*)?(?:#{1,6}\s+)?(\d+(?:\.\d+)*\.?)\s+",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        # 无编号时保留整段，避免子图空跑、报告无正文
        logger.warning(
            "[工作流·写作] 大纲中未识别到编号小节，将整段作为单节任务；"
            "请检查模型是否按提示返回 1.1 / 1. 等格式。"
        )
        return [text]

    result: List[str] = []
    for i, m in enumerate(matches):
        start = m.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
    return result

async def writing_director_node(
    state: WritingState, runtime: Runtime[WritingRunContext]
) -> Dict[str, Any]:
    state_queue = runtime.context.state_queue
    await state_queue.put(BackToFrontData(step=ExecutionState.WRITING_DIRECTOR,state="initializing",data=None))
    try: 
        logger.info("开始执行写作主管节点")
        """写作主管节点：生成大纲，并将大纲拆分成子任务"""
        user_request = state["user_request"]
        global_analysis = state["global_analysis"]
        prompt = f"""
        用户的需求:
        {user_request}
        该领域的分析:
        {global_analysis}
        请根据用户提供的需求和关于该领域的分析，生成结构清晰、逻辑连贯的写作子任务：
        """
        logger.info("[工作流·写作] 写作主管：调用 LLM 生成大纲（非流式，请稍候）…")
        response = await writing_director_agent.run(task=prompt)
        msgs = getattr(response, "messages", None) or []
        last = msgs[-1] if msgs else None
        raw = last.content if last is not None else ""
        outline = raw if isinstance(raw, str) else str(raw)
        logger.info("[工作流·写作] 写作主管：大纲已返回，解析为小节列表…")
        sections = parse_outline(outline)
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING_DIRECTOR,state="completed",data=None))
        return {"sections": sections}
    except Exception as e:
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING_DIRECTOR,state="error",data=f"Writing director failed: {str(e)}"))
        return state
