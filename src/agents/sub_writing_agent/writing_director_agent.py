from autogen_agentchat.agents import AssistantAgent
from src.core.prompts import writing_director_agent_prompt
from src.agents.sub_writing_agent.writing_state_models import WritingState
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
    解析大纲字符串，提取每个带编号的小节
    
    Args:
        outline_str: 包含编号小节的字符串，每个小节以编号开头（如1.1, 2.3等）
        
    Returns:
        小节列表，每个元素是一个小节的完整内容
    """
    from typing import List
    import re
    
    # 使用正则表达式匹配小节编号（如1., 1.1, 2.3等）
    # 分割字符串并保留分隔符
    sections = re.split(r'(\d+\.\d+|\d+)\s', outline_str.strip())
    
    # 处理分割结果，组合成完整的小节
    result = []
    for i in range(1, len(sections), 2):
        # 组合编号和内容
        section = f"{sections[i].strip()} {sections[i+1].strip()}"
        result.append(section)
    
    return result

async def writing_director_node(state: WritingState) -> Dict[str, Any]:
    state_queue = state["state_queue"]
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
