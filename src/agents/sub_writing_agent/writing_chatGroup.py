from src.core.model_client import create_subwriting_writing_model_client
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from src.agents.sub_writing_agent.writing_agent import create_writing_agent
from src.agents.sub_writing_agent.retrieval_agent import create_retrieval_agent
from src.agents.sub_writing_agent.review_agent import create_review_agent
from src.core.prompts import selector_prompt

# SelectorGroupChat 每有一名参与者完成一轮响应，group manager 的 turn 计数 +1。
# 依据 project.log（如 2026-04-02）：第 2/13 章约 16 次 TextMessage 发言切换后正常结束；
# 第 7/13 章在同一段日志中已超过 13 次仍无「章节完成」，存在长时间循环风险。
# 取最坏正常完成值 16，上浮约 50% 作为硬上限，避免无限拉扯。
WRITING_SELECTOR_GROUP_MAX_TURNS = 24


def create_writing_group():
    model_client = create_subwriting_writing_model_client()

    text_termination = TextMentionTermination("APPROVE")
    
    
    writing_agent = create_writing_agent()
    review_agent = create_review_agent()
    retrieval_agent = create_retrieval_agent()

    # 写作组
    task_group = SelectorGroupChat(
        [writing_agent, retrieval_agent, review_agent],
        model_client=model_client,
        termination_condition=text_termination,
        max_turns=WRITING_SELECTOR_GROUP_MAX_TURNS,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,
    )
    return task_group
