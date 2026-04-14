from autogen_agentchat.agents import AssistantAgent

from src.core.model_client import create_subwriting_review_model_client
from src.core.prompts import review_agent_prompt
from src.agents.sub_writing_agent.writing_state_models import ReviewDecision


def create_review_agent():
    model_client = create_subwriting_review_model_client()

    review_agent = AssistantAgent(
        name="review_agent",
        description="一个审查助手。",
        model_client=model_client,
        system_message=review_agent_prompt,
        tools=[],
        output_content_type=ReviewDecision,
        model_client_stream=False,
    )
    return review_agent
