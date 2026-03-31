from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from src.agents.userproxy_agent import WebUserProxyAgent,userProxyAgent
from pydantic import BaseModel, Field
from typing import Optional,List
import re
import ast

from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearcher
from src.core.state_models import State,ExecutionState
from src.core.prompts import search_agent_prompt
from src.core.state_models import BackToFrontData

from src.core.model_client import create_search_model_client

logger = setup_logger(__name__)


model_client = create_search_model_client()

# 创建一个查询条件类，包括查询内容、主题、时间范围等信息，用于存储用户的查询需求
class SearchQuery(BaseModel):
    """查询条件类，存储用户查询需求"""
    querys: List[str] = Field(default=None, description="查询条件列表")
    start_date: Optional[str] = Field(default=None, description="开始时间, 格式: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束时间, 格式: YYYY-MM-DD")

search_agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    system_message=search_agent_prompt,
    output_content_type=SearchQuery
)

def parse_search_query(s: str) -> SearchQuery:
    """将前端传回的字符串转为 SearchQuery 对象"""
    # 提取 querys（使用 ast.literal_eval 保证安全）
    querys_match = re.search(r"querys\s*=\s*(\[[^\]]*\])", s)
    start_match = re.search(r"start_date\s*=\s*'([^']*)'", s)
    end_match = re.search(r"end_date\s*=\s*'([^']*)'", s)

    querys = []
    if querys_match:
        try:
            querys = ast.literal_eval(querys_match.group(1))
        except Exception:
            querys = []

    start_date = start_match.group(1) if start_match else None
    end_date = end_match.group(1) if end_match else None

    return SearchQuery(querys=querys, start_date=start_date, end_date=end_date)

async def search_node(state: State) -> State:
    """搜索论文节点"""
    state_queue = None
    try:
        state_queue = state["state_queue"] 
        current_state = state["value"]
        current_state.current_step = ExecutionState.SEARCHING
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="initializing",data=None)) # 将初始状态推送到队列

        prompt = f"""
        请根据用户查询需求，生成检索查询条件。
        用户查询需求：{current_state.user_request}
        """
        # 调用search_agent生成查询条件
        response = await search_agent.run(task = prompt) 
        search_query = response.messages[-1].content
        # 将查询条件推送到队列，等待前端人工审核
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="user_review",data=f"{search_query}"))
        # 等待前端人工审核
        result = await userProxyAgent.on_messages(
            [TextMessage(content="请人工审核：查询条件是否符合？", source="AI")],
            cancellation_token=CancellationToken()
        )
        search_query = parse_search_query(result.content)

        # 调用检索服务
        paper_searcher = PaperSearcher()
        # 调用检索服务搜索论文
        results = await paper_searcher.search_papers(
            querys = search_query.querys,
            start_date = search_query.start_date,
            end_date = search_query.end_date,
        )
        current_state.search_results = results
        if len(results) > 0:
            # 将搜索结果推送到队列
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="completed",data=f"论文搜索完成，共找到 {len(results)} 篇论文"))
        else:
            # 将错误信息推送到队列
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="error",data="没有找到相关论文,请尝试其他查询条件"))
            current_state.error.search_node_error = "没有找到相关论文,请尝试其他查询条件"
        return {"value": current_state}
            
    except Exception as e:
        err_msg = f"Search failed: {str(e)}"
        state["value"].error.search_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="error",data=err_msg))
        return state