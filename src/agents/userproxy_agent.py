import asyncio
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class WebUserProxyAgent(UserProxyAgent):
    def __init__(self, name):
        super().__init__(name)
        self.waiting_future = None  # 保存等待的future对象
    
    async def on_messages(self, messages, cancellation_token: CancellationToken):
        # 触发等待：通知前端“等待人工输入”
        self.waiting_future = asyncio.get_event_loop().create_future() # create_future() 只创建占位符
        # 等待前端输入
        user_input = await self.waiting_future # 挂起等待前端输入
        # 收到输入后返回给AutoGen 
        return TextMessage(content=user_input, source="human")

    def set_user_input(self, user_input: str | None) -> str:
        """唤醒当前 HITL 等待。返回 ok | no_pending_hitl | already_resolved。"""
        fut = self.waiting_future
        if fut is None:
            return "no_pending_hitl"
        if fut.done():
            return "already_resolved"
        fut.set_result(user_input if user_input is not None else "")
        return "ok"
