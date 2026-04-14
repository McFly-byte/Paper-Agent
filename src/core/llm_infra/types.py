from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LLMChannel(str, Enum):
    """OpenAI 兼容调用通道：百炼 Batch Chat 仅换 base_url，路径仍为 chat completions。"""

    REALTIME = "realtime"
    BATCH_CHAT = "batch_chat"


@dataclass(frozen=True)
class InvocationPolicy:
    """单次 Chat 客户端构建所需的结构化策略（由集中路由解析，业务节点不手写）。"""

    client_type: str
    provider: str
    """配置块键名：siliconflow、dashscope、openai、ollama、ark 等。"""
    model: str
    channel: LLMChannel
    enable_thinking: Optional[bool]
    """dashscope/百炼：必须显式 True/False；其它 provider 为 None 表示不传 extra_body。"""
    timeout: Optional[float]
    max_retries: int
    concurrency_class: str
    rate_limit_bucket: Optional[str]
    degradable: bool
    fallback_model: Optional[str]
    fallback_provider: Optional[str]
    node_name: str
    """日志 / LangSmith metadata 用，与 LangGraph 节点或脚本名对齐。"""
