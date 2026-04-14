"""LLM 调用基础设施：路由、策略、百炼 thinking/batch 与观测字段。

``factory`` / ``invoke`` 依赖 autogen_ext，请从子模块导入，避免在无 AutoGen 环境下 import 失败。
"""

from src.core.llm_infra.types import InvocationPolicy, LLMChannel
from src.core.llm_infra.routing import resolve_invocation_policy

__all__ = [
    "InvocationPolicy",
    "LLMChannel",
    "resolve_invocation_policy",
]
