from __future__ import annotations

import time
from typing import Any, Optional

from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import APIError, RateLimitError
from httpx import ReadTimeout, TimeoutException

from src.core.llm_infra.factory import create_openai_chat_client_from_policy
from src.core.llm_infra.routing import resolve_invocation_policy
from src.core.llm_infra.types import InvocationPolicy, LLMChannel
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _estimate_prompt_tokens(user_prompt: str, overhead: int = 200) -> int:
    return max(1, len(user_prompt) // 4 + overhead)


def _usage_from_result(result: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """从 AutoGen create 结果抽取 token 用量（若存在）。"""
    usage = getattr(result, "usage", None)
    if usage is None:
        return None, None, None
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    try:
        return (
            int(pt) if pt is not None else None,
            int(ct) if ct is not None else None,
            int(tt) if tt is not None else None,
        )
    except (TypeError, ValueError):
        return None, None, None


async def chat_completion_text_routed(
    client_type: str,
    user_prompt: str,
    *,
    node_name: Optional[str] = None,
    temperature: float = 0.2,
    source: str = "paper-agent",
) -> str:
    """按 client_type 解析策略并调用 Chat；支持可降级节点的单次 fallback。"""
    policy = resolve_invocation_policy(client_type, node_name=node_name or client_type)
    return await _chat_once(policy, user_prompt, temperature=temperature, source=source, fallback_used=False)


async def _chat_once(
    policy: InvocationPolicy,
    user_prompt: str,
    *,
    temperature: float,
    source: str,
    fallback_used: bool,
) -> str:
    client: Optional[OpenAIChatCompletionClient] = None
    client = create_openai_chat_client_from_policy(policy)
    t0 = time.perf_counter()
    est = _estimate_prompt_tokens(user_prompt)
    retry_count = 0
    try:
        result = await client.create(
            [UserMessage(content=user_prompt, source=source)],
            extra_create_args={"temperature": temperature},
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        pt, ct, tt = _usage_from_result(result)
        content = getattr(result, "content", None)
        text = (content if isinstance(content, str) else (str(content) if content is not None else str(result))).strip()
        logger.info(
            "llm_invoke provider=%s model=%s client_type=%s node=%s use_batch=%s enable_thinking=%s "
            "retry_count=%s fallback_used=%s latency_ms=%s est_prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            policy.provider,
            policy.model,
            policy.client_type,
            policy.node_name,
            policy.channel == LLMChannel.BATCH_CHAT,
            policy.enable_thinking,
            retry_count,
            fallback_used,
            latency_ms,
            est,
            ct,
            tt,
        )
        return text
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "llm_invoke_fail provider=%s model=%s client_type=%s latency_ms=%s err=%s",
            policy.provider,
            policy.model,
            policy.client_type,
            latency_ms,
            exc,
        )
        if (
            policy.degradable
            and not fallback_used
            and policy.fallback_model
            and policy.fallback_provider
        ):
            if isinstance(exc, (RateLimitError, APIError, TimeoutException, ReadTimeout, OSError)):
                fb = InvocationPolicy(
                    client_type=policy.client_type + ":fallback",
                    provider=policy.fallback_provider,
                    model=policy.fallback_model,
                    channel=LLMChannel.REALTIME,
                    enable_thinking=False if policy.fallback_provider == "dashscope" else None,
                    timeout=policy.timeout,
                    max_retries=policy.max_retries,
                    concurrency_class=policy.concurrency_class,
                    rate_limit_bucket=policy.rate_limit_bucket,
                    degradable=False,
                    fallback_model=None,
                    fallback_provider=None,
                    node_name=policy.node_name,
                )
                logger.info("llm_fallback to provider=%s model=%s", fb.provider, fb.model)
                return await _chat_once(fb, user_prompt, temperature=temperature, source=source, fallback_used=True)
        raise
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    logger.debug("model client close() 忽略异常", exc_info=True)
