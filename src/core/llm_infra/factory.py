from __future__ import annotations

import os
from typing import Any, Dict, Optional

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.core.config import config
from src.core.llm_infra.types import InvocationPolicy, LLMChannel
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _resolve_request_timeout(provider: str, model_entry: Optional[Dict[str, Any]] = None) -> float:
    model_entry = model_entry or {}
    for src in (model_entry, config.get(provider) or {}):
        t = src.get("request_timeout")
        if t is not None:
            try:
                return float(t)
            except (TypeError, ValueError):
                break
    return 1800.0 if provider == "ollama" else 600.0


def _provider_base_url(provider: str, policy: InvocationPolicy) -> str:
    pcfg = config.get(provider) or {}
    if provider == "dashscope" and policy.channel == LLMChannel.BATCH_CHAT:
        u = (os.environ.get("DASHSCOPE_BATCH_BASE_URL") or pcfg.get("batch_base_url") or "").strip()
        if u:
            return u.rstrip("/")
        logger.warning("百炼 Batch Chat 未配置 batch_base_url / DASHSCOPE_BATCH_BASE_URL，回退 realtime base_url")
    return str(pcfg.get("base_url") or "").strip().rstrip("/")


def create_openai_chat_client_from_policy(
    policy: InvocationPolicy,
    *,
    vision: bool = True,
    function_calling: bool = True,
    json_output: bool = True,
    structured_output: bool = True,
    family: str = "Qwen",
) -> OpenAIChatCompletionClient:
    """由 InvocationPolicy 构建 AutoGen ``OpenAIChatCompletionClient``（OpenAI 兼容 SDK）。"""
    provider = policy.provider
    pcfg = config.get(provider) or {}
    api_key = pcfg.get("api_key")
    base_url = _provider_base_url(provider, policy)

    if not policy.model:
        raise ValueError(f"未指定模型名称（client_type={policy.client_type}）")
    if not base_url:
        raise ValueError(f"未配置 {provider}.base_url（client_type={policy.client_type}）")

    fam = family
    if fam == "Qwen" and provider != "siliconflow":
        fam = "GPT" if provider == "openai" else provider.capitalize()

    model_info = ModelInfo(
        vision=vision,
        function_calling=function_calling,
        json_output=json_output,
        family=fam,
        structured_output=structured_output,
    )

    timeout = policy.timeout if policy.timeout is not None else _resolve_request_timeout(provider, {})
    if policy.channel == LLMChannel.BATCH_CHAT:
        try:
            batch_floor = float(config.get("llm_batch_chat.default_timeout_seconds") or 1800)
        except (TypeError, ValueError):
            batch_floor = 1800.0
        timeout = max(float(timeout), batch_floor)

    kwargs: Dict[str, Any] = dict(
        model=policy.model,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info,
        max_retries=policy.max_retries,
        timeout=timeout,
    )

    # 百炼：所有请求显式 enable_thinking（OpenAI SDK extra_body）
    if provider == "dashscope" and policy.enable_thinking is not None:
        kwargs["extra_body"] = {"enable_thinking": bool(policy.enable_thinking)}

    client = OpenAIChatCompletionClient(**kwargs)
    logger.debug(
        "LLM client built: client_type=%s node=%s provider=%s model=%s channel=%s use_batch=%s thinking=%s",
        policy.client_type,
        policy.node_name,
        provider,
        policy.model,
        policy.channel.value,
        policy.channel == LLMChannel.BATCH_CHAT,
        policy.enable_thinking,
    )
    return client


def connection_params_for_policy(policy: InvocationPolicy) -> Dict[str, Any]:
    """供 LangChain ChatOpenAI 等与 OpenAI 兼容的 HTTP 客户端复用同一套 base_url / thinking。"""
    pcfg = config.get(policy.provider) or {}
    api_key = pcfg.get("api_key")
    base_url = _provider_base_url(policy.provider, policy)
    timeout = policy.timeout if policy.timeout is not None else _resolve_request_timeout(policy.provider, {})
    if policy.channel == LLMChannel.BATCH_CHAT:
        try:
            batch_floor = float(config.get("llm_batch_chat.default_timeout_seconds") or 1800)
        except (TypeError, ValueError):
            batch_floor = 1800.0
        timeout = max(float(timeout), batch_floor)
    extra_body: Optional[Dict[str, Any]] = None
    if policy.provider == "dashscope" and policy.enable_thinking is not None:
        extra_body = {"enable_thinking": bool(policy.enable_thinking)}
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": policy.model,
        "timeout": float(timeout),
        "extra_body": extra_body,
        "max_retries": int(policy.max_retries),
    }
