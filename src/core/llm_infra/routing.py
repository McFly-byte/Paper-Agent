from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.core.config import config
from src.core.llm_infra.types import InvocationPolicy, LLMChannel
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def normalize_provider_key(name: str) -> str:
    """bailian 与 dashscope 共用百炼凭据与端点配置块。"""
    n = _lower(name)
    if n == "bailian":
        return "dashscope"
    return n or "siliconflow"


def effective_chat_provider(declared: str) -> str:
    """DEFAULT_LLM_PROVIDER 可强制全局 chat 走硅基或百炼（不改各节点代码）。"""
    force = _lower(os.environ.get("DEFAULT_LLM_PROVIDER") or config.get("DEFAULT_LLM_PROVIDER"))
    if force in ("bailian", "dashscope"):
        return "dashscope"
    if force == "siliconflow":
        return "siliconflow"
    return normalize_provider_key(declared)


def effective_embedding_provider(declared: str) -> str:
    force = _lower(os.environ.get("DEFAULT_EMBEDDING_PROVIDER") or config.get("DEFAULT_EMBEDDING_PROVIDER"))
    if force in ("bailian", "dashscope"):
        return "dashscope"
    if force == "siliconflow":
        return "siliconflow"
    return normalize_provider_key(declared)


def _coerce_channel(raw: Any) -> LLMChannel:
    s = _lower(str(raw) if raw is not None else "")
    if s in ("batch", "batch_chat", "offline_batch"):
        return LLMChannel.BATCH_CHAT
    return LLMChannel.REALTIME


def _resolve_model(pol: Dict[str, Any], base: Dict[str, Any]) -> str:
    env_key = pol.get("resolve_model_from_env_key")
    if isinstance(env_key, str) and env_key.strip():
        v = config.get(env_key.strip())
        if v is not None and str(v).strip():
            return str(v).strip()
    m = pol.get("model")
    if m is not None and str(m).strip():
        return str(m).strip()
    m2 = base.get("model")
    if m2 is not None and str(m2).strip():
        return str(m2).strip()
    return ""


def _bool_from_config(key: str, default: bool) -> bool:
    v = config.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def resolve_invocation_policy(client_type: str, *, node_name: Optional[str] = None) -> InvocationPolicy:
    """合并 ``*-model`` 配置与 ``llm-routing.client_policies`` 覆盖项。"""
    base = config.get(client_type, {}) or {}
    if not isinstance(base, dict):
        base = {}
    routing_root = config.get("llm-routing", {}) or {}
    policies = routing_root.get("client_policies", {}) or {}
    pol = policies.get(client_type, {}) or {}
    if not isinstance(pol, dict):
        pol = {}

    declared_provider = pol.get("provider") or base.get("model-provider") or "siliconflow"
    provider = effective_chat_provider(str(declared_provider))

    # 硅基等：始终用各 ``*-model`` 块的 model（避免百炼专用 id 泄漏到 SiliconFlow）
    if provider == "dashscope":
        model = _resolve_model(pol, base)
        channel = _coerce_channel(pol.get("channel", base.get("llm_channel")))
    else:
        model = str(base.get("model") or pol.get("model") or "").strip()
        channel = LLMChannel.REALTIME

    # thinking：仅 dashscope 强制；可由全局开关与路由表覆盖
    enable_thinking: Optional[bool] = None
    if provider == "dashscope":
        if "enable_thinking" in pol:
            raw_th = pol.get("enable_thinking")
            enable_thinking = False if raw_th is None else bool(raw_th)
        elif _lower(pol.get("thinking_tier")) == "complex":
            enable_thinking = _bool_from_config("ENABLE_THINKING_FOR_COMPLEX", True)
        elif _lower(pol.get("thinking_tier")) == "light":
            enable_thinking = _bool_from_config("ENABLE_THINKING_FOR_LIGHT", False)
        else:
            enable_thinking = False

    timeout = pol.get("request_timeout")
    if timeout is None:
        timeout = base.get("request_timeout")
    timeout_f: Optional[float] = None
    if timeout is not None:
        try:
            timeout_f = float(timeout)
        except (TypeError, ValueError):
            timeout_f = None

    max_retries = pol.get("max_retries")
    if max_retries is None:
        max_retries = config.get("MAX_RETRIES")
    try:
        max_retries_i = int(max_retries) if max_retries is not None else 5
    except (TypeError, ValueError):
        max_retries_i = 5

    if timeout_f is None:
        rts = config.get("REQUEST_TIMEOUT_SECONDS")
        if rts is not None:
            try:
                timeout_f = float(rts)
            except (TypeError, ValueError):
                pass

    concurrency_class = str(pol.get("concurrency_class") or base.get("concurrency_class") or "realtime_high_priority")
    rate_limit_bucket = pol.get("rate_limit_bucket") if provider == "dashscope" else None
    if rate_limit_bucket is not None:
        rate_limit_bucket = str(rate_limit_bucket).strip() or None
    degradable = bool(pol.get("degradable", False)) and provider == "dashscope"
    fb_model = pol.get("fallback_model") or config.get("MODEL_FALLBACK_LIGHT")
    if fb_model is not None:
        fb_model = str(fb_model).strip() or None
    fb_prov = pol.get("fallback_provider")
    if fb_prov is not None:
        fb_prov = normalize_provider_key(str(fb_prov))
    elif fb_model and degradable:
        fb_prov = provider

    nn = node_name or client_type
    if not model:
        logger.warning("llm 路由：%s 未解析到 model，将依赖下游报错", client_type)

    return InvocationPolicy(
        client_type=client_type,
        provider=provider,
        model=model,
        channel=channel,
        enable_thinking=enable_thinking,
        timeout=timeout_f,
        max_retries=max_retries_i,
        concurrency_class=concurrency_class,
        rate_limit_bucket=rate_limit_bucket,
        degradable=degradable,
        fallback_model=fb_model,
        fallback_provider=fb_prov,
        node_name=nn,
    )
