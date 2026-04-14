"""LangChain ChatOpenAI 与集中路由对齐（LangSmith 离线评测等）。"""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI

from src.core.llm_infra.factory import connection_params_for_policy
from src.core.llm_infra.routing import resolve_invocation_policy


def build_langchain_chat_openai(
    client_type: str,
    *,
    node_name: Optional[str] = None,
    temperature: float = 0.2,
    timeout_override: Optional[float] = None,
    max_retries: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    pol = resolve_invocation_policy(client_type, node_name=node_name or client_type)
    p: dict[str, Any] = connection_params_for_policy(pol)
    to = float(timeout_override) if timeout_override is not None else float(p["timeout"])
    kwargs: dict[str, Any] = {
        "model": p["model"],
        "api_key": p["api_key"],
        "base_url": str(p["base_url"] or "").strip().rstrip("/"),
        "temperature": temperature,
        "timeout": to,
        "max_retries": int(max_retries if max_retries is not None else p["max_retries"]),
    }
    eb = p.get("extra_body")
    if isinstance(eb, dict) and eb:
        kwargs["extra_body"] = eb
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)
