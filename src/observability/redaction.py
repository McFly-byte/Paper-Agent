"""审计导出用脱敏与截断（递归）。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

SENSITIVE_KEY_PATTERNS = [
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "cookie",
]


def _key_is_sensitive(key: str) -> bool:
    lk = (key or "").lower()
    return any(pat in lk for pat in SENSITIVE_KEY_PATTERNS)


def _truncate_str(s: str, max_text_chars: int) -> str:
    if max_text_chars <= 0:
        return ""
    if len(s) <= max_text_chars:
        return s
    return s[:max_text_chars] + "...[truncated]"


def redact_value(value: Any, max_text_chars: int = 4000) -> Any:
    """递归脱敏与截断字符串；敏感 dict 键对应的值替换为 ****。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_str(value, max_text_chars)
    if isinstance(value, BaseModel):
        try:
            dumped = value.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            dumped = str(value)
        return redact_value(dumped, max_text_chars=max_text_chars)
    if isinstance(value, Mapping):
        return redact_mapping(value, max_text_chars=max_text_chars)
    if isinstance(value, (list, tuple)):
        seq = [redact_value(x, max_text_chars=max_text_chars) for x in value]
        return seq if isinstance(value, list) else tuple(seq)
    try:
        import json as _json

        _json.dumps(value)
        return value
    except Exception:  # noqa: BLE001
        raw = str(value)
        return _truncate_str(raw, max_text_chars)


def redact_mapping(data: Mapping[str, Any], max_text_chars: int = 4000) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        sk = str(k)
        if _key_is_sensitive(sk):
            out[sk] = "****"
        else:
            out[sk] = redact_value(v, max_text_chars=max_text_chars)
    return out
