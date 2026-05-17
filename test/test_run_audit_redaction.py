"""run_audit 脱敏与截断单元测试。"""

from __future__ import annotations

from pydantic import BaseModel

from src.observability.redaction import redact_mapping, redact_value


def test_sensitive_keys_redacted() -> None:
    d = {"api_key": "sk-secret", "nested": {"my_token": "abc", "ok": 1}}
    out = redact_mapping(d, max_text_chars=100)
    assert out["api_key"] == "****"
    assert out["nested"]["my_token"] == "****"
    assert out["nested"]["ok"] == 1


def test_long_string_truncated() -> None:
    s = "x" * 5000
    v = redact_value(s, max_text_chars=10)
    assert isinstance(v, str)
    assert v.endswith("...[truncated]")
    assert len(v) == 10 + len("...[truncated]")


def test_pydantic_basemodel() -> None:
    class M(BaseModel):
        name: str
        api_key: str

    m = M(name="n", api_key="secret")
    out = redact_value(m, max_text_chars=200)
    assert isinstance(out, dict)
    assert out["api_key"] == "****"
    assert out["name"] == "n"


def test_nested_collections() -> None:
    data = {"list": [{"password": "p"}, "plain"], "tuple": (1, 2)}
    out = redact_mapping(data, max_text_chars=50)
    assert out["list"][0]["password"] == "****"
    assert out["list"][1] == "plain"
    assert out["tuple"] == (1, 2)
