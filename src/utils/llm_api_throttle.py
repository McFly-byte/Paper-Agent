"""远端 LLM API（如硅基流动）的 RPM / TPM 粗粒度限流。

使用双令牌桶：每分钟请求数、每分钟 token 数（按字符粗估），并带 safety_factor 预留余量。
同步嵌入与 asyncio 工作流共用同一套桶（threading 实现），避免并发打穿配额。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Dict, Optional, Set

from src.core.config import config
from src.core.llm_infra.routing import effective_chat_provider
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class _SyncTokenBucket:
    """线程安全的令牌桶；rate 为每秒补充量，capacity 为桶容量。"""

    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        if rate_per_sec <= 0 or capacity <= 0:
            raise ValueError("rate_per_sec and capacity must be positive")
        self._rate = rate_per_sec
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> float:
        return float(self._capacity)

    def consume(self, amount: float) -> None:
        if amount <= 0:
            return
        while True:
            wait: float = 0.0
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = amount - self._tokens
                wait = deficit / self._rate if self._rate > 0 else 0.1
            time.sleep(min(max(wait, 0.01), 2.0))


class RemoteLLMThrottler:
    """RPM（按次）+ TPM（按粗估 token）双桶；一次 API 调用 consume 1 次请求 + 预估 token。"""

    def __init__(
        self,
        requests_per_minute: float,
        tokens_per_minute: float,
        safety_factor: float,
    ) -> None:
        sf = max(0.05, min(1.0, safety_factor))
        rpm = max(1.0, requests_per_minute) * sf
        tpm = max(1.0, tokens_per_minute) * sf
        self._req_bucket = _SyncTokenBucket(rpm / 60.0, rpm)
        # 桶容量必须 ≥ 单次可能 acquire 的 token，否则 consume() 在 amount > capacity 时会永远无法凑齐
        burst = max(
            tpm,
            float(
                max(
                    config.get_int("llm_remote_rate_limit.token_bucket_max_burst_tokens", 100000),
                    int(tpm),
                )
            ),
        )
        self._tok_bucket = _SyncTokenBucket(tpm / 60.0, burst)

    def acquire_blocking(self, requests: int = 1, tokens: int = 0) -> None:
        r = max(0, int(requests))
        t = max(0, int(tokens))
        if r == 0 and t == 0:
            return
        # 先 TPM 再 RPM。单次预扣可能大于桶容量，需分块 consume，否则会永远无法凑齐 amount
        if t:
            max_chunk = max(500.0, self._tok_bucket.capacity * 0.95)
            rem = float(t)
            while rem > 0:
                step = min(rem, max_chunk)
                self._tok_bucket.consume(step)
                rem -= step
        if r:
            self._req_bucket.consume(float(r))

    async def acquire(self, requests: int = 1, tokens: int = 0) -> None:
        await asyncio.to_thread(self.acquire_blocking, requests, tokens)


_throttler_lock = threading.Lock()
_throttler_instance: Optional[RemoteLLMThrottler] = None


def _providers_when_enabled() -> Set[str]:
    raw = config.get_list("llm_remote_rate_limit.apply_when_model_provider_in", None)
    if not raw:
        raw = ["siliconflow"]
    return {str(x).strip().lower() for x in raw if str(x).strip()}


def _default_model_provider_for_throttle() -> str:
    """考虑 DEFAULT_LLM_PROVIDER 强制走百炼时，限流仍应启用。"""
    declared = (config.get("default-model") or {}).get("model-provider") or "siliconflow"
    return effective_chat_provider(str(declared))


def should_apply_remote_llm_throttle() -> bool:
    if not config.get_bool("llm_remote_rate_limit.enabled", True):
        return False
    provider = _default_model_provider_for_throttle()
    if not provider or not isinstance(provider, str):
        return False
    return provider.strip().lower() in _providers_when_enabled()


def get_remote_llm_throttler() -> Optional[RemoteLLMThrottler]:
    """返回全局单例；未启用或不应套用时返回 None。"""
    global _throttler_instance
    if not should_apply_remote_llm_throttle():
        return None
    if _throttler_instance is not None:
        return _throttler_instance
    with _throttler_lock:
        if _throttler_instance is not None:
            return _throttler_instance
        rpm = float(config.get_int("llm_remote_rate_limit.requests_per_minute", 1000))
        tpm = float(config.get_int("llm_remote_rate_limit.tokens_per_minute", 40000))
        safety = config.get_float("llm_remote_rate_limit.safety_factor", 0.85)
        _throttler_instance = RemoteLLMThrottler(rpm, tpm, safety)
        logger.info(
            "远端 LLM 限流已启用：RPM≈%s、TPM≈%s（effective=safety_factor=%s），适用 provider∈%s",
            int(rpm * safety),
            int(tpm * safety),
            safety,
            sorted(_providers_when_enabled()),
        )
        return _throttler_instance


_bucket_lock = threading.Lock()
_bucket_registry: Dict[str, RemoteLLMThrottler] = {}


def _int_for_bucket_field(bucket_cfg: dict, yaml_key: str, env_key: Optional[str], default: int) -> int:
    if isinstance(env_key, str) and env_key.strip():
        raw = config.get(env_key.strip())
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    raw2 = bucket_cfg.get(yaml_key)
    if raw2 is not None:
        try:
            return int(raw2)
        except (TypeError, ValueError):
            pass
    return default


def get_bucket_throttler(bucket_id: Optional[str]) -> Optional[RemoteLLMThrottler]:
    """按 ``llm_rate_limit_buckets.<id>`` 返回独立双桶；未配置或禁用时返回 None。"""
    if not bucket_id or not str(bucket_id).strip():
        return None
    if not config.get_bool("llm_rate_limit_buckets.enabled", True):
        return None
    bid = str(bucket_id).strip()
    if bid in _bucket_registry:
        return _bucket_registry[bid]
    root = config.get("llm_rate_limit_buckets", {}) or {}
    buckets = root.get("buckets", {}) or {}
    bc = buckets.get(bid)
    if not isinstance(bc, dict):
        return None
    with _bucket_lock:
        if bid in _bucket_registry:
            return _bucket_registry[bid]
        rpm = _int_for_bucket_field(
            bc,
            "requests_per_minute",
            str(bc.get("requests_per_minute_env") or "").strip() or None,
            config.get_int("llm_remote_rate_limit.requests_per_minute", 1000),
        )
        tpm = _int_for_bucket_field(
            bc,
            "tokens_per_minute",
            str(bc.get("tokens_per_minute_env") or "").strip() or None,
            config.get_int("llm_remote_rate_limit.tokens_per_minute", 40000),
        )
        safety = config.get_float("llm_remote_rate_limit.safety_factor", 0.85)
        if "safety_factor" in bc and bc["safety_factor"] is not None:
            try:
                safety = float(bc["safety_factor"])
            except (TypeError, ValueError):
                pass
        thr = RemoteLLMThrottler(float(rpm), float(tpm), safety)
        _bucket_registry[bid] = thr
        logger.info(
            "LLM 限流桶已创建 bucket=%s RPM=%s TPM=%s safety=%s",
            bid,
            rpm,
            tpm,
            safety,
        )
        return thr


def get_throttler_for_client_type(client_type: str) -> Optional[RemoteLLMThrottler]:
    """优先使用 ``llm-routing`` 解析出的 rate_limit_bucket；否则回退全局硅基限流。"""
    try:
        from src.core.llm_infra.routing import resolve_invocation_policy

        pol = resolve_invocation_policy(client_type)
        b = get_bucket_throttler(pol.rate_limit_bucket)
        if b is not None:
            return b
    except Exception:
        logger.debug("解析 client_type=%s 限流桶失败，回退全局 throttler", client_type, exc_info=True)
    return get_remote_llm_throttler()


def coarse_token_estimate(
    text: str,
    overhead: int = 2000,
    cap: int = 32000,
) -> int:
    """按字符数粗估 tokenizer token（约 4 字符/token），并上下限裁剪。"""
    base = len(text) // 4 + overhead
    return int(min(max(base, 400), cap))


def configured_token_cap() -> int:
    return max(1024, config.get_int("llm_remote_rate_limit.per_call_token_estimate_cap", 32000))


def reading_throttle_token_estimate(paper_blob: str) -> int:
    """阅读节点专用：长上下文 + 输出占位，避免预扣远低于真实 TPM 导致并发打穿。"""
    div = max(2, config.get_int("llm_remote_rate_limit.reading_chars_per_token_divisor", 3))
    oh = config.get_int("llm_remote_rate_limit.reading_overhead_tokens", 5000)
    out_r = config.get_int("llm_remote_rate_limit.reading_output_reserve_tokens", 8000)
    cap = max(
        4096,
        config.get_int(
            "llm_remote_rate_limit.reading_per_call_token_estimate_cap",
            configured_token_cap(),
        ),
    )
    mult = config.get_float("llm_remote_rate_limit.reading_token_estimate_multiplier", 1.0)
    mult = max(1.0, min(mult, 2.5))
    base = int((len(paper_blob) // div + oh + out_r) * mult)
    return int(min(max(base, 800), cap))
