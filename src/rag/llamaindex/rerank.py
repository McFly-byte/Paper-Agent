"""检索第二阶段：可插拔 rerank（TF-IDF / Cross-Encoder / Cohere / Voyage）。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.llamaindex.types import RetrievalHit
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_COHERE_URL = "https://api.cohere.com/v1/rerank"
_VOYAGE_URL = "https://api.voyageai.com/v1/rerank"

# CrossEncoder 单例（按模型名缓存）
_cross_encoder: Any = None
_cross_encoder_name: str | None = None


def _normalize_scores(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _apply_fusion_and_section(
    hits: list[RetrievalHit],
    *,
    dense_n: list[float],
    secondary_n: list[float],
    fuse_alpha: float,
    preferred_sections: list[str] | None,
    section_boost: float,
) -> list[float]:
    pref = set(preferred_sections or [])
    boost = max(0.0, float(section_boost))
    fused: list[float] = []
    for i, h in enumerate(hits):
        base = fuse_alpha * dense_n[i] + (1.0 - fuse_alpha) * secondary_n[i]
        st = str((h.metadata or {}).get("section_type") or "")
        if pref and st in pref:
            base = min(1.0, base + boost)
        fused.append(base)
    return fused


def _build_ranked_hits(
    hits: list[RetrievalHit],
    fused: list[float],
    *,
    dense_n: list[float],
    secondary_n: list[float],
    secondary_key: str,
    top_k: int,
    reranker: str,
    t0: float,
    fuse_alpha: float | None = None,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    order = sorted(range(len(hits)), key=lambda i: fused[i], reverse=True)
    out: list[RetrievalHit] = []
    for i in order[: max(1, top_k)]:
        h = hits[i]
        md = dict(h.metadata or {})
        md["rerank_score"] = round(fused[i], 6)
        md["dense_score_norm"] = round(dense_n[i], 6)
        md[secondary_key] = round(secondary_n[i], 6)
        out.append(
            RetrievalHit(
                text=h.text,
                score=float(fused[i]),
                title=h.title,
                paper_id=h.paper_id,
                source=h.source,
                chunk_id=h.chunk_id,
                metadata=md,
            )
        )
    stats: dict[str, Any] = {
        "reranker": reranker,
        "candidates_in": len(hits),
        "top_k": top_k,
        "fuse_alpha": fuse_alpha,
        "rerank_ms": int((time.perf_counter() - t0) * 1000),
        "candidates_out": len(out),
    }
    return out, stats


def lex_vector_fuse_rerank(
    query: str,
    hits: list[RetrievalHit],
    *,
    top_k: int,
    fuse_alpha: float,
    preferred_sections: list[str] | None,
    section_boost: float,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    t0 = time.perf_counter()
    stats: dict[str, Any] = {
        "reranker": "tfidf_fuse",
        "candidates_in": len(hits),
        "top_k": top_k,
        "fuse_alpha": fuse_alpha,
    }
    if not hits:
        stats["rerank_ms"] = int((time.perf_counter() - t0) * 1000)
        stats["candidates_out"] = 0
        return [], stats

    q = (query or "").strip()
    texts = [(h.text or "")[:4000] for h in hits]
    dense = [float(h.score) for h in hits]
    dense_n = _normalize_scores(dense)

    if not q:
        lex_n = [0.5] * len(hits)
    else:
        try:
            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=4096,
                lowercase=True,
            )
            mat = vec.fit_transform([q] + texts)
            sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
            lex_n = _normalize_scores(list(map(float, sims)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("TF-IDF rerank 失败，回退为纯向量分: %s", exc)
            lex_n = [0.5] * len(hits)

    fused = _apply_fusion_and_section(
        hits,
        dense_n=dense_n,
        secondary_n=lex_n,
        fuse_alpha=fuse_alpha,
        preferred_sections=preferred_sections,
        section_boost=section_boost,
    )
    out, st2 = _build_ranked_hits(
        hits,
        fused,
        dense_n=dense_n,
        secondary_n=lex_n,
        secondary_key="lexical_rerank_norm",
        top_k=top_k,
        reranker="tfidf_fuse",
        t0=t0,
        fuse_alpha=fuse_alpha,
    )
    return out, st2


def _get_cross_encoder(model_name: str) -> Any:
    global _cross_encoder, _cross_encoder_name
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "Cross-Encoder rerank 需要安装 sentence-transformers："
            "poetry install -E rag-cross-encoder 或 pip install sentence-transformers"
        ) from e
    if _cross_encoder is None or _cross_encoder_name != model_name:
        logger.info("加载 CrossEncoder 模型: %s", model_name)
        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_name = model_name
    return _cross_encoder


def cross_encoder_rerank(
    query: str,
    hits: list[RetrievalHit],
    *,
    top_k: int,
    fuse_alpha: float,
    model_name: str,
    preferred_sections: list[str] | None,
    section_boost: float,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    t0 = time.perf_counter()
    if not hits:
        return [], {
            "reranker": "cross_encoder",
            "candidates_in": 0,
            "candidates_out": 0,
            "rerank_ms": int((time.perf_counter() - t0) * 1000),
            "fuse_alpha": fuse_alpha,
        }

    ce = _get_cross_encoder(model_name)
    q = (query or "").strip() or " "
    pairs = [[q, (h.text or "")[:2000]] for h in hits]
    raw = ce.predict(pairs)
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    ce_scores = [float(arr[i]) if i < len(arr) else 0.0 for i in range(len(hits))]

    ce_n = _normalize_scores(ce_scores)
    dense = [float(h.score) for h in hits]
    dense_n = _normalize_scores(dense)
    fused = _apply_fusion_and_section(
        hits,
        dense_n=dense_n,
        secondary_n=ce_n,
        fuse_alpha=fuse_alpha,
        preferred_sections=preferred_sections,
        section_boost=section_boost,
    )
    out, st = _build_ranked_hits(
        hits,
        fused,
        dense_n=dense_n,
        secondary_n=ce_n,
        secondary_key="cross_encoder_norm",
        top_k=top_k,
        reranker="cross_encoder",
        t0=t0,
        fuse_alpha=fuse_alpha,
    )
    st["cross_encoder_model"] = model_name
    return out, st


async def _cohere_rerank_async(
    query: str,
    hits: list[RetrievalHit],
    *,
    top_k: int,
    fuse_alpha: float,
    model: str,
    api_key: str,
    preferred_sections: list[str] | None,
    section_boost: float,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    t0 = time.perf_counter()
    if not hits:
        return [], {
            "reranker": "cohere",
            "candidates_in": 0,
            "candidates_out": 0,
            "rerank_ms": 0,
            "fuse_alpha": fuse_alpha,
        }
    if not api_key:
        raise ValueError("Cohere rerank 需要 COHERE_API_KEY 或 rag.llamaindex.rerank.cohere_api_key")

    documents = [(h.text or "")[:8000] for h in hits]
    payload = {
        "model": model,
        "query": (query or "")[:4000],
        "documents": documents,
        "top_n": min(len(documents), max(top_k * 3, 30)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(_COHERE_URL, json=payload, headers=headers) as resp:
            body = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"Cohere rerank HTTP {resp.status}: {body[:500]}")
            data = json.loads(body)

    results = data.get("results") or []
    scores_by_index: dict[int, float] = {}
    for r in results:
        idx = int(r.get("index", -1))
        sc = r.get("relevance_score")
        if idx >= 0 and sc is not None:
            scores_by_index[idx] = float(sc)

    ce_scores = [scores_by_index.get(i, 0.0) for i in range(len(hits))]
    ce_n = _normalize_scores(ce_scores)
    dense = [float(h.score) for h in hits]
    dense_n = _normalize_scores(dense)
    fused = _apply_fusion_and_section(
        hits,
        dense_n=dense_n,
        secondary_n=ce_n,
        fuse_alpha=fuse_alpha,
        preferred_sections=preferred_sections,
        section_boost=section_boost,
    )
    out, st = _build_ranked_hits(
        hits,
        fused,
        dense_n=dense_n,
        secondary_n=ce_n,
        secondary_key="cohere_rerank_norm",
        top_k=top_k,
        reranker="cohere",
        t0=t0,
        fuse_alpha=fuse_alpha,
    )
    st["cohere_model"] = model
    return out, st


async def _voyage_rerank_async(
    query: str,
    hits: list[RetrievalHit],
    *,
    top_k: int,
    fuse_alpha: float,
    model: str,
    api_key: str,
    preferred_sections: list[str] | None,
    section_boost: float,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    t0 = time.perf_counter()
    if not hits:
        return [], {
            "reranker": "voyage",
            "candidates_in": 0,
            "candidates_out": 0,
            "rerank_ms": 0,
            "fuse_alpha": fuse_alpha,
        }
    if not api_key:
        raise ValueError("Voyage rerank 需要 VOYAGE_API_KEY 或 rag.llamaindex.rerank.voyage_api_key")

    documents = [(h.text or "")[:8000] for h in hits]
    payload = {
        "model": model,
        "query": (query or "")[:4000],
        "documents": documents,
        "top_k": min(len(documents), max(top_k * 3, 30)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(_VOYAGE_URL, json=payload, headers=headers) as resp:
            body = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"Voyage rerank HTTP {resp.status}: {body[:500]}")
            data = json.loads(body)

    results = data.get("data") or data.get("results") or []
    scores_by_index: dict[int, float] = {}
    for r in results:
        idx = int(r.get("index", -1))
        sc = r.get("relevance_score")
        if idx >= 0 and sc is not None:
            scores_by_index[idx] = float(sc)

    ce_scores = [scores_by_index.get(i, 0.0) for i in range(len(hits))]
    ce_n = _normalize_scores(ce_scores)
    dense = [float(h.score) for h in hits]
    dense_n = _normalize_scores(dense)
    fused = _apply_fusion_and_section(
        hits,
        dense_n=dense_n,
        secondary_n=ce_n,
        fuse_alpha=fuse_alpha,
        preferred_sections=preferred_sections,
        section_boost=section_boost,
    )
    out, st = _build_ranked_hits(
        hits,
        fused,
        dense_n=dense_n,
        secondary_n=ce_n,
        secondary_key="voyage_rerank_norm",
        top_k=top_k,
        reranker="voyage",
        t0=t0,
        fuse_alpha=fuse_alpha,
    )
    st["voyage_model"] = model
    return out, st


async def apply_rerank(
    mode: str,
    query: str,
    hits: list[RetrievalHit],
    *,
    top_k: int,
    fuse_alpha: float,
    preferred_sections: list[str] | None,
    section_boost: float,
    cross_encoder_model: str,
    cohere_model: str,
    voyage_model: str,
    cohere_api_key: str,
    voyage_api_key: str,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    """按 mode 分发；失败时回退 tfidf_fuse 并记录原因。"""
    m = (mode or "tfidf_fuse").lower()
    try:
        if m == "tfidf_fuse":
            return await asyncio.to_thread(
                lex_vector_fuse_rerank,
                query,
                hits,
                top_k=top_k,
                fuse_alpha=fuse_alpha,
                preferred_sections=preferred_sections,
                section_boost=section_boost,
            )
        if m == "cross_encoder":
            return await asyncio.to_thread(
                cross_encoder_rerank,
                query,
                hits,
                top_k=top_k,
                fuse_alpha=fuse_alpha,
                model_name=cross_encoder_model,
                preferred_sections=preferred_sections,
                section_boost=section_boost,
            )
        if m == "cohere":
            return await _cohere_rerank_async(
                query,
                hits,
                top_k=top_k,
                fuse_alpha=fuse_alpha,
                model=cohere_model,
                api_key=cohere_api_key,
                preferred_sections=preferred_sections,
                section_boost=section_boost,
            )
        if m == "voyage":
            return await _voyage_rerank_async(
                query,
                hits,
                top_k=top_k,
                fuse_alpha=fuse_alpha,
                model=voyage_model,
                api_key=voyage_api_key,
                preferred_sections=preferred_sections,
                section_boost=section_boost,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("rerank mode=%s 失败，回退 tfidf_fuse: %s", m, exc)
        out, st = await asyncio.to_thread(
            lex_vector_fuse_rerank,
            query,
            hits,
            top_k=top_k,
            fuse_alpha=fuse_alpha,
            preferred_sections=preferred_sections,
            section_boost=section_boost,
        )
        st["rerank_fallback_from"] = m
        st["rerank_fallback_error"] = str(exc)[:300]
        return out, st

    return await asyncio.to_thread(
        lex_vector_fuse_rerank,
        query,
        hits,
        top_k=top_k,
        fuse_alpha=fuse_alpha,
        preferred_sections=preferred_sections,
        section_boost=section_boost,
    )
