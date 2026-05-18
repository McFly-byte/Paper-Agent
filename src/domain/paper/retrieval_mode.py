"""Deterministic diagnosis of the writing retrieval mode."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RetrievalMode = Literal["llamaindex", "legacy_tmp_store", "evidence_context_only", "disabled"]


class RetrievalModeDiagnosis(BaseModel):
    retrieval_mode: RetrievalMode
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _configured_backend(config: dict[str, Any] | None) -> str:
    cfg = config or {}
    raw = cfg.get("RAG_BACKEND") or cfg.get("rag_backend") or cfg.get("rag.backend")
    if raw is None:
        rag = cfg.get("rag")
        if isinstance(rag, dict):
            raw = rag.get("backend")
    s = str(raw or "").strip().lower()
    return s if s in {"legacy", "llamaindex"} else "legacy"


def diagnose_retrieval_mode(
    *,
    config: dict[str, Any] | None = None,
    llamaindex_node_count: Any = None,
    kb_write_count: Any = None,
    rag_retrieval_logs: list[dict[str, Any]] | None = None,
    evidence_bound_sections_count: Any = None,
) -> RetrievalModeDiagnosis:
    backend = _configured_backend(config)
    li_nodes = _as_int(llamaindex_node_count, 0)
    kb_writes = _as_int(kb_write_count, 0)
    evidence_sections = _as_int(evidence_bound_sections_count, 0)
    logs = rag_retrieval_logs or []
    rag_calls = len(logs)
    llamaindex_calls = sum(1 for x in logs if isinstance(x, dict) and x.get("rag_backend") == "llamaindex")
    legacy_calls = sum(1 for x in logs if isinstance(x, dict) and x.get("rag_backend") == "legacy")
    positive_calls = 0
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        final_n = _as_int(entry.get("final_context_count") or entry.get("retrieved_count"), 0)
        if final_n > 0:
            positive_calls += 1

    metrics = {
        "configured_backend": backend,
        "llamaindex_node_count": li_nodes,
        "kb_write_count": kb_writes,
        "rag_call_count": rag_calls,
        "rag_positive_call_count": positive_calls,
        "llamaindex_rag_call_count": llamaindex_calls,
        "legacy_rag_call_count": legacy_calls,
        "evidence_bound_sections_count": evidence_sections,
    }
    reasons: list[str] = []

    if backend == "llamaindex" and li_nodes > 0 and llamaindex_calls > 0 and positive_calls > 0:
        return RetrievalModeDiagnosis(
            retrieval_mode="llamaindex",
            reasons=["llamaindex_ingested_and_used_for_writing"],
            metrics=metrics,
        )

    if legacy_calls > 0 and positive_calls > 0:
        return RetrievalModeDiagnosis(
            retrieval_mode="legacy_tmp_store",
            reasons=["legacy_tmp_store_retrieval_used_for_writing"],
            metrics=metrics,
        )

    if evidence_sections > 0:
        if backend == "llamaindex" and li_nodes <= 0:
            reasons.append("llamaindex_configured_but_no_ingested_nodes")
        if rag_calls <= 0:
            reasons.append("writing_rag_call_count_is_zero")
        return RetrievalModeDiagnosis(
            retrieval_mode="evidence_context_only",
            reasons=reasons or ["evidence_context_injected_without_tool_retrieval"],
            metrics=metrics,
        )

    if kb_writes > 0:
        return RetrievalModeDiagnosis(
            retrieval_mode="legacy_tmp_store",
            reasons=["legacy_tmp_store_populated_but_no_evidence_context_detected"],
            metrics=metrics,
        )

    return RetrievalModeDiagnosis(
        retrieval_mode="disabled",
        reasons=["no_retrieval_or_evidence_context_observed"],
        metrics=metrics,
    )
