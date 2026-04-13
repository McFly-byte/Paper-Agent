from __future__ import annotations

import json
import uuid
from typing import Any, Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from src.knowledge.knowledge.utils.kb_utils import get_embedding_config
from src.rag.llamaindex import config as li_cfg
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _primary_category(paper: dict[str, Any]) -> str:
    cats = paper.get("categories")
    if isinstance(cats, list) and cats:
        return str(cats[0])
    if isinstance(cats, str) and cats:
        return cats.split(",")[0].strip()
    return ""


def _paper_id(paper: dict[str, Any]) -> str:
    return str(paper.get("arxiv_id") or paper.get("paper_id") or paper.get("id") or "")


def _flatten_extracted(extracted: dict[str, Any]) -> list[tuple[str, str]]:
    """返回 (section_type, text) 列表。"""
    out: list[tuple[str, str]] = []
    if not isinstance(extracted, dict):
        return out

    cp = extracted.get("core_problem")
    if cp:
        out.append(("core_problem", str(cp)))

    km = extracted.get("key_methodology")
    if isinstance(km, dict):
        parts = []
        for k in ("name", "principle", "novelty"):
            v = km.get(k)
            if v:
                parts.append(f"{k}: {v}")
        if parts:
            out.append(("key_methodology", "\n".join(parts)))
    elif km:
        out.append(("key_methodology", str(km)))

    for key, label in (
        ("datasets_used", "datasets_used"),
        ("evaluation_metrics", "evaluation_metrics"),
        ("main_results", "main_results"),
        ("limitations", "limitations"),
    ):
        v = extracted.get(key)
        if isinstance(v, list) and v:
            out.append((label, "\n".join(str(x) for x in v)))
        elif v:
            out.append((label, str(v)))

    contrib = extracted.get("contributions")
    if isinstance(contrib, list) and contrib:
        out.append(("contributions", "\n".join(str(x) for x in contrib)))
    elif contrib:
        out.append(("contributions", str(contrib)))

    return out


def build_documents_from_extracted_pairs(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    run_id: str,
) -> list[Document]:
    """将 (arxiv_paper_dict, extracted_json_dict) 转为带元数据的 Document（尚未句切分）。"""
    docs: list[Document] = []
    abstract_tpl = "摘要（arXiv）:\n{}"

    for paper, extracted in pairs:
        pid = _paper_id(paper)
        title = str(paper.get("title") or "")
        published = str(paper.get("published") or "")
        pcat = _primary_category(paper)
        base_meta = {
            "paper_id": pid,
            "title": title,
            "published": published,
            "primary_category": pcat,
            "source_type": "extracted_structured",
            "run_id": run_id,
            "rag_backend": "llamaindex",
        }

        abst = paper.get("abstract")
        if abst:
            docs.append(
                Document(
                    text=abstract_tpl.format(str(abst).strip()),
                    metadata={**base_meta, "section_type": "abstract"},
                )
            )

        for section_type, text in _flatten_extracted(extracted):
            t = (text or "").strip()
            if not t:
                continue
            docs.append(
                Document(
                    text=f"[{section_type}]\n{t}",
                    metadata={**base_meta, "section_type": section_type},
                )
            )

    return docs


def sentence_chunk_documents(
    documents: list[Document],
    embed_model: Any,
) -> list[Document]:
    """对 Document 做切分：默认 SentenceSplitter；可选 SemanticSplitter（二选一，失败回退）。"""
    if not documents:
        return []
    nodes: list[Any] = []

    if li_cfg.llamaindex_use_semantic_splitter():
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser

            sem = SemanticSplitterNodeParser.from_defaults(
                embed_model=embed_model,
                buffer_size=6,
                breakpoint_percentile_threshold=95,
            )
            nodes = sem.get_nodes_from_documents(documents)
            logger.info("LlamaIndex ingestion: 使用 SemanticSplitter，得到 %s 个节点", len(nodes))
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticSplitter 失败，回退 SentenceSplitter: %s", exc)
            nodes = []

    if not nodes:
        chunk_size = li_cfg.llamaindex_chunk_size()
        chunk_overlap = li_cfg.llamaindex_chunk_overlap()
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info("LlamaIndex ingestion: 使用 SentenceSplitter，得到 %s 个节点", len(nodes))

    out: list[Document] = []
    for n in nodes:
        meta = dict(n.metadata or {})
        cid = meta.get("chunk_id") or f"{meta.get('paper_id', 'na')}__{uuid.uuid4().hex[:12]}"
        meta["chunk_id"] = str(cid)
        meta.setdefault("source_type", "extracted_structured")
        out.append(Document(text=n.get_content(), metadata=meta))
    logger.info("LlamaIndex ingestion: 最终可入库节点 %s 个", len(out))
    return out


def serialize_nodes_for_debug(nodes: list[Document]) -> str:
    try:
        return json.dumps(
            [{"text": d.text[:200], "metadata": d.metadata} for d in nodes[:20]],
            ensure_ascii=False,
        )
    except Exception:  # noqa: BLE001
        return "[]"


def embedding_settings_for_db(embed_info: dict[str, Any] | None) -> dict[str, Any]:
    return get_embedding_config(embed_info or {})
