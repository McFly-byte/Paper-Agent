"""Phase 2 节点最小异步烟测（mock 外部 IO）。"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_langchain_import_stubs() -> None:
    """最小环境可能未装全量 langchain 依赖，避免收集用例时加载索引链失败。"""
    m_lcts = MagicMock()
    m_lcts.RecursiveCharacterTextSplitter = MagicMock
    sys.modules.setdefault("langchain_text_splitters", m_lcts)

    m_lcdl = MagicMock()
    for name in (
        "CSVLoader",
        "JSONLoader",
        "PyPDFLoader",
        "TextLoader",
        "UnstructuredHTMLLoader",
        "UnstructuredMarkdownLoader",
        "UnstructuredWordDocumentLoader",
    ):
        setattr(m_lcdl, name, MagicMock)
    sys.modules.setdefault("langchain_community.document_loaders", m_lcdl)
    sys.modules.setdefault("langchain_community", types.ModuleType("langchain_community"))


_install_langchain_import_stubs()

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.researcher.evidence_index_node import evidence_index_node
from src.agents.researcher.paper_filter_node import paper_filter_node
from src.core.state_models import NodeError, PaperAgentState, PaperRunContext, State
from src.services.run_tmp_state_store import KEY_EXTRACTED_DATA, KEY_READING_SUCCESSFUL_PAPERS


def test_paper_filter_writes_filtered_papers():
    st: State = {
        "value": PaperAgentState(
            run_id="run_pf",
            user_request="machine learning",
            error=NodeError(),
            config={"tmp_db_id": "fake"},
            search_results=[
                {"title": "Neural nets for ML", "abstract": "deep learning " * 20, "paper_id": "1"},
                {"title": "Cooking recipes", "abstract": "cake " * 20, "paper_id": "2"},
            ],
            max_papers=10,
        )
    }
    q: asyncio.Queue = asyncio.Queue()
    rt = MagicMock()
    rt.context = PaperRunContext(state_queue=q, user_proxy=MagicMock())

    async def _run():
        with (
            patch("src.agents.researcher.paper_filter_node.ensure_run_tmp_kb", new_callable=AsyncMock),
            patch("src.agents.researcher.paper_filter_node.put_json", new_callable=AsyncMock) as pj,
            patch("src.agents.researcher.paper_filter_node.get_json", new_callable=AsyncMock, return_value=None),
        ):
            out = await paper_filter_node(st, rt)
        return out, pj

    out, pj = asyncio.run(_run())
    val = out["value"]
    assert val.filtered_papers
    assert isinstance(val.search_results, list)
    assert pj.await_count >= 2


def test_paper_filter_exception_fallback():
    st: State = {
        "value": PaperAgentState(
            run_id="run_pfx",
            user_request="x",
            error=NodeError(),
            config={"tmp_db_id": "fake"},
            search_results=[{"title": "A", "abstract": "b" * 100, "paper_id": "z"}],
            max_papers=5,
        )
    }
    q: asyncio.Queue = asyncio.Queue()
    rt = MagicMock()
    rt.context = PaperRunContext(state_queue=q, user_proxy=MagicMock())

    async def boom(*_a, **_k):
        raise RuntimeError("disk full")

    async def _run():
        with (
            patch("src.agents.researcher.paper_filter_node.ensure_run_tmp_kb", new_callable=AsyncMock),
            patch("src.agents.researcher.paper_filter_node.put_json", new_callable=AsyncMock, side_effect=boom),
            patch("src.agents.researcher.paper_filter_node.get_json", new_callable=AsyncMock, return_value=None),
        ):
            return await paper_filter_node(st, rt)

    val = asyncio.run(_run())["value"]
    assert val.search_results and val.search_results[0].get("paper_id") == "z"
    assert val.workflow_errors


def test_evidence_index_from_tmp_keys():
    extracted = {
        "papers": [
            {
                "core_problem": "Problem statement is long enough for evidence extraction here.",
                "key_methodology": {"name": "M", "principle": "Principle text long enough.", "novelty": "N"},
                "datasets_used": ["D1"],
                "evaluation_metrics": ["m1"],
                "main_results": "Results are strong with measurable improvements over baseline.",
                "limitations": "Limited to single domain evaluation in this work.",
                "contributions": ["Contribution one is substantive and non-trivial."],
            }
        ]
    }
    metas = [{"paper_id": "p1", "title": "T1", "authors": ["A"], "published": "2024"}]

    st: State = {
        "value": PaperAgentState(
            run_id="run_ei",
            user_request="u",
            error=NodeError(),
            config={"tmp_db_id": "fake"},
        )
    }
    q: asyncio.Queue = asyncio.Queue()
    rt = MagicMock()
    rt.context = PaperRunContext(state_queue=q, user_proxy=MagicMock())

    async def fake_get_json(state, key):
        if key == KEY_EXTRACTED_DATA:
            return extracted
        if key == KEY_READING_SUCCESSFUL_PAPERS:
            return metas
        return None

    async def _run():
        with (
            patch("src.agents.researcher.evidence_index_node.ensure_run_tmp_kb", new_callable=AsyncMock),
            patch("src.agents.researcher.evidence_index_node.put_json", new_callable=AsyncMock) as pj,
            patch("src.agents.researcher.evidence_index_node.get_json", side_effect=fake_get_json),
        ):
            out = await evidence_index_node(st, rt)
        return out, pj

    out, pj = asyncio.run(_run())
    val = out["value"]
    assert val.evidence_ledger is not None
    assert len(val.evidence_ledger.items) >= 1
    assert pj.await_count >= 1


def test_evidence_index_no_extracted_no_crash():
    st: State = {
        "value": PaperAgentState(run_id="run_e2", user_request="u", error=NodeError(), config={"tmp_db_id": "fake"})
    }
    q: asyncio.Queue = asyncio.Queue()
    rt = MagicMock()
    rt.context = PaperRunContext(state_queue=q, user_proxy=MagicMock())

    async def _run():
        with (
            patch("src.agents.researcher.evidence_index_node.ensure_run_tmp_kb", new_callable=AsyncMock),
            patch("src.agents.researcher.evidence_index_node.put_json", new_callable=AsyncMock),
            patch("src.agents.researcher.evidence_index_node.get_json", new_callable=AsyncMock, return_value=None),
        ):
            return await evidence_index_node(st, rt)

    val = asyncio.run(_run())["value"]
    assert val.evidence_ledger is not None
    assert val.evidence_ledger.items == []
