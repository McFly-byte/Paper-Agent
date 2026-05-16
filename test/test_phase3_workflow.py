"""Phase 3 编排路由与建图烟测。

在无完整 Poetry 依赖的环境下，通过最小 sys.modules 桩加载 `orchestrator`（与 test_phase2_nodes 思路一致）。
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from unittest.mock import MagicMock


def _install_langchain_import_stubs() -> None:
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

    if importlib.util.find_spec("langchain_openai") is None:
        lco = types.ModuleType("langchain_openai")
        lco.ChatOpenAI = MagicMock
        sys.modules["langchain_openai"] = lco


def _install_autogen_stubs() -> None:
    msgs_mod = types.ModuleType("autogen_agentchat.messages")
    msgs_mod.TextMessage = MagicMock
    msgs_mod.BaseAgentEvent = MagicMock
    msgs_mod.BaseChatMessage = MagicMock
    msgs_mod.StructuredMessage = MagicMock

    agents_mod = types.ModuleType("autogen_agentchat.agents")
    agents_mod.UserProxyAgent = MagicMock
    agents_mod.AssistantAgent = MagicMock
    agents_mod.BaseChatAgent = MagicMock

    base_mod = types.ModuleType("autogen_agentchat.base")
    base_mod.TaskResult = MagicMock
    base_mod.Response = MagicMock

    root = types.ModuleType("autogen_agentchat")
    root.__path__ = []  # 使子模块可被 importlib 识别为包
    setattr(root, "agents", agents_mod)
    setattr(root, "messages", msgs_mod)
    setattr(root, "base", base_mod)

    conditions_mod = types.ModuleType("autogen_agentchat.conditions")
    conditions_mod.TextMentionTermination = MagicMock
    setattr(root, "conditions", conditions_mod)

    teams_mod = types.ModuleType("autogen_agentchat.teams")
    teams_mod.SelectorGroupChat = MagicMock
    setattr(root, "teams", teams_mod)

    msgs_mod.ModelClientStreamingChunkEvent = MagicMock
    msgs_mod.ThoughtEvent = MagicMock
    msgs_mod.ToolCallSummaryMessage = MagicMock
    msgs_mod.ToolCallExecutionEvent = MagicMock

    sys.modules["autogen_agentchat"] = root
    sys.modules["autogen_agentchat.agents"] = agents_mod
    sys.modules["autogen_agentchat.messages"] = msgs_mod
    sys.modules["autogen_agentchat.base"] = base_mod
    sys.modules["autogen_agentchat.conditions"] = conditions_mod
    sys.modules["autogen_agentchat.teams"] = teams_mod

    core_mod = types.ModuleType("autogen_core")
    core_mod.__path__ = []
    core_mod.CancellationToken = MagicMock
    core_mod.RoutedAgent = MagicMock

    def message_handler(*_args, **_kwargs):
        def _wrap(target):
            return target

        return _wrap

    core_mod.message_handler = message_handler

    tools_mod = types.ModuleType("autogen_core.tools")
    tools_mod.FunctionTool = MagicMock
    setattr(core_mod, "tools", tools_mod)
    ac_models = types.ModuleType("autogen_core.models")
    ac_models.ModelInfo = MagicMock
    ac_models.UserMessage = MagicMock
    setattr(core_mod, "models", ac_models)
    sys.modules.setdefault("autogen_core", core_mod)
    sys.modules.setdefault("autogen_core.models", ac_models)
    sys.modules.setdefault("autogen_core.tools", tools_mod)

    openai_sub = types.ModuleType("autogen_ext.models.openai")
    openai_sub.OpenAIChatCompletionClient = MagicMock
    models_pkg = types.ModuleType("autogen_ext.models")
    ext_root = types.ModuleType("autogen_ext")
    setattr(ext_root, "models", models_pkg)
    setattr(models_pkg, "openai", openai_sub)
    sys.modules.setdefault("autogen_ext", ext_root)
    sys.modules.setdefault("autogen_ext.models", models_pkg)
    sys.modules.setdefault("autogen_ext.models.openai", openai_sub)


def _install_langsmith_traceable_stub() -> None:
    ls = types.ModuleType("langsmith")

    def traceable(**_kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    ls.traceable = traceable
    ls.aevaluate = MagicMock()
    sys.modules.setdefault("langsmith", ls)

    schemas = types.ModuleType("langsmith.schemas")
    schemas.Example = MagicMock
    schemas.Run = MagicMock
    sys.modules.setdefault("langsmith.schemas", schemas)


def _install_arxiv_stub() -> None:
    arxiv_mod = types.ModuleType("arxiv")

    class _SC:
        Relevance = object()
        SubmittedDate = object()

    class _SO:
        Descending = object()

    arxiv_mod.SortCriterion = _SC
    arxiv_mod.SortOrder = _SO
    arxiv_mod.HTTPError = type("HTTPError", (Exception,), {})
    arxiv_mod.Client = MagicMock
    arxiv_mod.Search = MagicMock
    arxiv_mod.Result = MagicMock
    sys.modules.setdefault("arxiv", arxiv_mod)


_install_langchain_import_stubs()
_install_autogen_stubs()
_install_langsmith_traceable_stub()
_install_arxiv_stub()

from src.agents.orchestrator import (  # noqa: E402
    PaperAgentOrchestrator,
    route_after_faithfulness_review,
    route_after_writing,
)
from src.core.state_models import NodeError, PaperAgentState, State  # noqa: E402


def test_graph_contains_faithfulness_node():
    orch = PaperAgentOrchestrator(asyncio.Queue())
    nodes = getattr(orch.graph, "nodes", None)
    assert nodes is not None
    assert "faithfulness_review_node" in nodes


def test_route_after_writing_goes_to_faithfulness():
    st: State = {"value": PaperAgentState(run_id="r", user_request="u", error=NodeError(), config={})}
    assert route_after_writing(st) == "faithfulness_review_node"


def test_route_after_faithfulness_goes_to_report():
    st: State = {"value": PaperAgentState(run_id="r", user_request="u", error=NodeError(), config={})}
    assert route_after_faithfulness_review(st) == "report_node"


def test_strict_faithfulness_error_routes_away_from_report():
    st: State = {"value": PaperAgentState(run_id="r", user_request="u", error=NodeError(), config={})}
    st["value"].error.faithfulness_review_node_error = "failed review"
    assert route_after_faithfulness_review(st) != "report_node"


def test_writing_error_skips_faithfulness_chain():
    st: State = {"value": PaperAgentState(run_id="r", user_request="u", error=NodeError(), config={})}
    st["value"].error.writing_node_error = "write failed"
    assert route_after_writing(st) != "faithfulness_review_node"
