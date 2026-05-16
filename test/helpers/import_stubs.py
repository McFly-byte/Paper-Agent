"""仅用于 pytest **import 烟测**：补齐缺失第三方模块的最小桩。

说明：
- 不代表真实 AutoGen / LangChain / arXiv 行为；**禁止**用本模块桩替代集成测试。
- **禁止**整包覆盖已安装的 ``langchain_core``（会破坏 LangGraph 对 ``AnyMessage`` 等符号的导入）。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import MagicMock


def install_minimal_langchain_document_loader_stubs() -> None:
    """Phase 2 节点等：仅需 ``langchain_community.document_loaders`` 与 ``langchain_text_splitters`` 存在即可 import。"""
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


def install_orchestrator_import_smoke_stubs() -> None:
    """加载 ``src.agents.orchestrator`` 前安装：langchain 文档桩 + 条件 langchain_openai + autogen + langsmith + arxiv。"""
    install_minimal_langchain_document_loader_stubs()

    if importlib.util.find_spec("langchain_openai") is None:
        lco = types.ModuleType("langchain_openai")
        lco.ChatOpenAI = MagicMock
        sys.modules["langchain_openai"] = lco

    msgs_mod = types.ModuleType("autogen_agentchat.messages")
    msgs_mod.TextMessage = MagicMock
    msgs_mod.BaseAgentEvent = MagicMock
    msgs_mod.BaseChatMessage = MagicMock
    msgs_mod.StructuredMessage = MagicMock
    msgs_mod.ModelClientStreamingChunkEvent = MagicMock
    msgs_mod.ThoughtEvent = MagicMock
    msgs_mod.ToolCallSummaryMessage = MagicMock
    msgs_mod.ToolCallExecutionEvent = MagicMock

    agents_mod = types.ModuleType("autogen_agentchat.agents")
    agents_mod.UserProxyAgent = MagicMock
    agents_mod.AssistantAgent = MagicMock
    agents_mod.BaseChatAgent = MagicMock

    base_mod = types.ModuleType("autogen_agentchat.base")
    base_mod.TaskResult = MagicMock
    base_mod.Response = MagicMock

    root = types.ModuleType("autogen_agentchat")
    root.__path__ = []
    setattr(root, "agents", agents_mod)
    setattr(root, "messages", msgs_mod)
    setattr(root, "base", base_mod)

    conditions_mod = types.ModuleType("autogen_agentchat.conditions")
    conditions_mod.TextMentionTermination = MagicMock
    setattr(root, "conditions", conditions_mod)

    teams_mod = types.ModuleType("autogen_agentchat.teams")
    teams_mod.SelectorGroupChat = MagicMock
    setattr(root, "teams", teams_mod)

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
