"""Phase 3 编排路由与建图烟测。

在无完整 Poetry 依赖的环境下，通过最小 sys.modules 桩加载 `orchestrator`（与 test_phase2_nodes 思路一致）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from test.helpers.import_stubs import install_orchestrator_import_smoke_stubs


# 仅用于 import 烟测：不代表真实依赖行为（见 test.helpers.import_stubs）。
install_orchestrator_import_smoke_stubs()

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
