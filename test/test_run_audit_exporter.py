"""RunAuditExporter 与 tmp store 读取（mock）。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.core.state_models import ExecutionState, NodeError, PaperAgentState
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger
from src.domain.paper.models import PaperCandidate
from src.observability.run_audit import (
    KEY_ANALYSE_RESULTS,
    KEY_CITATION_MAP,
    KEY_EVIDENCE_LEDGER,
    KEY_FAITHFULNESS_REVIEW,
    KEY_FILTERED_PAPERS,
    KEY_REPORT_CITATION_VALIDATION,
    RunAuditExporter,
)
from src.runtime.state import ResearchBrief


def test_exporter_writes_manifest_and_redacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_json: dict = {
        KEY_FILTERED_PAPERS: [
            PaperCandidate(
                paper_id="p1",
                title="T1",
                abstract="A" * 100,
                published="2024-01-01",
                url="http://example.com",
            ).model_dump(mode="json")
        ],
        KEY_EVIDENCE_LEDGER: EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="e1",
                    paper_id="p1",
                    title="T1",
                    claim="claim text",
                    section_type="method",
                    confidence=0.9,
                )
            ]
        ).model_dump(mode="json"),
        KEY_CITATION_MAP: {"refs": [{"inline_marker": "[C1]", "api_key": "should_redact_in_dump"}]},
        KEY_FAITHFULNESS_REVIEW: {"verdict": "pass", "summary": "ok"},
        KEY_REPORT_CITATION_VALIDATION: {"verdict": "pass", "summary": "ok"},
    }

    async def fake_get_json(state: PaperAgentState, blob_key: str):
        return store_json.get(blob_key)

    async def fake_get_text(state: PaperAgentState, blob_key: str):
        if blob_key == KEY_ANALYSE_RESULTS:
            return "## analysis\n\nhello"
        return None

    monkeypatch.setattr("src.observability.run_audit.get_json", fake_get_json)
    monkeypatch.setattr("src.observability.run_audit.get_text", fake_get_text)

    brief = ResearchBrief(
        original_query="q",
        clarified_topic="topic",
    )
    state = PaperAgentState(
        run_id="run-audit-test-1",
        user_request="用户请求示例",
        max_papers=5,
        error=NodeError(),
        brief=brief,
        filtered_papers=[],
        evidence_ledger=None,
        citation_map=None,
        faithfulness_review=None,
        writted_sections=["# Sec\n\nbody " + ("x" * 50)],
        report_markdown="# Report\n\n正文 [C1].",
        outline="# O",
        config={
            "tmp_db_id": "fake",
            "paper_filter_report": {"kept": 1, "secret_token": "hide_me"},
            "search_gate_context": {"querys": ["all:ti:diffusion"], "start_date": None, "end_date": None},
        },
        trace_events=[{"event_type": "TEST", "node_name": "n", "status": "finished"}],
        current_step=ExecutionState.COMPLETED,
    )

    async def _run() -> Path:
        out_root = tmp_path / "audits"
        exp = RunAuditExporter(
            str(out_root),
            max_text_chars=80,
            max_papers=5,
            max_evidence_items=10,
            include_full_report=True,
            include_written_sections=True,
            include_evidence_claims=True,
            redact_secrets=True,
        )
        return await exp.export(state)

    path = asyncio.run(_run())
    assert path.is_dir()
    mf = path / "manifest.json"
    assert mf.is_file()
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-audit-test-1"

    assert (path / "report" / "final_report.md").is_file()

    ev = path / "evidence" / "evidence_items.sample.json"
    assert "evidence_id" in ev.read_text(encoding="utf-8")

    cmap = json.loads((path / "writing" / "citation_map.json").read_text(encoding="utf-8"))
    dumped = json.dumps(cmap, ensure_ascii=False)
    assert "should_redact" not in dumped
    assert "****" in dumped

    fraw = (path / "paper_filter" / "filter_report.json").read_text(encoding="utf-8")
    assert "hide_me" not in fraw


def test_exporter_no_full_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_json(state: PaperAgentState, blob_key: str):
        return None

    async def fake_get_text(state: PaperAgentState, blob_key: str):
        return None

    monkeypatch.setattr("src.observability.run_audit.get_json", fake_get_json)
    monkeypatch.setattr("src.observability.run_audit.get_text", fake_get_text)

    state = PaperAgentState(
        run_id="run-2",
        user_request="r",
        max_papers=3,
        error=NodeError(),
        report_markdown="# long\n" + "z" * 500,
    )

    async def _run() -> Path:
        exp = RunAuditExporter(
            str(tmp_path / "a"),
            include_full_report=False,
            include_written_sections=False,
            redact_secrets=True,
        )
        return await exp.export(state)

    path = asyncio.run(_run())
    text = (path / "report" / "final_report.md").read_text(encoding="utf-8")
    assert "not_available" in text or "run_audit_include_full_report" in text
