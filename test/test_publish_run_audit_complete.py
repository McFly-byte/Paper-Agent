from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.core.state_models import ExecutionState, NodeError, PaperAgentState
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger
from src.observability.audit_analyzer import REQUIRED_ARTIFACTS, analyze_run_audit
from src.observability.run_audit import (
    KEY_CITATION_MAP,
    KEY_EVIDENCE_LEDGER,
    KEY_FAITHFULNESS_REVIEW,
    KEY_REPORT_CITATION_VALIDATION,
    KEY_WRITTED_SECTIONS,
    RunAuditExporter,
)


def test_run_audit_exporter_writes_complete_required_artifacts(tmp_path: Path, monkeypatch):
    store = {
        KEY_EVIDENCE_LEDGER: EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="p1:problem:0",
                    paper_id="p1",
                    title="T1",
                    claim="Claim text long enough.",
                    section_type="introduction",
                    supports_argument="problem",
                    confidence=0.8,
                )
            ]
        ).model_dump(mode="json"),
        KEY_CITATION_MAP: {
            "refs": [
                {
                    "citation_id": "C1",
                    "evidence_id": "p1:problem:0",
                    "paper_id": "p1",
                    "title": "T1",
                    "inline_marker": "[C1]",
                }
            ],
            "evidence_id_to_citation_id": {"p1:problem:0": "C1"},
        },
        KEY_WRITTED_SECTIONS: ["Body [C1]."],
        KEY_FAITHFULNESS_REVIEW: {"verdict": "pass", "summary": "ok"},
        KEY_REPORT_CITATION_VALIDATION: {"verdict": "pass", "summary": "ok"},
    }

    async def fake_get_json(_state, key):
        return store.get(key)

    async def fake_get_text(_state, _key):
        return "analysis text"

    monkeypatch.setattr("src.observability.run_audit.get_json", fake_get_json)
    monkeypatch.setattr("src.observability.run_audit.get_text", fake_get_text)

    state = PaperAgentState(
        run_id="complete-audit",
        user_request="u",
        max_papers=3,
        error=NodeError(),
        report_markdown="# R\n\nBody [C1].\n\n## References\n\n- [C1] T1",
        current_step=ExecutionState.COMPLETED,
        config={"retrieval_mode": "evidence_context_only"},
    )

    async def _run():
        return await RunAuditExporter(str(tmp_path), redact_secrets=False).export(state)

    out = asyncio.run(_run())
    for rel in REQUIRED_ARTIFACTS:
        assert (out / rel).is_file(), rel

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for rel in REQUIRED_ARTIFACTS:
        assert rel in manifest["artifacts"], rel
    analysis = analyze_run_audit(out)
    assert analysis["error_count"] == 0
