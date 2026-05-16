"""Phase 3 确定性闭环烟测：Evidence → CitationMap → evidence bundle → faithfulness → report validation。

不调用真实 LLM / arXiv / Chroma / FastAPI。
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from src.agents.planner.models import ResearchPlan, WritingTask
from src.agents.writing_evidence_helpers import prepare_evidence_bound_writing_bundle
from src.core.state_models import NodeError, PaperAgentState
from src.domain.paper.citation import CitationMap
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger
from src.domain.paper.faithfulness import build_faithfulness_report
from src.domain.paper.report_citation_validation import validate_report_citations
from src.services.run_tmp_state_store import KEY_CITATION_MAP


def _ledger_two_papers() -> EvidenceLedger:
    return EvidenceLedger(
        items=[
            EvidenceItem(
                evidence_id="p1:m:0",
                paper_id="paper-1",
                title="Paper One",
                authors=["A"],
                year=2023,
                section_type="method",
                claim="Method claim is long enough for the evidence context builder.",
                supports_argument="method",
                confidence=0.8,
            ),
            EvidenceItem(
                evidence_id="p1:r:0",
                paper_id="paper-1",
                title="Paper One",
                authors=["A"],
                year=2023,
                section_type="result",
                claim="Result claim is long enough for the evidence context builder.",
                supports_argument="result",
                confidence=0.8,
            ),
            EvidenceItem(
                evidence_id="p2:p:0",
                paper_id="paper-2",
                title="Paper Two",
                authors=["B"],
                year=2022,
                section_type="introduction",
                claim="Problem claim is long enough for the evidence context builder.",
                supports_argument="problem",
                confidence=0.7,
            ),
            EvidenceItem(
                evidence_id="p2:l:0",
                paper_id="paper-2",
                title="Paper Two",
                authors=["B"],
                year=2022,
                section_type="limitation",
                claim="Limitation claim is long enough for the evidence context builder.",
                supports_argument="limitation",
                confidence=0.7,
            ),
        ]
    )


def test_phase3_deterministic_chain_smoke():
    store: dict[str, object] = {}

    async def fake_put(_state, key: str, obj: object) -> None:
        store[key] = obj

    async def fake_get(_state, key: str) -> object | None:
        return store.get(key)

    plan = ResearchPlan(
        title="t",
        writing_tasks=[
            WritingTask(section_title="方法", section_goal="技术路线"),
            WritingTask(section_title="结果", section_goal="实验效果"),
        ],
    )
    state = PaperAgentState(
        run_id="r_e2e",
        user_request="u",
        error=NodeError(),
        config={"tmp_db_id": "fake"},
        evidence_ledger=_ledger_two_papers(),
        plan=plan,
    )

    async def _run() -> None:
        with (
            patch("src.agents.writing_evidence_helpers.put_json", new=fake_put),
            patch("src.agents.writing_evidence_helpers.get_json", new=fake_get),
        ):
            bundle = await prepare_evidence_bound_writing_bundle(state)
        assert bundle.get("citation_marker_instruction")
        assert state.citation_map is not None
        assert KEY_CITATION_MAP in store
        assert bundle.get("section_evidence_blocks") or bundle.get("evidence_bound_block")

        cmap = CitationMap.model_validate(state.citation_map)
        sections = [
            "该方法提出了结构引导重建策略 [C1]。",
            "实验表明该方法提升了重建质量 [C2]。",
        ]
        fr = build_faithfulness_report(sections, cmap, long_section_chars=5000)
        assert fr.verdict != "fail"

        final = "# Report\n\n" + "\n\n".join(sections) + "\n\n## References\n\n" + cmap.to_markdown_references()
        vr = validate_report_citations(final, cmap)
        assert vr.unknown_markers == []
        assert vr.missing_references_section is False
        assert vr.verdict in ("pass", "warning")

    asyncio.run(_run())
