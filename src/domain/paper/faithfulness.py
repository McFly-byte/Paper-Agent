"""Faithfulness：规则审查 + 报告聚合模型（可选 LLM 由节点层调用）。"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.domain.paper.citation import CitationMap


class UnsupportedClaim(BaseModel):
    claim: str
    reason: str
    severity: Literal["low", "medium", "high"] = "medium"
    suggested_action: str | None = None


class CitationIssue(BaseModel):
    marker: str
    issue_type: Literal["unknown_marker", "missing_marker", "unused_reference", "invalid_evidence_id"]
    detail: str


class SectionFaithfulnessResult(BaseModel):
    section_title: str
    verdict: Literal["pass", "warning", "fail"] = "pass"
    supported_claim_count: int = 0
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    citation_issues: list[CitationIssue] = Field(default_factory=list)
    evidence_ids_used: list[str] = Field(default_factory=list)
    citation_ids_used: list[str] = Field(default_factory=list)
    revision_instruction: str | None = None


class FaithfulnessReviewReport(BaseModel):
    verdict: Literal["pass", "warning", "fail"] = "pass"
    section_results: list[SectionFaithfulnessResult] = Field(default_factory=list)
    global_citation_issues: list[CitationIssue] = Field(
        default_factory=list,
        description="跨章节未使用的 CitationMap 条目",
    )
    total_unsupported_claims: int = 0
    total_citation_issues: int = 0
    summary: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


_MARKER_RE = re.compile(r"\[C(\d+)\]")


_FACT_EN = re.compile(
    r"\b(achieves|improves|outperforms|demonstrates|shows|reports|proves|state[s]? that)\b",
    re.IGNORECASE,
)
_FACT_ZH = re.compile(
    r"(提出|证明|显著|优于|实验表明|结果显示|达到|提升|表明|证实|超过|降低了|提高了)",
)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 12]


def _sentence_has_marker(s: str) -> bool:
    return bool(_MARKER_RE.search(s))


def analyze_section_text(
    section_title: str,
    text: str,
    cmap: CitationMap,
    *,
    long_section_chars: int = 500,
) -> SectionFaithfulnessResult:
    """确定性章节审查（不调用 LLM）。"""
    valid_cids = cmap.citation_ids()
    used_markers = _MARKER_RE.findall(text or "")
    used_display = [f"[C{m}]" for m in used_markers]
    citation_ids_used = sorted({f"C{m}" for m in used_markers})

    issues: list[CitationIssue] = []
    for m in used_markers:
        cid = f"C{m}"
        if cid not in valid_cids:
            issues.append(
                CitationIssue(
                    marker=f"[C{m}]",
                    issue_type="unknown_marker",
                    detail=f"marker {cid} not defined in CitationMap",
                )
            )

    used_cids_in_text = {f"C{m}" for m in used_markers if f"C{m}" in valid_cids}

    unsupported: list[UnsupportedClaim] = []
    plain = text or ""
    if len(plain) >= long_section_chars and not used_display:
        issues.append(
            CitationIssue(
                marker="",
                issue_type="missing_marker",
                detail="long section without any [Cn] citation markers",
            )
        )

    for sent in _split_sentences(plain):
        if len(sent) < 10:
            continue
        if _sentence_has_marker(sent):
            continue
        zh_hit = bool(_FACT_ZH.search(sent))
        en_hit = bool(_FACT_EN.search(sent))
        if not zh_hit and not en_hit:
            continue
        # 英文启发式短语噪声多，保留较长句门槛；中文句通常较短，单独放宽
        if en_hit and len(sent) < 24:
            continue
        unsupported.append(
            UnsupportedClaim(
                claim=sent[:400],
                reason="factual_language without citation marker",
                severity="medium",
                suggested_action="Add an existing [Cn] marker from the CitationMap or soften the claim.",
            )
        )

    verdict: Literal["pass", "warning", "fail"] = "pass"
    if any(i.issue_type == "unknown_marker" for i in issues):
        verdict = "fail"
    elif issues or unsupported:
        verdict = "warning"

    return SectionFaithfulnessResult(
        section_title=section_title,
        verdict=verdict,
        supported_claim_count=max(0, len(used_cids_in_text)),
        unsupported_claims=unsupported[:30],
        citation_issues=issues[:50],
        evidence_ids_used=[],
        citation_ids_used=citation_ids_used,
        revision_instruction=None,
    )


def build_faithfulness_report(
    section_texts: list[str],
    cmap: CitationMap | None,
    *,
    long_section_chars: int = 500,
) -> FaithfulnessReviewReport:
    if cmap is None or not cmap.refs:
        return FaithfulnessReviewReport(
            verdict="warning",
            section_results=[],
            global_citation_issues=[
                CitationIssue(
                    marker="",
                    issue_type="invalid_evidence_id",
                    detail="CitationMap missing; deterministic faithfulness checks cannot validate markers",
                )
            ],
            total_unsupported_claims=0,
            total_citation_issues=1,
            summary="no_citation_map_warning_deterministic_checks_incomplete",
        )

    results: list[SectionFaithfulnessResult] = []
    for i, body in enumerate(section_texts):
        title = f"Section {i + 1}"
        results.append(
            analyze_section_text(title, body, cmap, long_section_chars=long_section_chars),
        )

    all_text = "\n".join(section_texts)
    all_markers = {f"C{m}" for m in _MARKER_RE.findall(all_text)}
    global_issues: list[CitationIssue] = []
    for r in cmap.refs:
        if r.citation_id not in all_markers:
            global_issues.append(
                CitationIssue(
                    marker=r.inline_marker,
                    issue_type="unused_reference",
                    detail=f"{r.citation_id} not referenced in any section body",
                )
            )

    tu = sum(len(r.unsupported_claims) for r in results)
    tc = sum(len(r.citation_issues) for r in results) + len(global_issues)
    overall: Literal["pass", "warning", "fail"] = "pass"
    if any(r.verdict == "fail" for r in results):
        overall = "fail"
    elif any(r.verdict == "warning" for r in results) or global_issues:
        overall = "warning"

    summary = (
        f"verdict={overall}; sections={len(results)}; "
        f"unsupported_claims={tu}; citation_issues={tc}; global_unused_refs={len(global_issues)}"
    )
    return FaithfulnessReviewReport(
        verdict=overall,
        section_results=results,
        global_citation_issues=global_issues,
        total_unsupported_claims=tu,
        total_citation_issues=tc,
        summary=summary,
    )


def merge_llm_section_result(
    base: SectionFaithfulnessResult,
    llm: SectionFaithfulnessResult | None,
) -> SectionFaithfulnessResult:
    """保守合并：LLM 仅可提升严重度，不得删除确定性 citation unknown 问题。"""
    if llm is None:
        return base
    merged_issues = list(base.citation_issues)
    merged_issues.extend([x for x in llm.citation_issues if x not in merged_issues][:20])
    merged_unsup = list(base.unsupported_claims)
    merged_unsup.extend(llm.unsupported_claims[:20])
    v = base.verdict
    if llm.verdict == "fail" or v == "fail":
        v = "fail"
    elif llm.verdict == "warning" or v == "warning":
        v = "warning"
    else:
        v = "pass"
    return base.model_copy(
        update={
            "verdict": v,
            "citation_issues": merged_issues[:80],
            "unsupported_claims": merged_unsup[:50],
            "revision_instruction": llm.revision_instruction or base.revision_instruction,
        }
    )


def merge_llm_report(rule_report: FaithfulnessReviewReport, llm_sections: list[SectionFaithfulnessResult] | None) -> FaithfulnessReviewReport:
    if not llm_sections or len(llm_sections) != len(rule_report.section_results):
        return rule_report
    merged = [
        merge_llm_section_result(a, b) for a, b in zip(rule_report.section_results, llm_sections)
    ]
    tu = sum(len(r.unsupported_claims) for r in merged)
    tc = sum(len(r.citation_issues) for r in merged) + len(rule_report.global_citation_issues)
    ov: Literal["pass", "warning", "fail"] = "pass"
    if any(r.verdict == "fail" for r in merged):
        ov = "fail"
    elif any(r.verdict == "warning" for r in merged) or rule_report.global_citation_issues:
        ov = "warning"
    return FaithfulnessReviewReport(
        verdict=ov,
        section_results=merged,
        global_citation_issues=rule_report.global_citation_issues,
        total_unsupported_claims=tu,
        total_citation_issues=tc,
        summary=f"{rule_report.summary} | llm_merged=true",
    )
