"""报告级引用校验：终稿 Markdown 中的 [Cn] 与 CitationMap / References 节一致性（纯规则，无 LLM）。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.domain.paper.citation import CitationMap

_MARKER_RE = re.compile(r"\[C(\d+)\]")
_REF_HEADING_RE = re.compile(r"(?im)^##\s+References\b")


class ReportCitationValidationResult(BaseModel):
    verdict: Literal["pass", "warning", "fail"] = "pass"
    markers_in_report: list[str] = Field(default_factory=list)
    defined_markers: list[str] = Field(default_factory=list)
    unknown_markers: list[str] = Field(default_factory=list)
    unused_references: list[str] = Field(default_factory=list)
    missing_references_section: bool = False
    citation_marker_count: int = 0
    reference_count: int = 0
    issues: list[str] = Field(default_factory=list)
    summary: str = ""


def _body_before_references(markdown: str) -> str:
    m = _REF_HEADING_RE.search(markdown or "")
    if not m:
        return markdown or ""
    return (markdown or "")[: m.start()]


def _has_references_heading(markdown: str) -> bool:
    return bool(_REF_HEADING_RE.search(markdown or ""))


def references_heading_present(markdown: str) -> bool:
    """终稿是否已含 ``## References`` 标题（大小写不敏感）。"""
    return _has_references_heading(markdown)


def _unique_markers_ordered(markers: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in markers:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def validate_report_citations(
    report_markdown: str,
    citation_map: CitationMap | None,
    *,
    require_references_section: bool = True,
    allow_unused_references: bool = True,
) -> ReportCitationValidationResult:
    """校验终稿中的 ``[Cn]`` 与 ``CitationMap`` 是否一致；不编造、不调用 LLM。"""
    text = report_markdown or ""
    issues: list[str] = []

    if citation_map is None or not citation_map.refs:
        return ReportCitationValidationResult(
            verdict="warning",
            markers_in_report=[],
            defined_markers=[],
            unknown_markers=[],
            unused_references=[],
            missing_references_section=False,
            citation_marker_count=0,
            reference_count=0,
            issues=["no_citation_map_validation_incomplete"],
            summary="no_citation_map_validation_incomplete",
        )

    defined_display = [r.inline_marker for r in citation_map.refs if r.inline_marker]
    defined_ids = {r.citation_id for r in citation_map.refs}

    found_nums = _MARKER_RE.findall(text)
    markers_in_report = _unique_markers_ordered([f"[C{n}]" for n in found_nums])
    citation_marker_count = len(markers_in_report)

    unknown_markers: list[str] = []
    for disp in markers_in_report:
        cid = disp.strip("[]")
        if cid not in defined_ids:
            unknown_markers.append(disp)
    if unknown_markers:
        issues.append(f"unknown_markers: {unknown_markers}")

    body = _body_before_references(text)
    body_nums = set(_MARKER_RE.findall(body))
    body_markers = {f"[C{n}]" for n in body_nums}
    body_markers_ordered = _unique_markers_ordered([f"[C{n}]" for n in _MARKER_RE.findall(body)])

    unused_references: list[str] = []
    for r in citation_map.refs:
        m = r.inline_marker or f"[{r.citation_id}]"
        if m and m not in body_markers:
            unused_references.append(m)
    if unused_references:
        issues.append(f"unused_in_body_before_references_heading: {unused_references}")

    missing_references_section = not _has_references_heading(text)
    if require_references_section and missing_references_section:
        issues.append("missing ## References section")

    reference_count = len(citation_map.refs)

    if unknown_markers:
        verdict = "fail"
    elif missing_references_section:
        verdict = "warning"
    elif not body_markers_ordered:
        verdict = "warning"
        issues.append("no_citation_markers_in_body_before_references_while_citation_map_nonempty")
    elif unused_references and not allow_unused_references:
        verdict = "fail"
        issues.append("unused_references_not_allowed")
    elif unused_references:
        verdict = "warning"
    else:
        verdict = "pass"

    summary = (
        f"verdict={verdict}; markers={citation_marker_count}; refs={reference_count}; "
        f"unknown={len(unknown_markers)}; unused_body={len(unused_references)}; "
        f"missing_ref_heading={missing_references_section}"
    )
    return ReportCitationValidationResult(
        verdict=verdict,
        markers_in_report=markers_in_report,
        defined_markers=defined_display,
        unknown_markers=unknown_markers,
        unused_references=unused_references,
        missing_references_section=missing_references_section,
        citation_marker_count=citation_marker_count,
        reference_count=reference_count,
        issues=issues,
        summary=summary,
    )
