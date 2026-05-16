"""章节级证据上下文：规则选取 EvidenceItem + Citation marker（无 LLM）。"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field

from src.domain.paper.citation import CitationMap, CitationRef
from src.domain.paper.evidence import EvidenceItem, EvidenceLedger


class SectionEvidenceContext(BaseModel):
    section_title: str
    section_goal: str | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    citation_refs: list[CitationRef] = Field(default_factory=list)
    context_text: str = ""


def _norm(s: str) -> str:
    return (s or "").lower()


def _priority_args_for_heading(title: str, goal: str | None) -> list[str]:
    blob = _norm(title) + " " + _norm(goal or "")
    if any(k in blob for k in ("method", "方法", "技术", "approach", "architecture")):
        return ["method", "experiment", "result", "problem", "contribution", "dataset", "metric", "limitation"]
    if any(
        k in blob
        for k in (
            "dataset",
            "数据集",
            "benchmark",
            "evaluation",
            "指标",
            "实验",
            "experiment",
        )
    ):
        return ["dataset", "metric", "experiment", "method", "result", "problem", "contribution", "limitation"]
    if any(k in blob for k in ("result", "性能", "效果", "accuracy", "表现")):
        return ["result", "metric", "experiment", "method", "dataset", "problem", "contribution", "limitation"]
    if any(k in blob for k in ("limitation", "局限", "challenge", "未来", "讨论")):
        return ["limitation", "problem", "method", "result", "contribution", "dataset", "metric", "experiment"]
    return ["problem", "method", "result", "contribution", "dataset", "metric", "experiment", "limitation"]


def _items_for_args(ledger: EvidenceLedger, args: Iterable[str], *, max_items: int) -> list[EvidenceItem]:
    picked: list[EvidenceItem] = []
    used: set[str] = set()
    for arg in args:
        for it in ledger.items:
            if len(picked) >= max_items:
                return picked
            if it.evidence_id in used:
                continue
            if (it.supports_argument or "").strip().lower() == arg.strip().lower():
                picked.append(it)
                used.add(it.evidence_id)
    for it in ledger.items:
        if len(picked) >= max_items:
            break
        if it.evidence_id not in used:
            picked.append(it)
            used.add(it.evidence_id)
    return picked


def _format_item(it: EvidenceItem, cmap: CitationMap) -> str:
    mk = cmap.get_marker(it.evidence_id)
    return (
        f"{mk} Evidence ID: {it.evidence_id}\n"
        f"Paper: {it.title}\n"
        f"Claim: {it.claim}\n"
        f"Supports: {it.supports_argument or 'unknown'}\n"
        f"Confidence: {it.confidence:.2f}\n"
    )


def build_section_evidence_context(
    section_title: str,
    section_goal: str | None,
    ledger: EvidenceLedger,
    citation_map: CitationMap,
    *,
    max_items: int = 16,
) -> SectionEvidenceContext:
    order = _priority_args_for_heading(section_title, section_goal)
    items = _items_for_args(ledger, order, max_items=max_items)
    refs = [citation_map.get_ref_by_evidence_id(it.evidence_id) for it in items]
    refs = [r for r in refs if r is not None]
    lines = [f"Evidence Context for section: {section_title.strip() or '(untitled)'}\n"]
    for it in items:
        lines.append(_format_item(it, citation_map))
    return SectionEvidenceContext(
        section_title=section_title,
        section_goal=section_goal,
        evidence_items=items,
        citation_refs=refs,
        context_text="\n".join(lines).strip(),
    )


def build_global_evidence_context(
    ledger: EvidenceLedger,
    citation_map: CitationMap,
    *,
    max_items: int = 24,
    max_chars: int = 12000,
) -> SectionEvidenceContext:
    """无小节结构时的全局证据块。"""
    items = _items_for_args(
        ledger,
        ["problem", "method", "result", "contribution", "dataset", "metric", "experiment", "limitation"],
        max_items=max_items,
    )
    refs = [r for it in items if (r := citation_map.get_ref_by_evidence_id(it.evidence_id)) is not None]
    lines = ["Evidence Context (global)\n"]
    for it in items:
        lines.append(_format_item(it, citation_map))
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n...(truncated)"
    return SectionEvidenceContext(
        section_title="(global)",
        section_goal=None,
        evidence_items=items,
        citation_refs=refs,
        context_text=text,
    )


def build_citation_writing_instruction(citation_map: CitationMap) -> str:
    """注入章节任务前的统一写作约束（不含具体 claim）。"""
    legend = ", ".join(f"{r.inline_marker}={r.title[:40]}" for r in citation_map.refs[:20])
    return (
        "【证据约束写作 / Phase 3】\n"
        "1) 请优先使用下方 Evidence Context 中的论断；不要编造未出现的引用编号。\n"
        "2) 每个关键事实性论断后尽量附上已有引用标记（如 [C1]），且标记必须来自下列映射之一。\n"
        "3) 不要删除或改写正文中的 [C1] 形式标记。\n"
        "4) 若证据不足以支持更强结论，请明确写出：「现有证据不足以支持进一步结论」。\n"
        f"5) 可用引用与论文标题摘要：{legend or '（无）'}\n"
    )
