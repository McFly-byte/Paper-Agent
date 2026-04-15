"""主流程五节点的确定性验收门禁（与 LLM-as-Judge 解耦）。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.core.config import config

GateNextAction = Literal["continue", "retry", "human_review", "fail"]


class NodeGateResult(BaseModel):
    """统一门禁输出结构，写入 PaperAgentState.boundary_checks[node_name]。"""

    passed: bool
    score: Optional[float] = Field(default=None, description="0~1 可选综合分，便于与 quality_eval 对齐")
    reasons: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    next_action: GateNextAction = "continue"


def _date_ok(s: Optional[str]) -> bool:
    if s is None or str(s).strip() == "":
        return True
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(s).strip()))


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def gate_search(ctx: Dict[str, Any]) -> NodeGateResult:
    """
    ctx 建议字段：querys, start_date, end_date, raw_result_count, dedup_count, top_titles
    """
    reasons: list[str] = []
    metrics: dict[str, Any] = {}
    querys = ctx.get("querys") or []
    if not isinstance(querys, list):
        querys = []
    metrics["query_count"] = len(querys)
    metrics["top_titles"] = ctx.get("top_titles") or []

    if not querys:
        reasons.append("检索子式 querys 为空，无法构成可执行 arXiv 查询")
    bad_cjk = [q for q in querys if isinstance(q, str) and _CJK_RE.search(q)]
    if bad_cjk:
        reasons.append(f"存在含 CJK 的子式（arXiv 不适用）: {bad_cjk[:2]!r}")

    if not _date_ok(ctx.get("start_date")) or not _date_ok(ctx.get("end_date")):
        reasons.append("start_date/end_date 格式非法，须为 YYYY-MM-DD 或留空")

    raw_n = int(ctx.get("raw_result_count") or 0)
    dedup_n = int(ctx.get("dedup_count") if ctx.get("dedup_count") is not None else raw_n)
    metrics["raw_result_count"] = raw_n
    metrics["dedup_count"] = dedup_n
    max_results = config.get_int("node_gates.search_max_results", 500)
    if raw_n <= 0:
        reasons.append("检索结果数为 0")
    if dedup_n > max_results:
        reasons.append(f"去重后结果数 {dedup_n} 超过合理上限 {max_results}，疑似查询过宽")

    passed = len(reasons) == 0
    score = 1.0 if passed else 0.0
    next_action: GateNextAction = "continue" if passed else "human_review"
    if raw_n <= 0 and not querys:
        next_action = "fail"
    return NodeGateResult(
        passed=passed,
        score=score,
        reasons=reasons,
        metrics=metrics,
        next_action=next_action,
    )


def _paper_core_fields_ok(p: Any) -> tuple[bool, float]:
    """返回 (是否非全空, 简单完整率 0~1)。"""
    if p is None:
        return False, 0.0
    fields = 0
    hit = 0
    core = getattr(p, "core_problem", None) or (isinstance(p, dict) and p.get("core_problem"))
    fields += 1
    if core and str(core).strip():
        hit += 1
    km = getattr(p, "key_methodology", None) or (isinstance(p, dict) and p.get("key_methodology"))
    fields += 1
    if km:
        if hasattr(km, "name") or hasattr(km, "principle"):
            if (getattr(km, "name", None) or "").strip() or (getattr(km, "principle", None) or "").strip():
                hit += 1
        elif isinstance(km, dict) and (str(km.get("name") or "").strip() or str(km.get("principle") or "").strip()):
            hit += 1
    mr = getattr(p, "main_results", None) or (isinstance(p, dict) and p.get("main_results"))
    fields += 1
    if mr and str(mr).strip():
        hit += 1
    contrib = getattr(p, "contributions", None) or (isinstance(p, dict) and p.get("contributions"))
    fields += 1
    if contrib:
        if isinstance(contrib, list) and any(str(c).strip() for c in contrib):
            hit += 1
        elif isinstance(contrib, str) and contrib.strip():
            hit += 1
    fill = hit / max(fields, 1)
    return hit > 0, fill


def gate_reading(
    *,
    input_paper_count: int,
    papers_parsed: List[Any],
    kb_write_count: int,
) -> NodeGateResult:
    reasons = []
    min_ratio = config.get_float("node_gates.reading_min_parse_ratio", 0.6)
    min_field = config.get_float("node_gates.reading_min_field_fill_ratio", 0.25)

    n_in = max(input_paper_count, 0)
    n_ok = len(papers_parsed)
    parse_ratio = (n_ok / n_in) if n_in else 0.0
    fills: list[float] = []
    non_empty_core = 0
    for p in papers_parsed:
        ok, fill = _paper_core_fields_ok(p)
        fills.append(fill)
        if ok:
            non_empty_core += 1
    avg_fill = sum(fills) / len(fills) if fills else 0.0

    metrics = {
        "input_paper_count": n_in,
        "parsed_paper_count": n_ok,
        "parse_ratio": round(parse_ratio, 4),
        "avg_core_field_fill": round(avg_fill, 4),
        "kb_write_count": kb_write_count,
    }

    if n_in <= 0:
        reasons.append("阅读节点输入论文数为 0")
    elif parse_ratio + 1e-9 < min_ratio:
        reasons.append(
            f"解析成功率 {parse_ratio:.2f} 低于阈值 {min_ratio:.2f}（成功 {n_ok}/{n_in}）"
        )
    if fills and avg_fill + 1e-9 < min_field:
        reasons.append(
            f"关键字段平均完整率 {avg_fill:.2f} 低于阈值 {min_field:.2f}"
        )
    if kb_write_count < n_ok:
        reasons.append(f"入库篇数 {kb_write_count} 少于成功解析篇数 {n_ok}")

    passed = len(reasons) == 0
    score = min(1.0, 0.5 * parse_ratio / max(min_ratio, 1e-6) + 0.5 * avg_fill / max(min_field, 1e-6))
    if not passed:
        score = min(score, 0.85)
    return NodeGateResult(
        passed=passed,
        score=round(float(score), 4),
        reasons=reasons,
        metrics=metrics,
        next_action="continue" if passed else "fail",
    )


_GLOBAL_MODULE_PATTERNS = (
    "技术趋势",
    "方法对比",
    "应用",
    "研究热点",
    "局限",
    "建议",
)


def gate_analyse(blob: Any) -> NodeGateResult:
    """blob: analyse_results JSON 字符串或 dict。"""
    reasons: list[str] = []
    raw = blob
    if isinstance(blob, str):
        try:
            raw = json.loads(blob) if blob.strip() else {}
        except json.JSONDecodeError:
            reasons.append("analyse_results 非合法 JSON，无法验收")
            return NodeGateResult(passed=False, score=0.0, reasons=reasons, metrics={}, next_action="fail")
    if not isinstance(raw, dict):
        reasons.append("分析结果不是对象结构")
        return NodeGateResult(passed=False, score=0.0, reasons=reasons, metrics={}, next_action="fail")

    if not raw.get("isSuccess", True):
        reasons.append(f"分析流程标记失败: {raw.get('global_analyse', '')[:200]}")

    summaries = raw.get("cluster_summaries") or []
    if not isinstance(summaries, list) or len(summaries) == 0:
        reasons.append("cluster_summaries 为空")
    else:
        for i, s in enumerate(summaries):
            if not isinstance(s, dict):
                reasons.append(f"cluster_summaries[{i}] 非对象")
                continue
            theme = (s.get("theme") or s.get("theme_description") or "").strip()
            kws = s.get("keywords") or []
            if not theme:
                reasons.append(f"簇 {i} 缺少 theme/theme_description")
            if not kws or not isinstance(kws, list):
                reasons.append(f"簇 {i} 缺少 keywords 列表")

    ga = (raw.get("global_analyse") or "").strip()
    cluster_count = len(summaries) if isinstance(summaries, list) else 0
    metrics = {
        "cluster_count": cluster_count,
        "deep_analysis_count": cluster_count,
        "global_analysis_length": len(ga),
    }

    if len(ga) < config.get_int("node_gates.analyse_global_min_chars", 400):
        reasons.append(f"global_analyse 长度过短（{len(ga)} 字符）")

    covered = sum(1 for p in _GLOBAL_MODULE_PATTERNS if p in ga)
    metrics["global_module_hits"] = covered
    if covered < config.get_int("node_gates.analyse_global_min_module_hits", 4):
        reasons.append(
            f"全局分析未覆盖足够模块关键词（命中 {covered}/{len(_GLOBAL_MODULE_PATTERNS)}）"
        )

    passed = len(reasons) == 0
    score = 1.0 if passed else max(0.0, 0.4 + 0.1 * covered)
    return NodeGateResult(
        passed=passed,
        score=round(score, 4),
        reasons=reasons,
        metrics=metrics,
        next_action="continue" if passed else "fail",
    )


def gate_writing(
    *,
    planned_sections: List[str],
    writted_sections: List[Any],
) -> NodeGateResult:
    blocking: list[str] = []
    planned = len(planned_sections or [])
    written_states = writted_sections or []
    non_empty = 0
    min_chars = config.get_int("node_gates.writing_min_section_chars", 200)
    lengths: list[int] = []
    verdicts: list[str] = []
    for sec in written_states:
        content = getattr(sec, "content", None) if not isinstance(sec, dict) else sec.get("content")
        c = (content or "").strip()
        lengths.append(len(c))
        if len(c) >= min_chars:
            non_empty += 1
        rv = getattr(sec, "review_verdict", None) if not isinstance(sec, dict) else sec.get("review_verdict")
        if rv:
            verdicts.append(str(rv).lower())

    match_ratio = (non_empty / planned) if planned else 0.0
    min_match = config.get_float("node_gates.writing_min_section_match_ratio", 0.75)

    metrics = {
        "planned_sections": planned,
        "written_sections_nonempty": non_empty,
        "section_match_ratio": round(match_ratio, 4),
        "section_lengths": lengths,
        "review_verdicts": verdicts,
    }
    if verdicts:
        passed_n = sum(1 for v in verdicts if v == "pass")
        metrics["review_pass_rate"] = round(passed_n / len(verdicts), 4)
    else:
        metrics["review_pass_rate"] = None

    if planned <= 0:
        blocking.append("写作主管未产出任何 section（sections 为空）")
    elif match_ratio + 1e-9 < min_match:
        blocking.append(
            f"达到最小字数的章节比例 {match_ratio:.2f} 低于阈值 {min_match:.2f}（{non_empty}/{planned}，每节≥{min_chars} 字）"
        )

    short_idx = [i for i, ln in enumerate(lengths) if 0 < ln < min_chars]
    if short_idx:
        blocking.append(f"下列章节长度不足 {min_chars} 字: {short_idx[:8]}")

    warnings: list[str] = []
    if verdicts:
        fail_n = sum(1 for v in verdicts if v == "fail")
        if fail_n > 0:
            blocking.append(f"存在 review verdict=fail 的章节（{fail_n} 次）")
        min_pass_rate = config.get_float("node_gates.writing_min_review_pass_rate", 0.5)
        pr = metrics["review_pass_rate"] or 0.0
        if pr + 1e-9 < min_pass_rate:
            blocking.append(
                f"审查通过率 {pr:.2f} 低于阈值 {min_pass_rate:.2f}"
            )
    else:
        warnings.append(
            "未捕获到结构化 review_verdict（审查结论未写入 SectionState；建议检查 review 模型 JSON 输出）"
        )

    passed = len(blocking) == 0
    score = match_ratio * (metrics["review_pass_rate"] or 0.65)
    reasons = blocking + warnings
    if warnings and passed:
        score = min(float(score), 0.88)
    return NodeGateResult(
        passed=passed,
        score=round(float(score), 4),
        reasons=reasons,
        metrics=metrics,
        next_action="continue" if passed else "fail",
    )


_REPORT_MD_NOISE = re.compile(r"[\s#*_>`\[\]\(\)\|'\"“”‘’:：,，.。;；\-—/\\|]+")


def _report_text_for_overlap(s: str) -> str:
    """弱化 Markdown 与空白噪声，便于与润色后的终稿做片段重叠判断。"""
    t = (s or "").lower()
    return _REPORT_MD_NOISE.sub("", t)


def _char_ngram_set(norm: str, n: int = 3) -> set[str]:
    if len(norm) < n:
        return set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def _trigram_recall_in_report(sec_raw: str, report_trigrams: set[str], *, max_sec_chars: int) -> float:
    """源节在报告中的字符 trigram 召回率（对改写/重排比子串匹配更稳）。"""
    norm = _report_text_for_overlap(sec_raw[:max_sec_chars])
    if len(norm) < 12:
        return 1.0
    ts = _char_ngram_set(norm, 3)
    if not ts:
        return 0.0
    return len(ts & report_trigrams) / len(ts)


def _identifier_tokens(s: str) -> list[str]:
    """英文/数字类标识（方法名、数据集、指标等），润色后仍易保留。"""
    out: list[str] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9._\-]{4,}", s):
        tok = m.group(0).lower()
        if len(tok) >= 5:
            out.append(tok)
    return out[:24]


def _section_covers_report(
    snip: str,
    joined_lower: str,
    report_trigrams: set[str],
    *,
    trigram_recall_min: float,
    max_section_chars: int,
) -> bool:
    """判定「组装报告」是否仍体现该源节：先严子串，再 trigram 召回，再标识符命中。"""
    s = (snip or "").strip()
    if len(s) < 20:
        return True
    s_low = s.lower()
    if s_low[:80] and s_low[:80] in joined_lower:
        return True
    first_line = (s.splitlines() or [""])[0].strip().lower()
    if len(first_line) >= 8 and first_line in joined_lower:
        return True
    recall = _trigram_recall_in_report(s, report_trigrams, max_sec_chars=max_section_chars)
    if recall >= trigram_recall_min:
        return True
    rep_fold = joined_lower
    idents = _identifier_tokens(s)
    if len(idents) >= 2 and sum(1 for t in idents if t in rep_fold) >= 2:
        return True
    if any(len(t) >= 10 and t in rep_fold for t in idents):
        return True
    return False


def gate_report(
    *,
    report_markdown: str,
    section_snippets: List[str],
) -> NodeGateResult:
    reasons = []
    text = (report_markdown or "").strip()
    min_len = config.get_int("node_gates.report_min_chars", 800)
    metrics: dict[str, Any] = {
        "final_length": len(text),
        "planned_source_sections": len(section_snippets or []),
    }
    if len(text) < min_len:
        reasons.append(f"报告长度过短（{len(text)} < {min_len}）")
    if "#" not in text:
        reasons.append("Markdown 缺少标题层级（未检测到 #）")

    joined = text.lower()
    max_sec = config.get_int("node_gates.report_section_overlap_max_chars", 1200)
    tri_min = config.get_float("node_gates.report_section_trigram_recall_min", 0.11)
    rep_norm = _report_text_for_overlap(text)
    report_trigrams = _char_ngram_set(rep_norm, 3)

    hits = 0
    recalls: list[float] = []
    for snip in section_snippets or []:
        s = str(snip or "").strip()
        if len(s) < 20:
            hits += 1
            continue
        ok = _section_covers_report(
            s,
            joined,
            report_trigrams,
            trigram_recall_min=tri_min,
            max_section_chars=max_sec,
        )
        if ok:
            hits += 1
        recalls.append(round(_trigram_recall_in_report(s, report_trigrams, max_sec_chars=max_sec), 4))
    n_sec = len([s for s in (section_snippets or []) if str(s or "").strip()])
    if recalls:
        metrics["section_trigram_recalls"] = recalls[:16]
    metrics["report_trigram_bucket_size"] = len(report_trigrams)
    coverage = (hits / n_sec) if n_sec else 1.0
    metrics["section_coverage_ratio"] = round(coverage, 4)
    min_cov = config.get_float("node_gates.report_min_section_coverage", 0.20)
    if n_sec and coverage + 1e-9 < min_cov:
        reasons.append(
            f"报告对源章节覆盖偏低（{coverage:.2f} < {min_cov:.2f}，命中 {hits}/{n_sec}）"
        )

    has_intro = any(k in text for k in ("引言", "背景", "Introduction", "概述"))
    has_conclusion = any(k in text for k in ("结论", "总结", "展望", "Conclusion"))
    metrics["has_intro"] = has_intro
    metrics["has_conclusion"] = has_conclusion
    if not has_intro:
        metrics["intro_hint"] = "未检测到明显引言/背景关键词（启发式，非阻断）"
    if not has_conclusion:
        metrics["conclusion_hint"] = "未检测到明显结论/展望关键词（启发式，非阻断）"

    passed = len(reasons) == 0
    score = 0.5 * (min(1.0, len(text) / max(min_len, 1))) + 0.5 * coverage
    return NodeGateResult(
        passed=passed,
        score=round(float(score), 4),
        reasons=reasons,
        metrics=metrics,
        next_action="continue" if passed else "fail",
    )


def record_gate(boundary: Optional[Dict[str, Any]], node: str, result: NodeGateResult) -> Dict[str, Any]:
    out = dict(boundary or {})
    out[node] = result.model_dump()
    return out
