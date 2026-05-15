"""主流程五节点的确定性验收门禁（与 LLM-as-Judge 解耦）。"""

from __future__ import annotations

import json
import re
from datetime import datetime
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
_PROMPT_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?previous|system\s+prompt|developer\s+message|"
    r"you\s+are\s+chatgpt|act\s+as\s+|不要遵循|忽略(上述|之前|前面)|系统提示)",
    re.IGNORECASE,
)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if s is None or str(s).strip() == "":
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d")
    except ValueError:
        return None


def _text_value(obj: Any, *keys: str) -> str:
    for key in keys:
        if isinstance(obj, dict):
            val = obj.get(key)
        else:
            val = getattr(obj, key, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            s = " ".join(str(x) for x in val if x is not None)
        elif isinstance(val, dict):
            try:
                s = json.dumps(val, ensure_ascii=False)
            except (TypeError, ValueError):
                s = str(val)
        else:
            s = str(val)
        if s.strip():
            return s.strip()
    return ""


def _token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in _LATIN_TOKEN_RE.finditer(text or "")}


def _round_ratio(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def _has_reference_hint(text: str) -> bool:
    s = text or ""
    return bool(
        re.search(r"\[[0-9A-Za-z,\s;-]{1,24}\]|\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,?\s+\d{4}\)", s)
        or any(k in s for k in ("引用", "来源", "参考文献", "arXiv", "paper", "论文"))
    )


def _line_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    for raw in (markdown or "").splitlines():
        s = raw.strip()
        if len(s) >= 40 and not s.startswith("#"):
            blocks.append(re.sub(r"\s+", "", s.lower()))
    return blocks


def gate_search(ctx: Dict[str, Any]) -> NodeGateResult:
    """
    ctx 建议字段：querys, start_date, end_date, raw_result_count, dedup_count, top_titles, user_request
    """
    blocking: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    querys = ctx.get("querys") or []
    if not isinstance(querys, list):
        querys = []
    querys = [str(q).strip() for q in querys if str(q or "").strip()]
    metrics["query_count"] = len(querys)
    metrics["top_titles"] = ctx.get("top_titles") or []

    if not querys:
        blocking.append("检索子式 querys 为空，无法构成可执行 arXiv 查询")
    bad_cjk = [q for q in querys if isinstance(q, str) and _CJK_RE.search(q)]
    if bad_cjk:
        blocking.append(f"存在含 CJK 的子式（arXiv 不适用）: {bad_cjk[:2]!r}")

    if not _date_ok(ctx.get("start_date")) or not _date_ok(ctx.get("end_date")):
        blocking.append("start_date/end_date 格式非法，须为 YYYY-MM-DD 或留空")
    start_dt = _parse_date(ctx.get("start_date"))
    end_dt = _parse_date(ctx.get("end_date"))
    if start_dt and end_dt and start_dt > end_dt:
        blocking.append("start_date 晚于 end_date，时间范围非法")

    raw_n = int(ctx.get("raw_result_count") or 0)
    dedup_n = int(ctx.get("dedup_count") if ctx.get("dedup_count") is not None else raw_n)
    metrics["raw_result_count"] = raw_n
    metrics["dedup_count"] = dedup_n
    dedup_ratio = _round_ratio(dedup_n, raw_n)
    metrics["dedup_ratio"] = dedup_ratio
    requested_max = int(ctx.get("requested_max_results") or 0)
    metrics["requested_max_results"] = requested_max or None
    max_results = config.get_int("node_gates.search_max_results", 500)
    min_results = config.get_int("node_gates.search_min_results", 3)
    if raw_n <= 0:
        blocking.append("检索结果数为 0")
    elif dedup_n < min_results:
        blocking.append(f"去重后结果数 {dedup_n} 低于最小阈值 {min_results}，检索召回不足")
    if dedup_n > max_results:
        blocking.append(f"去重后结果数 {dedup_n} 超过合理上限 {max_results}，疑似查询过宽")
    low_dedup = config.get_float("node_gates.search_min_dedup_ratio", 0.75)
    if raw_n >= 5 and dedup_ratio + 1e-9 < low_dedup:
        blocking.append(f"检索结果重复率偏高，去重保留率 {dedup_ratio:.2f} 低于阈值 {low_dedup:.2f}")

    q_tokens = [_token_set(q) for q in querys]
    avg_query_terms = (sum(len(t) for t in q_tokens) / len(q_tokens)) if q_tokens else 0.0
    metrics["avg_query_terms"] = round(avg_query_terms, 4)
    if querys and avg_query_terms < config.get_float("node_gates.search_min_avg_query_terms", 2.0):
        warnings.append("检索子式平均有效英文词过少，可能过宽")
    long_q = [q for q in querys if len(q) > config.get_int("node_gates.search_max_query_chars", 180)]
    if long_q:
        warnings.append("部分检索子式过长，可能过窄或包含非检索意图文本")

    if requested_max and raw_n >= requested_max and requested_max >= min_results:
        warnings.append("检索结果命中本次 max_results 上限，建议人工确认 query 是否过宽")

    user_tokens = _token_set(str(ctx.get("user_request") or ""))
    result_text = " ".join(str(x or "") for x in (ctx.get("top_titles") or []))
    if ctx.get("top_abstracts"):
        result_text += " " + " ".join(str(x or "")[:500] for x in (ctx.get("top_abstracts") or []))
    result_tokens = _token_set(result_text)
    overlap = _round_ratio(len(user_tokens & result_tokens), len(user_tokens)) if user_tokens else None
    metrics["topic_token_overlap"] = overlap
    min_overlap = config.get_float("node_gates.search_min_topic_overlap", 0.08)
    if overlap is not None and raw_n > 0 and overlap + 1e-9 < min_overlap:
        warnings.append(f"top 结果与用户主题英文 token 重叠偏低（{overlap:.2f} < {min_overlap:.2f}）")

    metrics["blocking_failure_count"] = len(blocking)
    metrics["warning_count"] = len(warnings)
    if warnings:
        metrics["warnings"] = warnings

    passed = len(blocking) == 0
    penalty = 0.12 * len(warnings)
    score = 1.0 - penalty if passed else min(0.45, max(0.0, 0.2 + 0.2 * dedup_ratio))
    next_action: GateNextAction = "continue" if passed else "human_review"
    if raw_n <= 0 and querys:
        next_action = "retry"
    if raw_n <= 0 and not querys:
        next_action = "fail"
    reasons = blocking + warnings
    return NodeGateResult(
        passed=passed,
        score=round(max(0.0, min(1.0, float(score))), 4),
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
    limits = getattr(p, "limitations", None) or (isinstance(p, dict) and p.get("limitations"))
    fields += 1
    if limits and str(limits).strip():
        hit += 1
    fill = hit / max(fields, 1)
    return hit > 0, fill


def _metadata_field_coverage(papers: list[Any], *field_groups: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(papers)
    for group in field_groups:
        key = "_or_".join(group)
        hit = 0
        for p in papers:
            if any(_text_value(p, f) for f in group):
                hit += 1
        out[key] = round(hit / n, 4) if n else 0.0
    return out


def gate_reading(
    *,
    input_paper_count: int,
    papers_parsed: List[Any],
    kb_write_count: int,
    paper_metadatas: Optional[List[Any]] = None,
    llamaindex_node_count: Optional[int] = None,
) -> NodeGateResult:
    blocking: list[str] = []
    warnings: list[str] = []
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
    paper_metadatas = paper_metadatas or []
    meta_cov = _metadata_field_coverage(
        paper_metadatas,
        ("paper_id", "id", "arxiv_id"),
        ("title",),
        ("summary", "abstract"),
        ("url", "pdf_url", "source"),
    )
    metrics["metadata_field_coverage"] = meta_cov
    short_abs_min = config.get_int("node_gates.reading_min_abstract_chars", 80)
    short_abs = 0
    injection_markers = 0
    for p in paper_metadatas:
        abstract = _text_value(p, "summary", "abstract")
        title = _text_value(p, "title")
        if abstract and len(abstract) < short_abs_min:
            short_abs += 1
        if _PROMPT_INJECTION_RE.search(abstract) or _PROMPT_INJECTION_RE.search(title):
            injection_markers += 1
    metrics["short_abstract_count"] = short_abs
    metrics["prompt_injection_marker_count"] = injection_markers
    metrics["llamaindex_node_count"] = llamaindex_node_count

    if paper_metadatas:
        min_meta = config.get_float("node_gates.reading_min_metadata_coverage", 0.8)
        low_meta = {k: v for k, v in meta_cov.items() if v + 1e-9 < min_meta}
        if low_meta:
            warnings.append(f"论文 metadata 覆盖不足: {low_meta}")
    if injection_markers > 0:
        warnings.append(
            f"检测到 {injection_markers} 条疑似 prompt-injection 标记，已作为外部数据降权观察"
        )
    if short_abs > 0:
        warnings.append(f"存在 {short_abs} 篇摘要过短，后续分析可信度需降权")

    if n_in <= 0:
        blocking.append("阅读节点输入论文数为 0")
    elif parse_ratio + 1e-9 < min_ratio:
        blocking.append(
            f"解析成功率 {parse_ratio:.2f} 低于阈值 {min_ratio:.2f}（成功 {n_ok}/{n_in}）"
        )
    if fills and avg_fill + 1e-9 < min_field:
        blocking.append(
            f"关键字段平均完整率 {avg_fill:.2f} 低于阈值 {min_field:.2f}"
        )
    if kb_write_count < n_ok:
        blocking.append(f"入库篇数 {kb_write_count} 少于成功解析篇数 {n_ok}")
    if llamaindex_node_count is not None and n_ok > 0 and llamaindex_node_count <= 0:
        warnings.append("LlamaIndex 入库节点数为 0，写作阶段会降级依赖 legacy 临时库")

    metrics["blocking_failure_count"] = len(blocking)
    metrics["warning_count"] = len(warnings)
    if warnings:
        metrics["warnings"] = warnings

    passed = len(blocking) == 0
    score = min(1.0, 0.5 * parse_ratio / max(min_ratio, 1e-6) + 0.5 * avg_fill / max(min_field, 1e-6))
    if not passed:
        score = min(score, 0.85)
    elif warnings:
        score = min(score, 1.0 - min(0.25, 0.06 * len(warnings)))
    reasons = blocking + warnings
    next_action: GateNextAction = "continue" if passed else "retry"
    if n_in <= 0 or (n_in > 0 and n_ok == 0):
        next_action = "fail"
    return NodeGateResult(
        passed=passed,
        score=round(float(score), 4),
        reasons=reasons,
        metrics=metrics,
        next_action=next_action,
    )


_GLOBAL_MODULE_PATTERNS = (
    "技术趋势",
    "方法对比",
    "应用",
    "研究热点",
    "局限",
    "建议",
)


def gate_analyse(
    blob: Any,
    *,
    input_paper_count: Optional[int] = None,
    parsed_paper_titles: Optional[List[str]] = None,
) -> NodeGateResult:
    """blob: analyse_results JSON 字符串或 dict。"""
    blocking: list[str] = []
    warnings: list[str] = []
    raw = blob
    if isinstance(blob, str):
        try:
            raw = json.loads(blob) if blob.strip() else {}
        except json.JSONDecodeError:
            blocking.append("analyse_results 非合法 JSON，无法验收")
            return NodeGateResult(
                passed=False,
                score=0.0,
                reasons=blocking,
                metrics={"blocking_failure_count": 1, "warning_count": 0},
                next_action="fail",
            )
    if not isinstance(raw, dict):
        blocking.append("分析结果不是对象结构")
        return NodeGateResult(
            passed=False,
            score=0.0,
            reasons=blocking,
            metrics={"blocking_failure_count": 1, "warning_count": 0},
            next_action="fail",
        )

    if not raw.get("isSuccess", True):
        blocking.append(f"分析流程标记失败: {raw.get('global_analyse', '')[:200]}")

    summaries = raw.get("cluster_summaries") or []
    cluster_sizes: list[int] = []
    representative_count = 0
    if not isinstance(summaries, list) or len(summaries) == 0:
        blocking.append("cluster_summaries 为空")
    else:
        for i, s in enumerate(summaries):
            if not isinstance(s, dict):
                blocking.append(f"cluster_summaries[{i}] 非对象")
                continue
            theme = (s.get("theme") or s.get("theme_description") or "").strip()
            kws = s.get("keywords") or []
            if not theme:
                blocking.append(f"簇 {i} 缺少 theme/theme_description")
            if not kws or not isinstance(kws, list):
                blocking.append(f"簇 {i} 缺少 keywords 列表")
            papers = s.get("papers") if isinstance(s.get("papers"), list) else []
            pc = s.get("paper_count")
            try:
                paper_count = int(pc) if pc is not None else len(papers)
            except (TypeError, ValueError):
                paper_count = len(papers)
            cluster_sizes.append(max(0, paper_count))
            reps = s.get("representative_papers") or s.get("representative_titles") or []
            if isinstance(reps, list):
                representative_count += len([x for x in reps if str(x or "").strip()])

    ga = (raw.get("global_analyse") or "").strip()
    cluster_count = len(summaries) if isinstance(summaries, list) else 0
    metrics = {
        "input_paper_count": input_paper_count,
        "cluster_count": cluster_count,
        "deep_analysis_count": cluster_count,
        "global_analysis_length": len(ga),
        "cluster_sizes": cluster_sizes,
        "representative_paper_count": representative_count,
    }

    if len(ga) < config.get_int("node_gates.analyse_global_min_chars", 400):
        blocking.append(f"global_analyse 长度过短（{len(ga)} 字符）")

    covered = sum(1 for p in _GLOBAL_MODULE_PATTERNS if p in ga)
    metrics["global_module_hits"] = covered
    if covered < config.get_int("node_gates.analyse_global_min_module_hits", 4):
        blocking.append(
            f"全局分析未覆盖足够模块关键词（命中 {covered}/{len(_GLOBAL_MODULE_PATTERNS)}）"
        )

    if input_paper_count is not None and input_paper_count > 0 and cluster_count > 0:
        cluster_ratio = cluster_count / input_paper_count
        metrics["cluster_to_paper_ratio"] = round(cluster_ratio, 4)
        if input_paper_count >= 8 and cluster_count == 1:
            warnings.append("论文数量较多但只形成 1 个簇，可能聚类过粗")
        max_ratio = config.get_float("node_gates.analyse_max_cluster_to_paper_ratio", 0.75)
        if input_paper_count >= 6 and cluster_ratio > max_ratio:
            warnings.append(f"cluster 数量相对论文数偏高（{cluster_ratio:.2f} > {max_ratio:.2f}），可能过度拆分")
    if cluster_sizes:
        nonzero = [x for x in cluster_sizes if x > 0]
        total_cluster_papers = sum(nonzero)
        metrics["cluster_paper_total"] = total_cluster_papers
        if input_paper_count and total_cluster_papers and abs(total_cluster_papers - input_paper_count) > max(2, input_paper_count * 0.3):
            warnings.append("cluster_summaries 的 paper_count 总量与阅读论文数偏差较大")
        if nonzero:
            balance = min(nonzero) / max(nonzero)
            metrics["cluster_size_balance"] = round(balance, 4)
            if len(nonzero) >= 3 and balance < config.get_float("node_gates.analyse_min_cluster_size_balance", 0.12):
                warnings.append("聚类规模极不均衡，可能存在长尾主题被吞并")
    if cluster_count > 0 and representative_count == 0:
        warnings.append("cluster_summaries 未提供代表论文，分析可追溯性偏弱")

    if parsed_paper_titles:
        title_tokens = set().union(*[_token_set(t) for t in parsed_paper_titles if t]) if parsed_paper_titles else set()
        reference_like = set(re.findall(r"\b[A-Z][A-Za-z0-9\-]{4,}\b", ga))
        if reference_like and title_tokens:
            matched = {t.lower() for t in reference_like if t.lower() in title_tokens}
            metrics["reference_token_match_ratio"] = _round_ratio(len(matched), len(reference_like))
        else:
            metrics["reference_token_match_ratio"] = None

    metrics["blocking_failure_count"] = len(blocking)
    metrics["warning_count"] = len(warnings)
    if warnings:
        metrics["warnings"] = warnings

    passed = len(blocking) == 0
    score = 1.0 if passed else max(0.0, 0.4 + 0.1 * covered)
    if warnings and passed:
        score = min(score, 1.0 - min(0.25, 0.06 * len(warnings)))
    reasons = blocking + warnings
    return NodeGateResult(
        passed=passed,
        score=round(score, 4),
        reasons=reasons,
        metrics=metrics,
        next_action="continue" if passed else "retry",
    )


def gate_writing(
    *,
    planned_sections: List[str],
    writted_sections: List[Any],
    rag_retrieval_logs: Optional[List[Dict[str, Any]]] = None,
) -> NodeGateResult:
    blocking: list[str] = []
    planned = len(planned_sections or [])
    written_states = writted_sections or []
    non_empty = 0
    min_chars = config.get_int("node_gates.writing_min_section_chars", 200)
    lengths: list[int] = []
    verdicts: list[str] = []
    evidence_hits = 0
    title_content_hits = 0
    for sec in written_states:
        content = getattr(sec, "content", None) if not isinstance(sec, dict) else sec.get("content")
        c = (content or "").strip()
        lengths.append(len(c))
        if len(c) >= min_chars:
            non_empty += 1
        if c and _has_reference_hint(c):
            evidence_hits += 1
        rv = getattr(sec, "review_verdict", None) if not isinstance(sec, dict) else sec.get("review_verdict")
        if rv:
            verdicts.append(str(rv).lower())

    for idx, planned_title in enumerate(planned_sections or []):
        if idx >= len(written_states):
            continue
        sec = written_states[idx]
        content = getattr(sec, "content", None) if not isinstance(sec, dict) else sec.get("content")
        title_tokens = _token_set(str(planned_title or "")) - {"section", "chapter", "analysis", "paper", "research"}
        if not title_tokens:
            title_content_hits += 1
            continue
        content_tokens = _token_set(str(content or "")[:1200])
        if title_tokens & content_tokens:
            title_content_hits += 1

    match_ratio = min(1.0, (non_empty / planned)) if planned else 0.0
    min_match = config.get_float("node_gates.writing_min_section_match_ratio", 0.75)
    evidence_ratio = min(1.0, (evidence_hits / len(written_states))) if written_states else 0.0
    title_match_ratio = min(1.0, (title_content_hits / planned)) if planned else 0.0
    rag_logs = rag_retrieval_logs or []
    rag_calls = len(rag_logs) if isinstance(rag_logs, list) else 0
    rag_positive = 0
    if isinstance(rag_logs, list):
        for entry in rag_logs:
            if not isinstance(entry, dict):
                continue
            try:
                final_n = int(entry.get("final_context_count") or entry.get("retrieved_count") or 0)
            except (TypeError, ValueError):
                final_n = 0
            if final_n > 0:
                rag_positive += 1
    rag_call_ratio = min(1.0, (rag_positive / planned)) if planned else 0.0

    metrics = {
        "planned_sections": planned,
        "written_sections_nonempty": non_empty,
        "section_match_ratio": round(match_ratio, 4),
        "section_lengths": lengths,
        "review_verdicts": verdicts,
        "section_evidence_hint_ratio": round(evidence_ratio, 4),
        "section_title_content_match_ratio": round(title_match_ratio, 4),
        "rag_call_count": rag_calls,
        "rag_positive_call_count": rag_positive,
        "rag_positive_call_ratio": round(rag_call_ratio, 4),
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

    min_evidence = config.get_float("node_gates.writing_min_evidence_hint_ratio", 0.5)
    if written_states and evidence_ratio + 1e-9 < min_evidence:
        warnings.append(f"章节证据/引用提示覆盖率偏低（{evidence_ratio:.2f} < {min_evidence:.2f}）")
    min_title_match = config.get_float("node_gates.writing_min_title_content_match_ratio", 0.5)
    if planned and title_match_ratio + 1e-9 < min_title_match:
        warnings.append(f"章节标题与正文 token 匹配率偏低（{title_match_ratio:.2f} < {min_title_match:.2f}）")
    min_rag = config.get_float("node_gates.writing_min_rag_positive_call_ratio", 0.5)
    if planned and rag_call_ratio + 1e-9 < min_rag:
        warnings.append(f"RAG 正向检索覆盖率偏低（{rag_call_ratio:.2f} < {min_rag:.2f}）")

    passed = len(blocking) == 0
    metrics["blocking_failure_count"] = len(blocking)
    metrics["warning_count"] = len(warnings)
    if warnings:
        metrics["warnings"] = warnings
    evidence_component = max(0.35, evidence_ratio) if written_states else 0.35
    rag_component = max(0.45, rag_call_ratio) if planned else 0.45
    score = match_ratio * (metrics["review_pass_rate"] or 0.65) * (0.75 + 0.15 * evidence_component + 0.10 * rag_component)
    reasons = blocking + warnings
    if warnings and passed:
        score = min(float(score), 0.88)
    return NodeGateResult(
        passed=passed,
        score=round(max(0.0, min(1.0, float(score))), 4),
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
    blocking: list[str] = []
    warnings: list[str] = []
    text = (report_markdown or "").strip()
    min_len = config.get_int("node_gates.report_min_chars", 800)
    metrics: dict[str, Any] = {
        "final_length": len(text),
        "planned_source_sections": len(section_snippets or []),
    }
    if len(text) < min_len:
        blocking.append(f"报告长度过短（{len(text)} < {min_len}）")
    if "#" not in text:
        blocking.append("Markdown 缺少标题层级（未检测到 #）")
    heading_count = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S+", text))
    metrics["heading_count"] = heading_count
    if heading_count < config.get_int("node_gates.report_min_heading_count", 2):
        warnings.append("Markdown 标题层级较少，报告结构可能偏弱")

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
        blocking.append(
            f"报告对源章节覆盖偏低（{coverage:.2f} < {min_cov:.2f}，命中 {hits}/{n_sec}）"
        )

    has_intro = any(k in text for k in ("引言", "背景", "Introduction", "概述"))
    has_conclusion = any(k in text for k in ("结论", "总结", "展望", "Conclusion"))
    has_body = any(k in text for k in ("方法", "分析", "趋势", "对比", "应用", "主体", "技术"))
    has_sources = _has_reference_hint(text)
    metrics["has_intro"] = has_intro
    metrics["has_conclusion"] = has_conclusion
    metrics["has_body_analysis"] = has_body
    metrics["has_source_or_reference_hint"] = has_sources
    if not has_intro:
        metrics["intro_hint"] = "未检测到明显引言/背景关键词（启发式，非阻断）"
    if not has_conclusion:
        metrics["conclusion_hint"] = "未检测到明显结论/展望关键词（启发式，非阻断）"
    if not has_body:
        warnings.append("未检测到明显主体分析模块关键词")
    if not has_sources:
        warnings.append("未检测到明显引用/来源说明")

    placeholder_hits = len(re.findall(r"TODO|TBD|待补充|占位|placeholder", text, flags=re.IGNORECASE))
    metrics["placeholder_count"] = placeholder_hits
    if placeholder_hits:
        blocking.append(f"报告包含 {placeholder_hits} 个 TODO/placeholder 标记")

    empty_heading_count = 0
    heading_matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s+.+$", text))
    for i, match in enumerate(heading_matches):
        heading_line = match.group(0).lstrip()
        if heading_line.startswith("# ") and i == 0:
            continue
        start = match.end()
        end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
        body = text[start:end].strip()
        if len(body) < config.get_int("node_gates.report_min_section_body_chars", 40):
            empty_heading_count += 1
    metrics["empty_section_count"] = empty_heading_count
    if empty_heading_count:
        warnings.append(f"存在 {empty_heading_count} 个疑似空章节")

    blocks = _line_blocks(text)
    duplicate_blocks = len(blocks) - len(set(blocks))
    dup_ratio = _round_ratio(duplicate_blocks, len(blocks))
    metrics["duplicate_paragraph_ratio"] = dup_ratio
    if len(blocks) >= 6 and dup_ratio > config.get_float("node_gates.report_max_duplicate_paragraph_ratio", 0.25):
        blocking.append(f"重复段落比例过高（{dup_ratio:.2f}）")

    metrics["blocking_failure_count"] = len(blocking)
    metrics["warning_count"] = len(warnings)
    if warnings:
        metrics["warnings"] = warnings

    passed = len(blocking) == 0
    structure_score = sum([has_intro, has_body, has_conclusion, has_sources]) / 4
    score = 0.4 * (min(1.0, len(text) / max(min_len, 1))) + 0.4 * coverage + 0.2 * structure_score
    if warnings and passed:
        score = min(float(score), 1.0 - min(0.25, 0.05 * len(warnings)))
    reasons = blocking + warnings
    return NodeGateResult(
        passed=passed,
        score=round(max(0.0, min(1.0, float(score))), 4),
        reasons=reasons,
        metrics=metrics,
        next_action="continue" if passed else "fail",
    )


def record_gate(boundary: Optional[Dict[str, Any]], node: str, result: NodeGateResult) -> Dict[str, Any]:
    out = dict(boundary or {})
    out[node] = result.model_dump()
    return out
