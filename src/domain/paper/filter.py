"""检索结果标准化、去重与规则型相关性过滤（无 LLM / 无向量依赖）。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field

from src.agents.planner.models import ResearchPlan
from src.domain.paper.models import PaperCandidate
from src.runtime.state import BackgroundContext


_ARXIV_ABS_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(?P<id>[\w.-]+)(?:\.pdf)?", re.IGNORECASE
)


class PaperScore(BaseModel):
    paper_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    keyword_score: float = Field(ge=0.0, le=1.0)
    inclusion_score: float = Field(ge=0.0, le=1.0)
    exclusion_penalty: float = Field(ge=0.0, le=1.0)
    metadata_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class RejectedPaper(BaseModel):
    paper_id: str
    title: str
    reason: str
    score: float


class PaperFilterResult(BaseModel):
    input_count: int
    normalized_count: int
    deduplicated_count: int
    output_count: int
    min_score: float
    selected: list[PaperCandidate]
    rejected: list[RejectedPaper]
    scores: list[PaperScore]
    warnings: list[str] = Field(default_factory=list)


class FilterStats(BaseModel):
    """去重阶段统计。"""

    input_count: int = 0
    output_count: int = 0
    duplicates_removed: int = 0


class FilterContext(BaseModel):
    """评分用上下文（由计划 / 背景 / 用户请求组装）。"""

    query_text: str = ""
    query_tokens: set[str] = Field(default_factory=set)
    inclusion_phrases: list[str] = Field(default_factory=list)
    exclusion_phrases: list[str] = Field(default_factory=list)
    category_hints: set[str] = Field(default_factory=set)


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def _arxiv_id_from_string(s: str) -> str | None:
    if not s:
        return None
    m = _ARXIV_ABS_RE.search(s)
    if m:
        return m.group("id").strip().lower()
    s2 = s.strip().lower()
    if s2.startswith("arxiv:"):
        return s2.split(":", 1)[1].strip()
    if re.fullmatch(r"[\d.]{7,}(v\d+)?", s2):
        return s2
    return None


def _extract_pdf_url(raw: dict[str, Any]) -> str | None:
    for key in ("pdf_url", "pdf"):
        u = _safe_str(raw.get(key))
        if u:
            return u
    links = raw.get("links")
    if isinstance(links, list):
        for item in links:
            if isinstance(item, str) and ".pdf" in item.lower():
                return item.strip()
            if isinstance(item, dict):
                href = _safe_str(item.get("href") or item.get("url"))
                title = _safe_str(item.get("title") or item.get("rel") or "")
                if href and (".pdf" in href.lower() or "pdf" in title.lower()):
                    return href
    return None


def _as_author_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [a.strip() for a in re.split(r"[,;，、]", v) if a.strip()]
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if isinstance(x, dict):
                name = _safe_str(x.get("name") or x.get("author"))
                if name:
                    out.append(name)
            elif x is not None:
                s = str(x).strip()
                if s:
                    out.append(s)
        return out
    return []


def _as_category_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [c.strip() for c in re.split(r"[\s,;]+", v) if c.strip()]
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []


def _stable_paper_id_fallback(title: str, url: str) -> str:
    base = (title or "") + "|" + (url or "")
    h = hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"hash:{h}"


def normalize_paper_candidate(raw: dict[str, Any]) -> PaperCandidate:
    """将单条检索 dict 规范为 PaperCandidate；字段缺失不抛错。"""
    if not isinstance(raw, dict):
        raw = {}

    title = _safe_str(raw.get("title")) or "Untitled Paper"

    pid = _safe_str(raw.get("paper_id"))
    if not pid:
        pid = _safe_str(raw.get("arxiv_id"))
    if not pid:
        pid = _safe_str(raw.get("id"))
    if not pid:
        pid = _safe_str(raw.get("entry_id"))
    if not pid:
        pid = _arxiv_id_from_string(_safe_str(raw.get("url"))) or ""
    if not pid:
        pid = _arxiv_id_from_string(_safe_str(raw.get("entry_id"))) or ""
    if not pid:
        pid = _stable_paper_id_fallback(title, _safe_str(raw.get("url")))

    abstract = _safe_str(raw.get("abstract")) or _safe_str(raw.get("summary")) or _safe_str(
        raw.get("description")
    ) or None
    if abstract == "":
        abstract = None

    authors = _as_author_list(raw.get("authors"))
    if not authors:
        authors = _as_author_list(raw.get("author"))
    if not authors:
        authors = _as_author_list(raw.get("creators"))

    published = (
        _safe_str(raw.get("published"))
        or _safe_str(raw.get("published_date"))
        or _safe_str(raw.get("updated"))
        or _safe_str(raw.get("year"))
        or None
    )
    if published == "":
        published = None

    url = _safe_str(raw.get("url")) or _safe_str(raw.get("entry_id")) or _safe_str(
        raw.get("link")
    ) or None
    if url == "":
        url = None

    pdf_url = _extract_pdf_url(raw)

    categories = _as_category_list(raw.get("categories"))
    if not categories:
        categories = _as_category_list(raw.get("tags"))
    pc = _safe_str(raw.get("primary_category"))
    if pc:
        categories = [pc] + [c for c in categories if c != pc]

    source = _safe_str(raw.get("source")) or "arxiv"
    if not source:
        source = "arxiv"

    return PaperCandidate(
        paper_id=pid,
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        source=source,
        url=url,
        pdf_url=pdf_url,
        categories=categories,
        raw=dict(raw),
    )


def normalize_paper_candidates(raw_results: list[dict[str, Any]] | None) -> list[PaperCandidate]:
    if not raw_results:
        return []
    out: list[PaperCandidate] = []
    for item in raw_results:
        if isinstance(item, dict):
            out.append(normalize_paper_candidate(item))
    return out


def normalized_title_key(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _candidate_dedupe_keys(c: PaperCandidate) -> set[str]:
    keys: set[str] = set()
    keys.add(f"id:{c.paper_id.strip().lower()}")
    aid = _arxiv_id_from_string(c.paper_id) or _arxiv_id_from_string(c.url or "") or _arxiv_id_from_string(
        c.pdf_url or ""
    )
    if aid:
        keys.add(f"arxiv:{aid}")
    if c.pdf_url:
        keys.add(f"pdf:{c.pdf_url.strip().lower()}")
    if c.url:
        keys.add(f"url:{c.url.strip().lower()}")
    nt = normalized_title_key(c.title)
    if nt:
        keys.add(f"title:{nt}")
    return keys


def _abstract_len(c: PaperCandidate) -> int:
    return len((c.abstract or "").strip())


def _dedupe_rank(c: PaperCandidate) -> tuple[int, int, int]:
    """越大越优先保留：摘要长度、是否有 pdf、是否有 published。"""
    return (_abstract_len(c), 1 if (c.pdf_url or "").strip() else 0, 1 if (c.published or "").strip() else 0)


def deduplicate_candidates(candidates: list[PaperCandidate]) -> tuple[list[PaperCandidate], FilterStats]:
    if not candidates:
        return [], FilterStats(input_count=0, output_count=0, duplicates_removed=0)

    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owner: dict[str, int] = {}
    for i, c in enumerate(candidates):
        for k in _candidate_dedupe_keys(c):
            if k in owner:
                union(i, owner[k])
            else:
                owner[k] = i

    buckets: dict[int, list[PaperCandidate]] = {}
    for i, c in enumerate(candidates):
        r = find(i)
        buckets.setdefault(r, []).append(c)

    out: list[PaperCandidate] = []
    removed = 0
    for _rep, bucket in buckets.items():
        best = sorted(bucket, key=_dedupe_rank, reverse=True)[0]
        out.append(best)
        removed += max(0, len(bucket) - 1)

    return out, FilterStats(
        input_count=n,
        output_count=len(out),
        duplicates_removed=removed,
    )


def _tokenize(text: str) -> set[str]:
    text = (text or "").lower()
    parts = re.split(r"[^\w\u4e00-\u9fff]+", text, flags=re.UNICODE)
    return {p for p in parts if len(p) >= 2}


def build_filter_context(
    plan: ResearchPlan | None,
    background_context: BackgroundContext | None,
    user_request: str,
    *,
    brief_topic: str | None = None,
) -> FilterContext:
    parts: list[str] = []
    if user_request:
        parts.append(user_request)
    if brief_topic:
        parts.append(brief_topic)

    query_tokens: set[str] = set()
    inclusion: list[str] = []
    exclusion: list[str] = []
    cats: set[str] = set()

    if plan:
        for st in plan.search_tasks or []:
            if st.query:
                parts.append(st.query)
            query_tokens |= _tokenize(st.query)
            for inc in st.inclusion_criteria or []:
                s = _safe_str(inc)
                if s:
                    inclusion.append(s.lower())
            for exc in st.exclusion_criteria or []:
                s = _safe_str(exc)
                if s:
                    exclusion.append(s.lower())

    if background_context:
        parts.append(background_context.topic or "")
        for k in background_context.expanded_keywords or []:
            s = _safe_str(k)
            if s:
                parts.append(s)
                query_tokens |= _tokenize(s)
        for r in background_context.related_terms or []:
            s = _safe_str(r)
            if s:
                parts.append(s)
                query_tokens |= _tokenize(s)
        for q in background_context.suggested_search_queries or []:
            s = _safe_str(q)
            if s:
                parts.append(s)
                query_tokens |= _tokenize(s)

    query_text = " ".join(parts)
    query_tokens |= _tokenize(user_request)
    query_tokens |= _tokenize(brief_topic or "")

    for inc in inclusion:
        query_tokens |= _tokenize(inc)

    return FilterContext(
        query_text=query_text,
        query_tokens={t for t in query_tokens if len(t) >= 2},
        inclusion_phrases=inclusion,
        exclusion_phrases=exclusion,
        category_hints=cats,
    )


def _overlap_score(tokens: set[str], text: str, *, weight: float) -> tuple[float, list[str]]:
    if not tokens or not text:
        return 0.0, []
    tl = text.lower()
    hits = [t for t in tokens if t in tl]
    if not hits:
        return 0.0, []
    # 归一：命中比例 * weight，上限 1
    ratio = min(1.0, len(hits) / max(1, len(tokens)))
    return min(1.0, ratio * weight), hits[:8]


def score_candidate(candidate: PaperCandidate, ctx: FilterContext) -> PaperScore:
    reasons: list[str] = []
    title = candidate.title or ""
    abstract = (candidate.abstract or "").lower()
    blob_title = (title or "").lower()
    blob = f"{blob_title} {abstract}".strip()

    kw_title, ht = _overlap_score(ctx.query_tokens, blob_title, weight=1.2)
    kw_abs, ha = _overlap_score(ctx.query_tokens, abstract, weight=0.7)
    keyword_score = min(1.0, (kw_title * 0.55 + kw_abs * 0.45))
    if ht:
        reasons.append(f"keywords_in_title:{','.join(ht[:5])}")
    if ha:
        reasons.append(f"keywords_in_abstract:{','.join(ha[:5])}")

    inc_hits = 0
    for phrase in ctx.inclusion_phrases:
        if phrase and phrase in blob:
            inc_hits += 1
            reasons.append(f"inclusion_hit:{phrase[:40]}")
    inclusion_score = min(1.0, inc_hits * 0.25)

    exc_hits = 0
    for phrase in ctx.exclusion_phrases:
        if phrase and phrase in blob:
            exc_hits += 1
            reasons.append(f"exclusion_hit:{phrase[:40]}")
    exclusion_penalty = min(1.0, exc_hits * 0.2)

    meta = 0.0
    if (candidate.abstract or "").strip():
        meta += 0.25
        reasons.append("has_abstract")
    if (candidate.pdf_url or "").strip():
        meta += 0.25
        reasons.append("has_pdf")
    if (candidate.published or "").strip():
        meta += 0.2
        reasons.append("has_published")
    if candidate.categories:
        meta += 0.15
        reasons.append("has_categories")
    metadata_score = min(1.0, meta)

    if title.strip() == "Untitled Paper":
        reasons.append("untitled_penalty")

    relevance = (
        0.52 * keyword_score
        + 0.28 * inclusion_score
        + 0.15 * metadata_score
        - 0.35 * exclusion_penalty
    )
    if title.strip() == "Untitled Paper":
        relevance -= 0.08
    relevance = max(0.0, min(1.0, relevance))

    return PaperScore(
        paper_id=candidate.paper_id,
        relevance_score=relevance,
        keyword_score=keyword_score,
        inclusion_score=inclusion_score,
        exclusion_penalty=exclusion_penalty,
        metadata_score=metadata_score,
        reasons=reasons[:20],
    )


def _resolve_top_k(
    plan: ResearchPlan | None,
    max_papers: int,
) -> int:
    k = max(1, int(max_papers))
    if plan and plan.search_tasks:
        tops = [int(t.top_k) for t in plan.search_tasks if getattr(t, "top_k", None)]
        if tops:
            k = min(k, min(tops))
    return max(1, k)


def filter_candidates(
    candidates: list[PaperCandidate],
    plan: ResearchPlan | None,
    background_context: BackgroundContext | None,
    max_papers: int,
    config: dict[str, Any],
    *,
    user_request: str = "",
    brief_topic: str | None = None,
    min_score: float = 0.05,
    fallback_keep_top_k: bool = True,
    max_reject_log: int = 20,
) -> PaperFilterResult:
    """规则过滤 + 排序；全低于 min_score 时可 fallback 保留 top。"""
    warnings: list[str] = []
    input_count = len(candidates)
    if not candidates:
        return PaperFilterResult(
            input_count=input_count,
            normalized_count=0,
            deduplicated_count=0,
            output_count=0,
            min_score=min_score,
            selected=[],
            rejected=[],
            scores=[],
            warnings=["empty_input"],
        )

    deduped, stats = deduplicate_candidates(candidates)
    if stats.duplicates_removed:
        warnings.append(f"deduplicated:{stats.duplicates_removed}")

    ctx = build_filter_context(plan, background_context, user_request, brief_topic=brief_topic)
    scores_list = [score_candidate(c, ctx) for c in deduped]

    ranked = sorted(
        zip(deduped, scores_list),
        key=lambda x: x[1].relevance_score,
        reverse=True,
    )

    top_k = _resolve_top_k(plan, max_papers)

    selected: list[PaperCandidate] = []
    rejected: list[RejectedPaper] = []

    above = [(c, s) for c, s in ranked if s.relevance_score >= min_score]
    if above:
        pool = above
    elif fallback_keep_top_k:
        warnings.append("all_below_min_score_fallback_top_k")
        pool = ranked[: min(top_k, len(ranked))]
    else:
        warnings.append("all_below_min_score_no_output")
        pool = []

    pool = pool[:top_k]

    selected_ids = {c.paper_id for c, _s in pool}
    for c, s in pool:
        selected.append(c)

    for c, s in ranked:
        if c.paper_id in selected_ids:
            continue
        if len(rejected) >= max_reject_log:
            break
        rejected.append(
            RejectedPaper(
                paper_id=c.paper_id,
                title=c.title,
                reason="below_threshold_or_not_in_top_k",
                score=s.relevance_score,
            )
        )

    return PaperFilterResult(
        input_count=input_count,
        normalized_count=input_count,
        deduplicated_count=len(deduped),
        output_count=len(selected),
        min_score=min_score,
        selected=selected,
        rejected=rejected,
        scores=[s for _, s in ranked],
        warnings=warnings,
    )
