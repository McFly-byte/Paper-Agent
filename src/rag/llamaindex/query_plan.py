"""多查询规划：在服务端把检索需求扩展为互补子查询（可选 LLM，一次调用）。"""

from __future__ import annotations

import json
import re
from typing import Any

from src.core.llm_infra.invoke import chat_completion_text_routed
from src.rag.llamaindex import config as li_cfg
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _dedupe_preserve(xs: list[str], *, max_n: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        s = (x or "").strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def _parse_json_list(raw: str) -> list[str]:
    s = (raw or "").strip()
    if not s:
        return []
    # 尝试直接 JSON
    for candidate in (s, s[s.find("[") : s.rfind("]") + 1] if "[" in s else s):
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            continue
    # 宽松：逐行引号包裹
    out: list[str] = []
    for m in re.finditer(r'"([^"]{3,500})"', s):
        out.append(m.group(1).strip())
    return out


async def plan_retrieval_queries(
    raw_queries: list[str],
    *,
    section_hint: str | None,
) -> tuple[list[str], dict[str, Any]]:
    """
    返回 (planned_queries, meta)。
    meta 含 used_llm / source 等供日志与 LangSmith。
    """
    base = _dedupe_preserve([str(x) for x in (raw_queries or [])], max_n=12)
    meta: dict[str, Any] = {"source": "raw", "used_llm": False, "section_hint_len": len(section_hint or "")}

    if not li_cfg.llamaindex_multi_query_enabled() or not base:
        meta["planned_count"] = len(base)
        return base, meta

    max_q = li_cfg.llamaindex_multi_query_max()
    seed = " ".join(base[:3])[:1200]
    hint = (section_hint or "").strip()[:800]

    prompt = (
        "你是学术文献检索规划助手。根据「写作检索需求」生成 3~5 条**互补**的短检索句，"
        "分别从：主题/任务、方法/技术、实验/结果与指标、局限与挑战 等视角覆盖；"
        "避免仅同义改写。输出只能是 JSON 数组，例如 "
        '["...","..."]，不要其它文字。\n\n'
        f"【当前小节上下文（可空）】\n{hint or '（无）'}\n\n"
        f"【检索需求】\n{seed}\n"
    )

    planned: list[str] = []
    try:
        text = await chat_completion_text_routed(
            "rag-generation-model",
            prompt,
            node_name="rag_query_plan",
            temperature=0.15,
            source="rag_query_plan",
        )
        planned = _parse_json_list(text or "")
        meta["used_llm"] = True
        meta["source"] = "llm_plan"
    except Exception as exc:  # noqa: BLE001
        logger.warning("多查询 LLM 规划失败，回退为原始查询列表: %s", exc)
        meta["source"] = "raw_fallback"

    merged = _dedupe_preserve(base + planned, max_n=max(1, max_q))
    if len(merged) < len(base):
        merged = base

    meta["planned_count"] = len(merged)
    meta["seed_queries"] = base[:5]
    return merged, meta
