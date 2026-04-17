"""根据当前写作小节文案推断 section_type 偏好：关键词规则 或 轻量 LLM 分类。"""

from __future__ import annotations

import json
from typing import Iterable

from src.core.llm_infra.invoke import chat_completion_text_routed
from src.rag.llamaindex import config as li_cfg
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

# 与 ingestion 中 section_type 取值对齐
_ALL_SECTIONS = frozenset(
    {
        "abstract",
        "core_problem",
        "key_methodology",
        "datasets_used",
        "evaluation_metrics",
        "main_results",
        "limitations",
        "contributions",
    }
)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def infer_section_preferences(section_hint: str) -> tuple[list[str], str]:
    """
    返回 (preferred_section_types, coarse_intent_label)。
    preferred 可能为空：表示不额外偏向某一类 chunk。
    """
    t = _norm(section_hint)
    if not t:
        return [], "unknown"

    preferred: list[str] = []

    def add_many(xs: Iterable[str]) -> None:
        for x in xs:
            if x in _ALL_SECTIONS and x not in preferred:
                preferred.append(x)

    # 方法 / 架构 / 算法
    if any(
        k in t
        for k in (
            "方法",
            "技术路线",
            "算法",
            "模型结构",
            "架构",
            "实现",
            "method",
            "approach",
            "methodology",
            "architecture",
            "algorithm",
        )
    ):
        add_many(("key_methodology", "contributions", "core_problem"))
        return preferred, "methodology"

    # 实验 / 结果 / 指标 / 数据
    if any(
        k in t
        for k in (
            "实验",
            "结果",
            "指标",
            "性能",
            "数据集",
            "评测",
            "准确率",
            "baseline",
            "sota",
            "experiment",
            "result",
            "metric",
            "dataset",
            "evaluation",
        )
    ):
        add_many(("main_results", "evaluation_metrics", "datasets_used", "abstract"))
        return preferred, "results"

    # 局限 / 挑战 / 未来工作
    if any(
        k in t
        for k in (
            "局限",
            "不足",
            "挑战",
            "风险",
            "问题",
            "未来",
            "展望",
            "改进",
            "limitation",
            "challenge",
            "future work",
            "drawback",
        )
    ):
        add_many(("limitations", "core_problem", "main_results"))
        return preferred, "limitations"

    # 背景 / 综述 / 引言类
    if any(
        k in t
        for k in (
            "背景",
            "综述",
            "引言",
            "相关工作",
            "概述",
            "introduction",
            "background",
            "survey",
            "overview",
            "related work",
        )
    ):
        add_many(("abstract", "core_problem", "contributions"))
        return preferred, "survey"

    # 贡献 / 创新点
    if any(k in t for k in ("贡献", "创新", "novelty", "contribution")):
        add_many(("contributions", "key_methodology", "abstract"))
        return preferred, "contributions"

    return [], "unknown"


def _parse_section_intent_json(raw: str) -> tuple[list[str], str]:
    s = (raw or "").strip()
    if not s:
        return [], "unknown"
    for candidate in (s, s[s.find("{") : s.rfind("}") + 1] if "{" in s else s):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        label = str(data.get("intent_label") or "unknown").strip()
        st = data.get("section_types")
        out: list[str] = []
        if isinstance(st, list):
            for x in st:
                xs = str(x).strip()
                if xs in _ALL_SECTIONS and xs not in out:
                    out.append(xs)
        return out[:6], label
    return [], "unknown"


async def infer_section_preferences_llm(section_hint: str) -> tuple[list[str], str]:
    """轻量 Chat 分类：输出 JSON，映射到 section_type 与 intent_label。"""
    hint = (section_hint or "").strip()[:2000]
    if not hint:
        return [], "empty"

    prompt = (
        "你是检索路由分类器。根据「写作小节任务描述」，判断应从论文向量片段中优先检索哪些 section_type。\n"
        "可选 section_type（多选 1～4 个，只使用下列英文标识）："
        "abstract, core_problem, key_methodology, datasets_used, evaluation_metrics, "
        "main_results, limitations, contributions。\n"
        "intent_label 单选其一：methodology, results, limitations, survey, contributions, unknown。\n"
        "只输出一行 JSON，不要其它文字：\n"
        '{"intent_label":"...","section_types":["..."]}\n\n'
        f"【小节任务】\n{hint}\n"
    )
    client_type = li_cfg.llamaindex_section_intent_llm_client_type()
    text = await chat_completion_text_routed(
        client_type,
        prompt,
        node_name="rag_section_intent",
        temperature=0.0,
        source="rag_section_intent",
    )
    preferred, label = _parse_section_intent_json(text or "")
    if not preferred and label == "unknown":
        logger.debug("LLM section 分类未解析出 section_types，原始=%s", (text or "")[:200])
    return preferred, label


async def resolve_section_preferences(section_hint: str) -> tuple[list[str], str]:
    """按配置选择 keywords 或 llm；llm 失败时回退关键词。"""
    if li_cfg.llamaindex_section_classifier() == "keywords":
        return infer_section_preferences(section_hint)
    try:
        return await infer_section_preferences_llm(section_hint)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM section 分类失败，回退关键词规则: %s", exc)
        return infer_section_preferences(section_hint)
