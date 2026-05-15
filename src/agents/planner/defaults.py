"""ResearchPlan 默认回退（无 LLM 客户端依赖，供 planner 与单测复用）。"""

from __future__ import annotations

import re

from src.agents.planner.models import (
    AnalysisTask,
    ReadingTask,
    ResearchPlan,
    SearchTask,
    WritingTask,
)
from src.runtime.state import BackgroundContext, ResearchBrief


def default_research_plan(brief: ResearchBrief, bg: BackgroundContext | None) -> ResearchPlan:
    topic = (brief.clarified_topic or brief.original_query or "").strip() or "调研主题"
    queries: list[str] = []
    if bg and bg.suggested_search_queries:
        queries.extend([q for q in bg.suggested_search_queries if q][:5])
    if bg and bg.expanded_keywords:
        for k in bg.expanded_keywords:
            if k and not re.search(r"[\u4e00-\u9fff]", k):
                queries.append(k)
    seen: set[str] = set()
    uniq: list[str] = []
    for q in queries:
        qn = q.strip()[:400]
        if qn and qn.lower() not in seen:
            seen.add(qn.lower())
            uniq.append(qn)
    if not uniq:
        uniq = [topic[:400]]
    search_tasks = [
        SearchTask(
            query=q,
            source="arxiv",
            time_range=brief.time_range,
            inclusion_criteria=["与澄清主题直接相关", "方法或实验可复现描述"],
            exclusion_criteria=["明显无关学科", "仅专利/marketing 文案"],
            top_k=50,
        )
        for q in uniq[:4]
    ]
    reading_tasks = [
        ReadingTask(
            focus="抽取结构化阅读字段以支撑后续聚类与写作",
            required_fields=[
                "core_problem",
                "methodology",
                "datasets",
                "metrics",
                "main_results",
                "limitations",
                "contributions",
            ],
            extraction_schema_name="ExtractedPaperData",
        )
    ]
    analysis_tasks = [
        AnalysisTask(
            name="主题聚类",
            analysis_type="clustering",
            target_dimensions=["methodology", "datasets"],
            description="按方法与问题设定对论文聚类并提炼主题簇。",
        ),
        AnalysisTask(
            name="方法对比",
            analysis_type="method_comparison",
            target_dimensions=["methodology", "metrics"],
            description="横向对比主流路线的假设、数据与指标。",
        ),
        AnalysisTask(
            name="趋势分析",
            analysis_type="trend_analysis",
            target_dimensions=["evaluation_metrics"],
            description="从时间维度观察指标与任务设定演变。",
        ),
    ]
    writing_tasks = [
        WritingTask(
            section_title="背景与问题定义",
            section_goal="阐明研究背景、任务定义与评价口径",
            required_evidence_types=["abstract", "introduction"],
            expected_length=900,
        ),
        WritingTask(
            section_title="方法体系与对比",
            section_goal="归纳主要技术路线并比较优劣",
            required_evidence_types=["method", "experiment"],
            expected_length=1400,
        ),
        WritingTask(
            section_title="实验、结果与局限性",
            section_goal="汇总数据集、指标、主要发现及不足",
            required_evidence_types=["result", "limitation"],
            expected_length=1200,
        ),
        WritingTask(
            section_title="总结与展望",
            section_goal="提炼共识、分歧与未来方向",
            required_evidence_types=["conclusion", "abstract"],
            expected_length=700,
        ),
    ]
    return ResearchPlan(
        title=f"调研计划：{topic[:80]}",
        thought="基于 brief/背景的默认可执行计划（LLM 失败或未启用时的回退）。",
        has_enough_context=True,
        search_tasks=search_tasks,
        reading_tasks=reading_tasks,
        analysis_tasks=analysis_tasks,
        writing_tasks=writing_tasks,
        evaluation_criteria=[
            "检索覆盖度",
            "方法对比深度",
            "对局限性与负面结果的覆盖",
            "结构是否匹配目标受众",
        ],
    )
