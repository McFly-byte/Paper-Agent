"""将 ResearchPlan 的检索任务注入 config，供 search 节点读取（无 search_agent 依赖）。"""

from __future__ import annotations

import re

from src.core.state_models import PaperAgentState
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def inject_search_plan_hints(val: PaperAgentState) -> None:
    """若存在已批准计划，将 search_tasks 摘要写入 config['search_plan_hints'] 供 search_node 提示词读取。"""
    cfg = val.config if isinstance(val.config, dict) else {}
    val.config = cfg
    plan = getattr(val, "plan", None)
    approved = bool(getattr(val, "plan_approved", False))
    if not plan or not approved:
        cfg.pop("search_plan_hints", None)
        return
    tasks = getattr(plan, "search_tasks", None) or []
    if not tasks:
        cfg.pop("search_plan_hints", None)
        return
    lines: list[str] = [
        "【已批准研究计划中的检索子任务】",
        "生成 arXiv 英文子查询时请优先对齐下列意图（仍需输出合法 arXiv 语法；勿直接粘贴中文）。",
    ]
    for i, t in enumerate(tasks):
        tr = getattr(t, "time_range", (None, None))
        lines.append(
            f"{i + 1}. query={getattr(t, 'query', '')!r} source={getattr(t, 'source', 'arxiv')!r} "
            f"time_range={tr!r} top_k={getattr(t, 'top_k', 50)!r} "
            f"inclusion={getattr(t, 'inclusion_criteria', [])!r} exclusion={getattr(t, 'exclusion_criteria', [])!r}"
        )
    cfg["search_plan_hints"] = "\n".join(lines)
    logger.info("[search_plan_context] 已注入 search_plan_hints（%s 条 search_task）", len(tasks))
