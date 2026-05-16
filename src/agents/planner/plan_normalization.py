"""将 Planner LLM 返回的宽松 JSON 规范为 ``ResearchPlan`` 可校验结构（无 AutoGen / model_client 依赖）。"""

from __future__ import annotations

from src.runtime.state import ResearchBrief


def _str_list(val: object) -> list[str]:
    """将 LLM 常返回的「单段中文说明」或混用类型规范为 str 列表。"""
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, list):
        out: list[str] = []
        for x in val:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif x is not None:
                out.append(str(x).strip())
        return [x for x in out if x]
    return [str(val).strip()] if str(val).strip() else []


_ANALYSIS_TYPES = frozenset(
    {"clustering", "method_comparison", "trend_analysis", "limitation_summary", "metric_comparison"}
)


def _normalize_search_task(raw: object) -> dict:
    if isinstance(raw, str):
        return {"query": raw.strip()[:400] or "survey review"}
    if not isinstance(raw, dict):
        return {"query": "survey review"}
    d = dict(raw)
    q = (d.get("query") or d.get("search_query") or "").strip()
    if not q:
        q = (d.get("description") or d.get("topic") or "survey benchmark")[:400]
    d["query"] = q
    d["inclusion_criteria"] = _str_list(d.get("inclusion_criteria"))
    d["exclusion_criteria"] = _str_list(d.get("exclusion_criteria"))
    src = d.get("source", "arxiv")
    if src not in ("arxiv", "local_pdf", "web", "vector_db"):
        d["source"] = "arxiv"
    else:
        d["source"] = src
    tr = d.get("time_range")
    if tr is None:
        d["time_range"] = (None, None)
    elif isinstance(tr, (list, tuple)) and len(tr) >= 2:
        d["time_range"] = (tr[0], tr[1])
    else:
        d["time_range"] = (None, None)
    try:
        d["top_k"] = int(d.get("top_k", 50))
    except (TypeError, ValueError):
        d["top_k"] = 50
    return d


def _normalize_reading_task(raw: object, idx: int) -> dict:
    if not isinstance(raw, dict):
        return {"focus": f"阅读任务 {idx + 1}"}
    d = dict(raw)
    focus = (
        d.get("focus")
        or d.get("description")
        or d.get("goal")
        or d.get("summary")
        or d.get("task_description")
        or d.get("objective")
    )
    if isinstance(focus, str) and focus.strip():
        d["focus"] = focus.strip()[:2000]
    else:
        tid = str(d.get("task_id") or d.get("id") or "").strip()
        d["focus"] = (f"阅读子任务：{tid}" if tid else f"结构化阅读任务 {idx + 1}")
    d["required_fields"] = _str_list(d.get("required_fields"))
    if not d["required_fields"]:
        d["required_fields"] = [
            "core_problem",
            "key_methodology",
            "datasets_used",
            "evaluation_metrics",
            "main_results",
            "limitations",
            "contributions",
        ]
    esn = d.get("extraction_schema_name")
    d["extraction_schema_name"] = esn.strip() if isinstance(esn, str) and esn.strip() else None
    return {
        "focus": d["focus"],
        "required_fields": d["required_fields"],
        "extraction_schema_name": d["extraction_schema_name"],
    }


def _normalize_analysis_task(raw: object, idx: int) -> dict:
    if not isinstance(raw, dict):
        return {"name": f"分析任务 {idx + 1}", "analysis_type": "clustering"}
    d = dict(raw)
    name = (d.get("name") or d.get("title") or "").strip()
    if not name:
        tid = str(d.get("task_id") or d.get("id") or "").strip()
        name = tid.replace("_", " ") if tid else f"分析任务 {idx + 1}"
    d["name"] = name[:500]
    at = d.get("analysis_type")
    if isinstance(at, str) and at.strip() in _ANALYSIS_TYPES:
        d["analysis_type"] = at.strip()
    else:
        d["analysis_type"] = "clustering"
    d["target_dimensions"] = _str_list(d.get("target_dimensions"))
    if not d["target_dimensions"]:
        d["target_dimensions"] = ["methodology"]
    desc = d.get("description")
    d["description"] = desc.strip()[:4000] if isinstance(desc, str) else (str(desc) if desc is not None else None)
    return {k: d[k] for k in ("name", "analysis_type", "target_dimensions", "description")}


def _normalize_writing_task(raw: object, idx: int) -> dict:
    if not isinstance(raw, dict):
        return {
            "section_title": f"第 {idx + 1} 节",
            "section_goal": "完成本节写作目标",
            "required_evidence_types": [],
            "expected_length": None,
        }
    d = dict(raw)
    title = (d.get("section_title") or d.get("title") or "").strip() or f"第 {idx + 1} 节"
    goal = (d.get("section_goal") or d.get("goal") or d.get("description") or "").strip() or "完成本节写作目标"
    d["section_title"] = title[:500]
    d["section_goal"] = goal[:2000]
    d["required_evidence_types"] = _str_list(d.get("required_evidence_types"))
    el = d.get("expected_length")
    try:
        d["expected_length"] = int(el) if el is not None else None
    except (TypeError, ValueError):
        d["expected_length"] = None
    return {k: d[k] for k in ("section_title", "section_goal", "required_evidence_types", "expected_length")}


def normalize_plan_dict(data: dict, brief: ResearchBrief) -> dict:
    """将 LLM 非严格 JSON 规范到 ``ResearchPlan.model_validate`` 可接受的 dict。"""
    d = dict(data or {})
    topic = (brief.clarified_topic or brief.original_query or "").strip() or "调研主题"

    if not (d.get("title") or "").strip():
        thought = (d.get("thought") or "").strip()
        first = thought.splitlines()[0].strip()[:120] if thought else ""
        d["title"] = first or f"调研计划：{topic[:80]}"
    if d.get("thought") is None:
        d["thought"] = ""
    if not isinstance(d.get("thought"), str):
        d["thought"] = str(d["thought"])
    hec = d.get("has_enough_context")
    if not isinstance(hec, bool):
        d["has_enough_context"] = bool(hec) if hec is not None else True

    st = d.get("search_tasks")
    if isinstance(st, list):
        d["search_tasks"] = [_normalize_search_task(x) for x in st]
    else:
        d["search_tasks"] = []

    rt = d.get("reading_tasks")
    if isinstance(rt, list):
        d["reading_tasks"] = [_normalize_reading_task(x, i) for i, x in enumerate(rt)]
    else:
        d["reading_tasks"] = []

    ats = d.get("analysis_tasks")
    if isinstance(ats, list):
        d["analysis_tasks"] = [_normalize_analysis_task(x, i) for i, x in enumerate(ats)]
    else:
        d["analysis_tasks"] = []

    wt = d.get("writing_tasks")
    if isinstance(wt, list):
        d["writing_tasks"] = [_normalize_writing_task(x, i) for i, x in enumerate(wt)]
    else:
        d["writing_tasks"] = []

    d["evaluation_criteria"] = _str_list(d.get("evaluation_criteria"))
    return d
