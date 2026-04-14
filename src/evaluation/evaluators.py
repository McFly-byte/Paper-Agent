"""LangSmith 工作流评估：第一层为节点确定性门禁（node_gates），第二层为少量 LLM-as-Judge。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from pydantic import BaseModel, Field

from src.core.llm_infra.invoke import chat_completion_text_routed
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _push_evaluation_feedback(payload: dict[str, Any]) -> None:
    if not (os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")):
        return
    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:
        return
    try:
        tree = get_current_run_tree()
        if tree is None:
            return
        rid = getattr(tree, "id", None) or getattr(tree, "run_id", None)
        if not rid:
            return
        from langsmith import Client

        client = Client()
        rid_str = str(rid)
        overall = payload.get("overall_score")
        if overall is not None:
            client.create_feedback(
                run_id=rid_str,
                key="paper_agent_overall_score",
                score=float(overall),
                comment=str(payload.get("summary", ""))[:500],
            )
        qe = payload.get("quality_eval")
        if isinstance(qe, dict):
            for k in ("analysis_quality", "writing_quality", "report_completeness", "rag_quality"):
                v = qe.get(k)
                if v is not None:
                    try:
                        client.create_feedback(
                            run_id=rid_str,
                            key=f"paper_agent_{k}",
                            score=float(v),
                        )
                    except (TypeError, ValueError):
                        continue
    except Exception as e:
        logger.debug("LangSmith create_feedback 跳过: %s", e)


def _normalize_analyse_for_eval(raw: Any) -> tuple[list[Any], str]:
    if raw is None:
        return [], ""
    if isinstance(raw, dict):
        clusters = raw.get("clusters", []) or raw.get("cluster_summaries", [])
        if not isinstance(clusters, list):
            clusters = []
        try:
            blob = json.dumps(raw, ensure_ascii=False)[:12000]
        except (TypeError, ValueError):
            blob = str(raw)[:12000]
        return clusters, blob
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return [], ""
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return [], s[:12000]
        if isinstance(obj, dict):
            clusters = obj.get("clusters", []) or obj.get("cluster_summaries", [])
            if not isinstance(clusters, list):
                clusters = []
            return clusters, s[:12000]
        return [], s[:12000]
    return [], str(raw)[:12000]


def _normalize_sections_for_eval(sections: Any) -> list[dict[str, Any]]:
    if not sections:
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(sections):
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"title": f"章节 {i + 1}", "content": item})
        else:
            out.append({"title": f"章节 {i + 1}", "content": str(item)})
    return out


def _report_completeness_deterministic(markdown: str, *, section_count: int) -> tuple[float, dict[str, Any]]:
    text = (markdown or "").strip()
    n = len(text)
    hashes = text.count("#")
    metrics = {"chars": n, "heading_hashes": hashes, "source_sections": section_count}
    score = 0.0
    if n >= 800:
        score += 0.35
    elif n >= 400:
        score += 0.2
    if hashes >= 2:
        score += 0.35
    elif hashes >= 1:
        score += 0.2
    if section_count <= 0:
        score += 0.3
    else:
        score += min(0.3, 0.1 * min(section_count, 3))
    return min(1.0, score), metrics


def _rag_quality_heuristic(logs: Any) -> tuple[float, dict[str, Any]]:
    if not isinstance(logs, list) or not logs:
        return 0.55, {"rag_calls": 0, "note": "无 RAG 检索日志，给中性分"}
    n = len(logs)
    return min(1.0, 0.45 + 0.08 * min(n, 6)), {"rag_calls": n}


class DualHeadQualityEval(BaseModel):
    """单次 Judge：分析质量 + 写作质量，减少评估分支与口径漂移。"""

    analysis_quality: float = Field(ge=0.0, le=1.0)
    writing_quality: float = Field(ge=0.0, le=1.0)
    analysis_note: str = ""
    writing_note: str = ""


@traceable(
    run_type="llm",
    name="Workflow Dual-Head Quality Judge",
    tags=["evaluation", "paper-agent", "llm-as-judge"],
)
async def evaluate_dual_head_quality(
    cluster_blob: str,
    sections_summary: str,
) -> DualHeadQualityEval:
    parser = PydanticOutputParser(pydantic_object=DualHeadQualityEval)
    prompt = ChatPromptTemplate.from_template(
        """
你是严格但务实的审稿人。只输出 JSON（不要 Markdown 围栏），字段严格匹配：
- analysis_quality / writing_quality: 0~1 小数
- analysis_note / writing_note: 各不超过 200 字

聚类/全局分析摘录（截断）：
{cluster_blob}

章节摘录（截断）：
{sections_summary}

{fmt}
"""
    )
    formatted = prompt.format(
        cluster_blob=cluster_blob[:6000],
        sections_summary=sections_summary[:4000],
        fmt=parser.get_format_instructions(),
    )
    try:
        text = await chat_completion_text_routed(
            "langsmith-eval-model",
            formatted,
            node_name="workflow_dual_head_judge",
            temperature=0.1,
            source="workflow_dual_head_judge",
        )
        return parser.parse(text)
    except Exception as e:
        logger.warning("双头 Judge 失败: %s", e)
        return DualHeadQualityEval(
            analysis_quality=0.5,
            writing_quality=0.5,
            analysis_note="judge 异常",
            writing_note=str(e)[:120],
        )


@traceable(run_type="chain", name="Run Paper Agent Evaluation", tags=["evaluation", "paper-agent"])
async def run_evaluation(
    run_id: str,
    state: Any,
    outputs: Dict[str, Any],
    reference_output: Optional[Dict] = None,
) -> Dict[str, Any]:
    _ = reference_output

    def safe_get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    node_gates = safe_get(state, "boundary_checks") or {}
    if not isinstance(node_gates, dict):
        node_gates = {}

    gate_scores: list[float] = []
    for k in ("search", "reading", "analyse", "writing", "report"):
        g = node_gates.get(k)
        if isinstance(g, dict) and g.get("score") is not None:
            try:
                gate_scores.append(float(g["score"]))
            except (TypeError, ValueError):
                pass
        elif isinstance(g, dict) and "passed" in g:
            gate_scores.append(1.0 if g.get("passed") else 0.0)
    gate_mean = sum(gate_scores) / len(gate_scores) if gate_scores else 0.5

    raw_analyse = outputs.get("analyse_results") or safe_get(state, "analyse_results")
    _, analyse_blob = _normalize_analyse_for_eval(raw_analyse)
    raw_sections = outputs.get("sections") or safe_get(state, "writted_sections")
    sections = _normalize_sections_for_eval(raw_sections)
    sections_summary = "\n".join(
        f"{s.get('title', '')}: {str(s.get('content', ''))[:500]}" for s in sections[:4]
    )

    blob_ok = bool((analyse_blob or "").strip())
    sec_ok = bool(sections_summary.strip())
    if blob_ok or sec_ok:
        judge = await evaluate_dual_head_quality(
            analyse_blob or "无分析文本",
            sections_summary or "无章节",
        )
    else:
        judge = DualHeadQualityEval(
            analysis_quality=0.5,
            writing_quality=0.5,
            analysis_note="无分析/章节可评",
            writing_note="无分析/章节可评",
        )

    report_md = outputs.get("report_markdown") or safe_get(state, "report_markdown") or ""
    rep_score, rep_metrics = _report_completeness_deterministic(
        str(report_md), section_count=len(sections)
    )

    rag_logs = safe_get(state, "rag_retrieval_logs") or []
    rag_score, rag_metrics = _rag_quality_heuristic(rag_logs)

    quality_eval = {
        "analysis_quality": float(judge.analysis_quality),
        "writing_quality": float(judge.writing_quality),
        "report_completeness": float(rep_score),
        "rag_quality": float(rag_score),
        "notes": {
            "analysis": judge.analysis_note,
            "writing": judge.writing_note,
            "report_metrics": rep_metrics,
            "rag_metrics": rag_metrics,
        },
    }
    qvals = [
        quality_eval["analysis_quality"],
        quality_eval["writing_quality"],
        quality_eval["report_completeness"],
        quality_eval["rag_quality"],
    ]
    quality_mean = sum(qvals) / len(qvals)
    overall = round(0.5 * gate_mean + 0.5 * quality_mean, 4)

    summary = (
        f"gate_mean={gate_mean:.2f}, quality_mean={quality_mean:.2f}；"
        f"analysis={quality_eval['analysis_quality']:.2f}, writing={quality_eval['writing_quality']:.2f}, "
        f"report={quality_eval['report_completeness']:.2f}, rag={quality_eval['rag_quality']:.2f}"
    )

    out: dict[str, Any] = {
        "run_id": run_id,
        "node_gates": node_gates,
        "quality_eval": quality_eval,
        "overall_score": overall,
        "summary": summary,
        "langsmith_traced": True,
    }
    serialized = _to_jsonable(out)
    _push_evaluation_feedback(serialized)
    return serialized


def create_evaluation_dataset() -> str:
    from langsmith import Client

    client = Client()
    dataset_name = "paper-agent-eval-v2"
    try:
        client.create_dataset(
            dataset_name=dataset_name,
            description="Paper-Agent 评估数据集（与 run_evaluation 输出结构对齐）",
        )
        logger.info("数据集 %s 创建成功", dataset_name)
    except Exception as e:
        logger.info("数据集创建跳过: %s", e)
    return dataset_name


__all__ = [
    "run_evaluation",
    "evaluate_dual_head_quality",
    "create_evaluation_dataset",
    "DualHeadQualityEval",
]
