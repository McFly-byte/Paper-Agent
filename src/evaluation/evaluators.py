"""LangSmith 工作流评估：第一层为节点确定性门禁（node_gates）；第二层为 LLM-as-Judge（分析/写作各四维分项，代码加权聚合）。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field

from src.core.llm_infra.invoke import chat_completion_text_routed
from src.core.prompts import workflow_dual_head_judge_prompt
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


def _section_substance_ratio(sections: list[dict[str, Any]], *, min_chars: int = 120) -> float:
    """有实质正文的章节占比（与 node_gates.writing 口径接近，用于校准 LLM Judge）。"""
    if not sections:
        return 0.0
    ok = 0
    for s in sections:
        c = str(s.get("content") or "").strip()
        if len(c) >= min_chars:
            ok += 1
    return ok / len(sections)


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
    # 与计划章节数对齐的奖励；无元数据时不白送分（此前 section_count<=0 固定 +0.3 会虚高）
    if section_count > 0:
        score += min(0.3, 0.1 * min(section_count, 3))
    return min(1.0, score), metrics


def _rag_quality_heuristic(logs: Any) -> tuple[float, dict[str, Any]]:
    if not isinstance(logs, list) or not logs:
        return 0.55, {"rag_calls": 0, "note": "无 RAG 检索日志，给中性分"}
    n = len(logs)
    return min(1.0, 0.45 + 0.08 * min(n, 6)), {"rag_calls": n}


def _rag_logs_excerpt_for_judge(logs: Any, *, max_chars: int = 3200) -> str:
    """将 RAG 调用摘要转为 Judge 可读文本（支撑「证据使用」维度）。"""
    if not isinstance(logs, list) or not logs:
        return "（无 RAG 检索日志；请按「无证据可查」保守评估 writing_evidence_usage，勿臆造检索内容。）"
    parts: list[str] = []
    for i, entry in enumerate(logs[-40:]):
        if not isinstance(entry, dict):
            parts.append(f"[{i}] {str(entry)[:400]}")
            continue
        try:
            parts.append(json.dumps(entry, ensure_ascii=False)[:600])
        except (TypeError, ValueError):
            parts.append(f"[{i}] {str(entry)[:400]}")
    blob = "\n".join(parts)
    if len(blob) > max_chars:
        return blob[:max_chars] + "\n…（截断）"
    return blob


# 分析四维权重：覆盖度、技术深度、结构/聚类一致性、洞见
W_ANALYSIS_COVERAGE = 0.25
W_ANALYSIS_TECH_DEPTH = 0.35
W_ANALYSIS_CLUSTER = 0.20
W_ANALYSIS_INSIGHT = 0.20

# 写作四维权重：忠实性、连贯性、证据使用、表达规范
W_WRITING_FAITH = 0.35
W_WRITING_COHERENCE = 0.25
W_WRITING_EVIDENCE = 0.25
W_WRITING_STYLE = 0.15


class DualHeadDimensionScores(BaseModel):
    """LLM 只输出各维度 0~1 分与短评；总分由代码按权重聚合。"""

    model_config = ConfigDict(extra="ignore")

    analysis_coverage: float = Field(ge=0.0, le=1.0, description="分析覆盖度")
    analysis_technical_depth: float = Field(ge=0.0, le=1.0, description="技术深度")
    analysis_cluster_consistency: float = Field(ge=0.0, le=1.0, description="结构/聚类一致性")
    analysis_insight: float = Field(ge=0.0, le=1.0, description="洞见价值")
    writing_faithfulness: float = Field(ge=0.0, le=1.0, description="相对分析的忠实性")
    writing_coherence: float = Field(ge=0.0, le=1.0, description="结构与连贯性")
    writing_evidence_usage: float = Field(ge=0.0, le=1.0, description="论断与证据/RAG 对齐度")
    writing_style: float = Field(ge=0.0, le=1.0, description="表达与学术规范")
    analysis_note: str = ""
    writing_note: str = ""


class DualHeadQualityEval(BaseModel):
    """聚合后的双头评估：分项 + 加权总分（与历史字段 analysis_quality / writing_quality 对齐）。"""

    analysis_quality: float = Field(ge=0.0, le=1.0)
    writing_quality: float = Field(ge=0.0, le=1.0)
    analysis_note: str = ""
    writing_note: str = ""
    analysis_dimensions: dict[str, float] = Field(default_factory=dict)
    writing_dimensions: dict[str, float] = Field(default_factory=dict)


def _aggregate_dual_head(dims: DualHeadDimensionScores) -> DualHeadQualityEval:
    aq = (
        W_ANALYSIS_COVERAGE * dims.analysis_coverage
        + W_ANALYSIS_TECH_DEPTH * dims.analysis_technical_depth
        + W_ANALYSIS_CLUSTER * dims.analysis_cluster_consistency
        + W_ANALYSIS_INSIGHT * dims.analysis_insight
    )
    wq = (
        W_WRITING_FAITH * dims.writing_faithfulness
        + W_WRITING_COHERENCE * dims.writing_coherence
        + W_WRITING_EVIDENCE * dims.writing_evidence_usage
        + W_WRITING_STYLE * dims.writing_style
    )
    return DualHeadQualityEval(
        analysis_quality=round(max(0.0, min(1.0, aq)), 4),
        writing_quality=round(max(0.0, min(1.0, wq)), 4),
        analysis_note=(dims.analysis_note or "")[:500],
        writing_note=(dims.writing_note or "")[:500],
        analysis_dimensions={
            "coverage": round(float(dims.analysis_coverage), 4),
            "technical_depth": round(float(dims.analysis_technical_depth), 4),
            "cluster_consistency": round(float(dims.analysis_cluster_consistency), 4),
            "insight": round(float(dims.analysis_insight), 4),
        },
        writing_dimensions={
            "faithfulness": round(float(dims.writing_faithfulness), 4),
            "coherence": round(float(dims.writing_coherence), 4),
            "evidence_usage": round(float(dims.writing_evidence_usage), 4),
            "style": round(float(dims.writing_style), 4),
        },
    )


@traceable(
    run_type="llm",
    name="Workflow Dual-Head Quality Judge",
    tags=["evaluation", "paper-agent", "llm-as-judge"],
)
async def evaluate_dual_head_quality(
    analyse_material: str,
    sections_text: str,
    rag_excerpt: str,
) -> DualHeadQualityEval:
    parser = PydanticOutputParser(pydantic_object=DualHeadDimensionScores)
    prompt = ChatPromptTemplate.from_template(workflow_dual_head_judge_prompt)
    formatted = prompt.format(
        analyse_material=(analyse_material or "无")[:8000],
        sections_text=(sections_text or "无")[:8000],
        rag_excerpt=(rag_excerpt or "")[:3500],
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
        dims = parser.parse(text)
        return _aggregate_dual_head(dims)
    except Exception as e:
        logger.warning("双头 Judge 失败: %s", e)
        return DualHeadQualityEval(
            analysis_quality=0.4,
            writing_quality=0.35,
            analysis_note="judge 异常",
            writing_note=str(e)[:200],
            analysis_dimensions={},
            writing_dimensions={},
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
    gate_mean: Optional[float]
    if gate_scores:
        gate_mean = sum(gate_scores) / len(gate_scores)
    else:
        gate_mean = None

    raw_analyse = outputs.get("analyse_results") or safe_get(state, "analyse_results")
    _, analyse_blob = _normalize_analyse_for_eval(raw_analyse)
    raw_sections = outputs.get("sections") or safe_get(state, "writted_sections")
    sections = _normalize_sections_for_eval(raw_sections)
    sections_text = "\n\n".join(
        f"## {s.get('title', '') or f'章节 {i + 1}'}\n{str(s.get('content', ''))[:1200]}"
        for i, s in enumerate(sections[:8])
    )

    rag_logs = safe_get(state, "rag_retrieval_logs") or []
    rag_excerpt = _rag_logs_excerpt_for_judge(rag_logs)

    blob_ok = bool((analyse_blob or "").strip())
    sec_ok = bool(sections_text.strip())
    if blob_ok or sec_ok:
        judge = await evaluate_dual_head_quality(
            analyse_blob or "无分析文本",
            sections_text or "无章节",
            rag_excerpt,
        )
    else:
        judge = DualHeadQualityEval(
            analysis_quality=0.35,
            writing_quality=0.3,
            analysis_note="无分析/章节可评",
            writing_note="无分析/章节可评",
            analysis_dimensions={},
            writing_dimensions={},
        )

    report_md = outputs.get("report_markdown") or safe_get(state, "report_markdown") or ""
    rep_score, rep_metrics = _report_completeness_deterministic(
        str(report_md), section_count=len(sections)
    )

    rag_score, rag_metrics = _rag_quality_heuristic(rag_logs)

    substance = _section_substance_ratio(sections)
    wg = node_gates.get("writing") if isinstance(node_gates, dict) else None
    writing_gate_failed = isinstance(wg, dict) and wg.get("passed") is False
    wg_score = float(wg["score"]) if isinstance(wg, dict) and wg.get("score") is not None else None

    quality_eval = {
        "analysis_quality": float(judge.analysis_quality),
        "writing_quality": float(judge.writing_quality),
        "analysis_dimensions": dict(judge.analysis_dimensions),
        "writing_dimensions": dict(judge.writing_dimensions),
        "analysis_score_weights": {
            "coverage": W_ANALYSIS_COVERAGE,
            "technical_depth": W_ANALYSIS_TECH_DEPTH,
            "cluster_consistency": W_ANALYSIS_CLUSTER,
            "insight": W_ANALYSIS_INSIGHT,
        },
        "writing_score_weights": {
            "faithfulness": W_WRITING_FAITH,
            "coherence": W_WRITING_COHERENCE,
            "evidence_usage": W_WRITING_EVIDENCE,
            "style": W_WRITING_STYLE,
        },
        "report_completeness": float(rep_score),
        "rag_quality": float(rag_score),
        "notes": {
            "analysis": judge.analysis_note,
            "writing": judge.writing_note,
            "report_metrics": rep_metrics,
            "rag_metrics": rag_metrics,
            "section_substance_ratio": round(substance, 4),
            "writing_gate_failed": writing_gate_failed,
        },
    }

    # 校准：写作门禁未通过或章节实质为空时，压低写作/报告分，避免「全文报错仍 0.8+」
    if writing_gate_failed:
        cap_w = min(float(quality_eval["writing_quality"]), float(wg_score) if wg_score is not None else 0.2, 0.22)
        quality_eval["writing_quality"] = max(0.0, cap_w)
        smr = float((wg.get("metrics") or {}).get("section_match_ratio") or substance)
        quality_eval["report_completeness"] = min(
            float(quality_eval["report_completeness"]),
            float(rep_score) * (0.25 + 0.75 * max(smr, substance, 0.05)),
        )
    elif len(sections) >= 2 and substance < 0.12:
        quality_eval["writing_quality"] = min(
            float(quality_eval["writing_quality"]),
            0.18 + 0.65 * substance,
        )

    qvals = [
        quality_eval["analysis_quality"],
        quality_eval["writing_quality"],
        quality_eval["report_completeness"],
        quality_eval["rag_quality"],
    ]
    quality_mean = sum(qvals) / len(qvals)
    if gate_mean is not None:
        overall = round(0.5 * gate_mean + 0.5 * quality_mean, 4)
    else:
        overall = round(quality_mean, 4)
    if writing_gate_failed:
        smr_gate = float((wg.get("metrics") or {}).get("section_match_ratio") or 0.0)
        recovery = max(substance, smr_gate, 0.0)
        # 写作为关键路径：门禁失败时总分随章节恢复度接近 0.45 上限，避免「其它节点全绿仍 0.8+」
        overall = min(overall, round(0.38 + 0.42 * recovery, 4))

    gm_str = f"{gate_mean:.2f}" if gate_mean is not None else "n/a"
    summary = (
        f"gate_mean={gm_str}, quality_mean={quality_mean:.2f}；"
        f"analysis={quality_eval['analysis_quality']:.2f}, writing={quality_eval['writing_quality']:.2f}, "
        f"report={quality_eval['report_completeness']:.2f}, rag={quality_eval['rag_quality']:.2f}"
    )
    if writing_gate_failed:
        summary += "；写作门禁未通过（已校准写作/报告分项）"

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
    "DualHeadDimensionScores",
]
