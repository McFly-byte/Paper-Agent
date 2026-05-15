"""Smoke test for Paper-Agent node gates.

Run:
    poetry run python scripts/check_node_gates.py
or:
    python scripts/check_node_gates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.node_gates import (  # noqa: E402
    gate_analyse,
    gate_reading,
    gate_report,
    gate_search,
    gate_writing,
)


def _dump(name: str, result) -> None:
    print(f"\n[{name}] passed={result.passed} score={result.score} next={result.next_action}")
    print("reasons:", result.reasons)
    print("metrics:", json.dumps(result.metrics, ensure_ascii=False, indent=2)[:1600])


def _paper_meta(i: int) -> dict:
    return {
        "paper_id": f"2401.0000{i}",
        "title": f"Retrieval Augmented Generation Agent Reliability Study {i}",
        "summary": "This paper studies retrieval augmented generation, workflow validation gates, "
        "agent reliability, evidence grounding, and failure recovery in LLM systems.",
        "url": f"https://arxiv.org/abs/2401.0000{i}",
    }


def _parsed_paper(i: int) -> dict:
    return {
        "core_problem": "LLM agents need reliable workflow boundary checks.",
        "key_methodology": {
            "name": "Validation gate with RAG evidence checking",
            "principle": "Use deterministic metrics before downstream nodes consume outputs.",
        },
        "main_results": "The guardrail improves failure isolation and report faithfulness.",
        "limitations": "The method still needs offline evaluation.",
        "contributions": ["node-level validation", "recovery feedback"],
    }


def main() -> None:
    search_ok = gate_search(
        {
            "querys": ['("retrieval augmented generation" AND "agent reliability")'],
            "start_date": "2023-01-01",
            "end_date": "2025-01-01",
            "raw_result_count": 8,
            "dedup_count": 8,
            "requested_max_results": 50,
            "top_titles": ["Retrieval Augmented Generation for Reliable LLM Agents"],
            "top_abstracts": ["RAG agent reliability and validation gates for workflow recovery."],
            "user_request": "RAG agent reliability validation gates",
        }
    )
    _dump("search_ok", search_ok)
    assert search_ok.passed
    assert "dedup_ratio" in search_ok.metrics

    search_empty = gate_search({"querys": [], "raw_result_count": 0, "dedup_count": 0})
    _dump("search_empty", search_empty)
    assert not search_empty.passed and search_empty.next_action == "fail"

    search_soft = gate_search(
        {
            "querys": ["AI"],
            "raw_result_count": 10,
            "dedup_count": 10,
            "top_titles": ["AI methods for agents"],
            "user_request": "AI agent",
        }
    )
    _dump("search_soft_warning", search_soft)
    assert search_soft.passed and search_soft.score < 1.0

    metas = [_paper_meta(i) for i in range(5)]
    parsed = [_parsed_paper(i) for i in range(5)]
    reading_ok = gate_reading(
        input_paper_count=5,
        papers_parsed=parsed,
        kb_write_count=5,
        paper_metadatas=metas,
        llamaindex_node_count=25,
    )
    _dump("reading_ok", reading_ok)
    assert reading_ok.passed
    assert "parse_ratio" in reading_ok.metrics

    reading_bad = gate_reading(
        input_paper_count=5,
        papers_parsed=[_parsed_paper(1)],
        kb_write_count=1,
        paper_metadatas=metas[:1],
    )
    _dump("reading_bad_quality", reading_bad)
    assert not reading_bad.passed and reading_bad.next_action in {"retry", "fail"}

    global_text = (
        "技术趋势：RAG Agent 正在从单次问答走向可观测工作流。"
        "方法对比：不同系统会在检索、阅读和写作边界做 deterministic validation。"
        "应用场景：科研综述、企业知识库问答和自动报告都依赖证据链。"
        "研究热点：节点级 guardrail、失败恢复、LangGraph 状态检查和 LLM-as-judge。"
        "局限：轻量规则只能发现结构性问题，不能完全替代人工评审。"
        "建议：后续可以接入离线数据集评估，并把 gate metrics 接到 tracing。"
    ) * 2
    analyse_ok = gate_analyse(
        {
            "isSuccess": True,
            "cluster_summaries": [
                {
                    "theme": "RAG Agent Reliability",
                    "keywords": ["RAG", "agent", "validation"],
                    "paper_count": 3,
                    "representative_papers": ["Retrieval Augmented Generation for Reliable LLM Agents"],
                },
                {
                    "theme": "Workflow Recovery",
                    "keywords": ["LangGraph", "recovery"],
                    "paper_count": 2,
                    "representative_papers": ["Validation Gates for Agent Workflows"],
                },
            ],
            "global_analyse": global_text,
        },
        input_paper_count=5,
    )
    _dump("analyse_ok", analyse_ok)
    assert analyse_ok.passed
    assert "cluster_to_paper_ratio" in analyse_ok.metrics

    analyse_bad = gate_analyse({"cluster_summaries": [], "global_analyse": "太短"}, input_paper_count=5)
    _dump("analyse_bad", analyse_bad)
    assert not analyse_bad.passed

    section_text = (
        "Retrieval Augmented Generation 章节内容讨论 RAG agent reliability、validation gate、"
        "workflow recovery 和 evidence grounding。引用 arXiv 论文作为来源，并说明 [1] 中的关键方法。"
    ) * 3
    writing_ok = gate_writing(
        planned_sections=["1. Retrieval Augmented Generation Agent Reliability", "2. Workflow Recovery"],
        writted_sections=[
            {"content": section_text, "review_verdict": "pass"},
            {"content": section_text.replace("Retrieval Augmented Generation", "Workflow Recovery"), "review_verdict": "pass"},
        ],
        rag_retrieval_logs=[
            {"final_context_count": 4, "section_hint": "Retrieval Augmented Generation"},
            {"final_context_count": 3, "section_hint": "Workflow Recovery"},
        ],
    )
    _dump("writing_ok", writing_ok)
    assert writing_ok.passed
    assert "rag_positive_call_ratio" in writing_ok.metrics

    writing_bad = gate_writing(planned_sections=["1. RAG"], writted_sections=[{"content": "too short", "review_verdict": "fail"}])
    _dump("writing_bad", writing_bad)
    assert not writing_bad.passed

    report = (
        "# RAG Agent 可靠性调研报告\n\n"
        "## 引言\n"
        "本报告基于写作节点的来源章节，分析 RAG agent reliability 与 validation gate。"
        f"{section_text}\n\n"
        "## 主体分析\n"
        "方法对比显示，节点级 workflow boundary check 可以避免低质量中间结果继续污染下游。"
        f"{section_text}\n\n"
        "## 结论与未来方向\n"
        "结论是该项目的策略更接近 Agent 可靠性安全，不是传统网络安全。参考来源包括 arXiv 与写作阶段 RAG 检索结果。"
    )
    report_ok = gate_report(report_markdown=report, section_snippets=[section_text, section_text])
    _dump("report_ok", report_ok)
    assert report_ok.passed
    assert "duplicate_paragraph_ratio" in report_ok.metrics

    report_bad = gate_report(report_markdown="# TODO\n\n待补充", section_snippets=[section_text])
    _dump("report_bad", report_bad)
    assert not report_bad.passed

    print("\nAll node gate smoke checks passed.")


if __name__ == "__main__":
    main()
