"""Paper-Agent 评估模块
基于 LangSmith 实现 LLM-as-Judge 评估体系（已修复导入问题）。
当前暴露：run_evaluation + 两个核心 Judge 函数 + 数据集创建工具。
"""

from .evaluators import (
    run_evaluation,
    evaluate_analysis_quality,
    evaluate_writing_quality,
    create_evaluation_dataset,
    AnalysisQualityEval,
    WritingQualityEval,
    ReportCompletenessEval,
    RAGEval,
)
from .langsmith_rag_eval import (
    arun_rag_dataset_experiment,
    default_rag_evaluators,
    local_llm_correctness,
    make_async_rag_target,
    run_rag_dataset_experiment,
)

__all__ = [
    "run_evaluation",
    "evaluate_analysis_quality",
    "evaluate_writing_quality",
    "create_evaluation_dataset",
    "AnalysisQualityEval",
    "WritingQualityEval",
    "ReportCompletenessEval",
    "RAGEval",
    "make_async_rag_target",
    "arun_rag_dataset_experiment",
    "run_rag_dataset_experiment",
    "default_rag_evaluators",
    "local_llm_correctness",
]
