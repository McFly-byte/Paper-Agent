#!/usr/bin/env python3
"""
对 LangSmith 已上传的 RAG 数据集运行 SDK 评估（``langsmith.evaluate``）。

用法（在项目根目录）:
  python scripts/run_langsmith_rag_eval.py --dataset RAG_test_LLMAPP --db-id kb_xxxxxxxx
  python scripts/run_langsmith_rag_eval.py --dataset RAG_test_LLMAPP
    # 若已在 yaml/.env 配置 rag_eval_db_id / RAG_EVAL_DB_ID，或应用已写入 current_db_id，可省略 --db-id

  # 无向量库时的对照实验（非 RAG）：
  python scripts/run_langsmith_rag_eval.py --dataset RAG_test_LLMAPP --allow-llm-only

  # 本机 LLM 判分 correctness_local（绕开 LangSmith 云端 Correctness 的 210s）；也可在 system_params 设 rag_eval_local_llm_correctness: true
  python scripts/run_langsmith_rag_eval.py --dataset RAG_test_LLMAPP --local-correctness

需配置：``.env`` LangSmith Key；``models.yaml`` 默认聊天模型。
**RAG 评测**必须能解析到 Chroma ``db_id``（见上）；否则默认报错退出，避免误测成「纯 LLM」。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_env() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass


def main() -> int:
    _bootstrap_env()

    parser = argparse.ArgumentParser(description="LangSmith RAG 数据集评估")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="LangSmith 数据集名称（默认读取 system_params observability.langsmith.rag_eval_dataset）",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="paper-agent-rag",
        help="实验名前缀（LangSmith Experiments 面板）",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="显式关闭检索（与 --allow-llm-only 二选一即可）",
    )
    parser.add_argument(
        "--allow-llm-only",
        action="store_true",
        help="未配置向量库时不报错，退化为纯 LLM（仅作对照，不反映 RAG 检索效果）",
    )
    parser.add_argument(
        "--db-id",
        type=str,
        default=None,
        help="Chroma 知识库 db_id；未传时依次尝试 RAG_EVAL_DB_ID、yaml rag_eval_db_id、current_db_id、tmp_db_id",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="并发数；0 表示完全串行（与 langsmith 默认一致）",
    )
    parser.add_argument(
        "--local-correctness",
        action="store_true",
        help="启用本机 LLM 判分（correctness_local），避免 LangSmith 托管 Correctness 504；建议在 LangSmith 数据集上移除云端 Correctness",
    )
    args = parser.parse_args()

    from src.core.config import config
    from src.evaluation.langsmith_rag_eval import run_rag_dataset_experiment

    dataset = args.dataset or config.get("observability.langsmith.rag_eval_dataset")
    if not dataset:
        print("错误：请指定 --dataset 或在 system_params.yaml 设置 rag_eval_dataset", file=sys.stderr)
        return 1

    if not (os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")):
        print(
            "错误：未检测到 LANGCHAIN_API_KEY 或 LANGSMITH_API_KEY，请在 .env 中配置。",
            file=sys.stderr,
        )
        return 1

    try:
        results = run_rag_dataset_experiment(
            dataset,
            experiment_prefix=args.experiment_prefix,
            db_id=args.db_id,
            use_retrieval=not args.no_retrieval,
            allow_llm_only=args.allow_llm_only or args.no_retrieval,
            local_llm_correctness=True if args.local_correctness else None,
            max_concurrency=args.max_concurrency,
            metadata={"cli": "scripts/run_langsmith_rag_eval.py"},
        )
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1
    exp_name = getattr(results, "experiment_name", None) or str(results)
    print(f"评估完成: {exp_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
