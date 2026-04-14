"""LangSmith SDK：对线上 Dataset 运行 RAG/问答实验并上传指标。

说明：**LangSmith 上的 Dataset 只提供评测用 query / 参考答案**，不包含可被向量检索的语料。
真正 RAG 的「检索」来自本地 **Chroma 知识库**（论文摘要等已写入的 ``db_id``），必须通过
环境变量、yaml 或应用已设置的 ``current_db_id`` 指明库，否则无法测检索链路。

依赖：
  - `.env`：LangSmith Key；可选 ``RAG_EVAL_DB_ID`` 指定评估用向量库
  - ``rag-eval-chat-model`` / ``default-model``：target 生成（LangChain ``ChatOpenAI``）
  - 可选 ``rag-eval-judge-model``：本机 ``correctness_local`` 判分（见 ``rag_eval_local_llm_correctness``），绕开 LangSmith 云端 210s

数据集约定（与 `data/csv/RAGdata.csv` 上传 LangSmith 一致）：
  - inputs: ``query``
  - outputs（参考答案）: ``answer``
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Optional, Sequence

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import aevaluate, traceable
from langsmith.schemas import Example, Run

from src.core.config import config
from src.core.llm_infra.langchain_chat import build_langchain_chat_openai
from src.knowledge.knowledge import knowledge_base

logger = logging.getLogger(__name__)

_lc_rag_eval_chat: ChatOpenAI | None = None
_lc_judge_chat: ChatOpenAI | None = None


def _build_lc_rag_eval_chat() -> ChatOpenAI:
    """与 ``llm-routing`` 对齐的 OpenAI 兼容 Chat（含百炼 batch / thinking）。"""
    return build_langchain_chat_openai("rag-eval-chat-model", node_name="langsmith_rag_eval_target", temperature=0.2, max_retries=3)


def _get_lc_rag_eval_chat() -> ChatOpenAI:
    global _lc_rag_eval_chat
    if _lc_rag_eval_chat is None:
        _lc_rag_eval_chat = _build_lc_rag_eval_chat()
    return _lc_rag_eval_chat


def _build_lc_judge_chat() -> ChatOpenAI:
    """本机 Correctness 判分：路由 ``rag-eval-judge-model``，短超时由 LangSmith yaml 控制。"""
    judge_timeout = float(config.get("observability.langsmith.rag_eval_judge_timeout", 180) or 180)
    return build_langchain_chat_openai(
        "rag-eval-judge-model",
        node_name="langsmith_rag_eval_judge",
        temperature=0.0,
        timeout_override=judge_timeout,
        max_retries=2,
        max_tokens=256,
    )


def _get_lc_judge_chat() -> ChatOpenAI:
    global _lc_judge_chat
    if _lc_judge_chat is None:
        _lc_judge_chat = _build_lc_judge_chat()
    return _lc_judge_chat


def _nonempty_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "null":
        return None
    return s


def resolve_rag_eval_db_id(explicit_db_id: Optional[str] = None) -> tuple[Optional[str], str]:
    """解析用于 LangSmith RAG 评估的 Chroma ``db_id``，返回 (id, 来源说明)。"""
    for label, getter in (
        ("参数 db_id", lambda: _nonempty_str(explicit_db_id)),
        ("环境变量 RAG_EVAL_DB_ID", lambda: _nonempty_str(os.environ.get("RAG_EVAL_DB_ID"))),
        ("system_params rag_eval_db_id", lambda: _nonempty_str(config.get("observability.langsmith.rag_eval_db_id"))),
        ("current_db_id（应用当前选中的知识库）", lambda: _nonempty_str(config.get("current_db_id"))),
        ("tmp_db_id（调研临时向量库）", lambda: _nonempty_str(config.get("tmp_db_id"))),
    ):
        got = getter()
        if got:
            return got, label
    return None, ""


def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _pick_prediction(run: Run) -> str:
    out = run.outputs or {}
    if isinstance(out, dict):
        v = out.get("answer")
        if v is not None:
            return str(v)
        v = out.get("output")
        if v is not None:
            return str(v)
    return ""


def _pick_reference(example: Example) -> str:
    refs = example.outputs or {}
    if not isinstance(refs, dict):
        return ""
    v = refs.get("answer")
    if v is not None:
        return str(v)
    v = refs.get("output")
    if v is not None:
        return str(v)
    return ""


def answer_sequence_similarity(run: Run, example: Example) -> dict[str, Any]:
    """参考答案与模型输出的序列相似度（0~1），适合中文长答案粗对齐。"""
    pred = _norm(_pick_prediction(run))
    ref = _norm(_pick_reference(example))
    if not ref:
        return {"key": "answer_sequence_similarity", "score": None, "comment": "无参考答案"}
    score = SequenceMatcher(None, pred, ref).ratio()
    return {"key": "answer_sequence_similarity", "score": score}


def answer_exact_match(run: Run, example: Example) -> dict[str, Any]:
    pred = _norm(_pick_prediction(run))
    ref = _norm(_pick_reference(example))
    if not ref:
        return {"key": "answer_exact_match", "score": None, "comment": "无参考答案"}
    return {"key": "answer_exact_match", "score": float(pred == ref)}


def _parse_judge_score(raw: str) -> float:
    text = (raw or "").strip()
    if not text:
        return 0.5
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        obj = json.loads(text)
        s = float(obj.get("score", 0.5))
    except (json.JSONDecodeError, TypeError, ValueError):
        m = re.search(r"\"score\"\s*:\s*([0-9]*\.?[0-9]+)", text)
        if m:
            s = float(m.group(1))
        else:
            return 0.5
    return max(0.0, min(1.0, s))


@traceable(run_type="llm", name="Local LLM Correctness", tags=["evaluation", "rag", "judge"])
async def local_llm_correctness(run: Run, example: Example) -> dict[str, Any]:
    """在本机用小模型判断「预测是否覆盖参考答案要点」，分数 0~1；反馈键为 ``correctness_local``（与 LangSmith 云端 Correctness 区分）。"""
    pred = _pick_prediction(run)
    ref = _pick_reference(example)
    if not ref:
        return {"key": "correctness_local", "score": None, "comment": "无参考答案"}

    max_c = max(500, config.get_int("observability.langsmith.rag_eval_judge_max_chars", 2400))
    ref_t = ref[:max_c]
    pred_t = pred[:max_c]

    judge_prompt = (
        "你是严格且简洁的阅卷助手。给定「参考答案」与「模型预测」，只判断预测是否在事实上覆盖参考要点"
        "（允许表述不同；预测多写无关内容可扣分）。\n"
        "只输出一行 JSON，不要其它文字：{\"score\": <0到1之间的小数>, \"brief\": \"不超过40字\"}\n\n"
        f"【参考答案】\n{ref_t}\n\n【模型预测】\n{pred_t}"
    )
    try:
        chat = _get_lc_judge_chat()
        msg = await chat.ainvoke([HumanMessage(content=judge_prompt)])
        raw = getattr(msg, "content", None) or ""
        if isinstance(raw, list):
            raw = str(raw)
        score = _parse_judge_score(str(raw))
        return {"key": "correctness_local", "score": score, "comment": str(raw)[:200]}
    except Exception as e:
        logger.warning("本机 correctness 判分失败: %s", e)
        return {"key": "correctness_local", "score": None, "comment": str(e)[:200]}


def default_rag_evaluators(*, include_local_correctness: bool) -> list[Any]:
    evs: list[Any] = [answer_sequence_similarity, answer_exact_match]
    if include_local_correctness:
        evs.append(local_llm_correctness)
    return evs


async def _retrieve_contexts(db_id: str, query: str) -> list[str]:
    top_k = config.get_int("observability.langsmith.rag_eval_top_k", 5)
    thr = config.get_float("observability.langsmith.rag_eval_similarity_threshold", 0.0)
    hits = await knowledge_base.aquery(query, db_id=db_id, top_k=top_k, similarity_threshold=thr)
    texts: list[str] = []
    for h in hits or []:
        md = h.get("metadata") or {}
        if md.get("state_blob") == "1":
            continue
        c = (h.get("content") or "").strip()
        if c:
            texts.append(c)
    return texts


async def _generate_answer(query: str, contexts: list[str]) -> str:
    if contexts:
        ctx = "\n\n---\n\n".join(contexts[:12])[:12000]
        prompt = (
            "你是严谨的学术问答助手。请仅根据下列「检索到的材料」回答问题；"
            "材料不足以回答时请明确说明，不要编造。\n\n"
            f"【材料】\n{ctx}\n\n【问题】\n{query}\n\n请用中文直接给出答案，勿重复问题。"
        )
    else:
        prompt = (
            "你是严谨的学术问答助手。请用中文直接回答下列问题；若你不确定请说明不确定，不要编造。\n\n"
            f"【问题】\n{query}"
        )
    chat = _get_lc_rag_eval_chat()
    msg = await chat.ainvoke([HumanMessage(content=prompt)])
    raw = getattr(msg, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()


async def _predict_async(inputs: dict[str, Any], db_id: Optional[str], use_retrieval: bool) -> dict[str, Any]:
    query = inputs.get("query") or inputs.get("question") or ""
    query = str(query).strip()
    if not query:
        return {"answer": "", "error": "empty_query"}

    contexts: list[str] = []
    if use_retrieval and db_id:
        try:
            contexts = await _retrieve_contexts(db_id, query)
        except Exception as e:
            logger.warning("RAG 检索失败，将退化为无检索生成: %s", e)

    try:
        answer = await _generate_answer(query, contexts)
    except Exception as e:
        logger.exception("RAG 评估生成失败")
        return {"answer": "", "error": str(e)[:500]}

    preview = "\n\n---\n\n".join(contexts[:3])[:4000] if contexts else ""
    return {
        "answer": answer,
        "retrieved_context_preview": preview or None,
        "retrieval_hits": len(contexts),
    }


def make_async_rag_target(
    *,
    db_id: Optional[str] = None,
    use_retrieval: bool = True,
) -> Callable[[dict[str, Any]], Any]:
    """返回供 ``langsmith.aevaluate`` 使用的异步 target（单事件循环，避免 ``asyncio.run`` 与 httpx 清理冲突）。"""

    @traceable(run_type="chain", name="Paper-Agent RAG Eval Target", tags=["evaluation", "rag"])
    async def predict(inputs: dict[str, Any]) -> dict[str, Any]:
        return await _predict_async(inputs, db_id, use_retrieval)

    return predict


async def arun_rag_dataset_experiment(
    dataset_name: str,
    *,
    experiment_prefix: str = "paper-agent-rag",
    db_id: Optional[str] = None,
    use_retrieval: bool = True,
    allow_llm_only: bool = False,
    evaluators: Optional[Sequence[Callable[[Run, Example], dict[str, Any]]]] = None,
    local_llm_correctness: Optional[bool] = None,
    max_concurrency: int = 4,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """对 LangSmith 上已存在的 Dataset 运行一次实验（异步 API，返回 ``AsyncExperimentResults``）。"""
    resolved_db, db_source = resolve_rag_eval_db_id(db_id)
    if use_retrieval and not resolved_db:
        msg = (
            "无法做 RAG 评估：未解析到任何 Chroma 知识库 db_id。"
            "LangSmith Dataset 不含向量语料，请指定本地库，例如：\n"
            "  - .env 中设置 RAG_EVAL_DB_ID=kb_xxxxxxxx\n"
            "  - 或 system_params.yaml 中 observability.langsmith.rag_eval_db_id\n"
            "  - 或先在 Web 端选中知识库（写入 current_db_id）后再运行脚本\n"
            "  - 或命令行：--db-id kb_xxxxxxxx\n"
            "若仅需无检索对照实验，请传 allow_llm_only=True 或 CLI --allow-llm-only。"
        )
        if allow_llm_only:
            logger.warning("%s 已允许 allow_llm_only：将按「纯 LLM」生成。", msg.replace("\n", " "))
            use_retrieval = False
        else:
            raise ValueError(msg)

    if use_retrieval and resolved_db:
        logger.info("LangSmith RAG 评估检索库: %s（来源：%s）", resolved_db, db_source)

    target = make_async_rag_target(db_id=resolved_db, use_retrieval=use_retrieval)
    if evaluators is not None:
        evs = list(evaluators)
    else:
        use_local = (
            local_llm_correctness
            if local_llm_correctness is not None
            else config.get_bool("observability.langsmith.rag_eval_local_llm_correctness", False)
        )
        evs = default_rag_evaluators(include_local_correctness=use_local)

    return await aevaluate(
        target,
        data=dataset_name,
        evaluators=evs,
        experiment_prefix=experiment_prefix,
        description=description or "Paper-Agent LangSmith RAG 数据集评估",
        max_concurrency=max_concurrency,
        metadata=metadata or {},
    )


def run_rag_dataset_experiment(
    dataset_name: str,
    *,
    experiment_prefix: str = "paper-agent-rag",
    db_id: Optional[str] = None,
    use_retrieval: bool = True,
    allow_llm_only: bool = False,
    evaluators: Optional[Sequence[Callable[[Run, Example], dict[str, Any]]]] = None,
    local_llm_correctness: Optional[bool] = None,
    max_concurrency: int = 4,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """同步入口：在进程级单事件循环上跑完 ``aevaluate``（避免每条样本 ``asyncio.run`` 关闭循环导致 httpx 报错）。"""
    return asyncio.run(
        arun_rag_dataset_experiment(
            dataset_name,
            experiment_prefix=experiment_prefix,
            db_id=db_id,
            use_retrieval=use_retrieval,
            allow_llm_only=allow_llm_only,
            evaluators=evaluators,
            local_llm_correctness=local_llm_correctness,
            max_concurrency=max_concurrency,
            description=description,
            metadata=metadata,
        )
    )
