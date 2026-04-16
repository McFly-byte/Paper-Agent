"""LangSmith SDK：对线上 Dataset 运行 RAG/问答实验并上传指标。

说明：**LangSmith 上的 Dataset 只提供评测用 query / 参考答案**，不包含可被向量检索的语料。
真正 RAG 的「检索」来自本地 **Chroma 知识库**（论文摘要等已写入的 ``db_id``），必须通过
环境变量、yaml 或应用已设置的 ``current_db_id`` 指明库，否则无法测检索链路。

依赖：
  - `.env`：LangSmith Key；可选 ``RAG_EVAL_DB_ID`` 指定评估用向量库
  - ``rag-eval-chat-model`` / ``default-model``：target 生成（LangChain ``ChatOpenAI``）
  - 可选 ``rag-eval-judge-model``：本机 ``correctness_local`` 八维量表判分（见 ``local_llm_correctness``），绕开 LangSmith 云端 210s

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
from src.knowledge.knowledge import knowledge_base

logger = logging.getLogger(__name__)

_lc_rag_eval_chat: ChatOpenAI | None = None
_lc_judge_chat: ChatOpenAI | None = None


def _build_lc_rag_eval_chat() -> ChatOpenAI:
    """从 models.yaml 构建 OpenAI 兼容 Chat 客户端（硅基 / Ollama 等），供 LangSmith 采集 usage。"""
    block = config.get("rag-eval-chat-model")
    if not isinstance(block, dict) or not block.get("model"):
        block = config.get("default-model") or {}
    provider = block.get("model-provider") or "dashscope"
    model = block.get("model")
    if not model:
        raise ValueError("rag-eval-chat-model / default-model 缺少 model")
    pcfg = config.get(provider) or {}
    api_key = pcfg.get("api_key")
    base_url = (pcfg.get("base_url") or "").strip().rstrip("/")
    if not api_key:
        raise ValueError(f"RAG 评估：{provider} 未配置 api_key（请检查 .env 与 models.yaml）")
    if not base_url:
        raise ValueError(f"RAG 评估：{provider} 未配置 base_url")
    timeout = float(pcfg.get("request_timeout", 600))
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        timeout=timeout,
        max_retries=3,
    )


def _get_lc_rag_eval_chat() -> ChatOpenAI:
    global _lc_rag_eval_chat
    if _lc_rag_eval_chat is None:
        _lc_rag_eval_chat = _build_lc_rag_eval_chat()
    return _lc_rag_eval_chat


def _build_lc_judge_chat() -> ChatOpenAI:
    """本机 Correctness 判分用小模型、低温度、较短超时。"""
    block = config.get("rag-eval-judge-model")
    if not isinstance(block, dict) or not block.get("model"):
        block = config.get("rag-eval-chat-model") or config.get("default-model") or {}
    provider = block.get("model-provider") or "dashscope"
    model = block.get("model")
    if not model:
        raise ValueError("rag-eval-judge-model 缺少 model")
    pcfg = config.get(provider) or {}
    api_key = pcfg.get("api_key")
    base_url = (pcfg.get("base_url") or "").strip().rstrip("/")
    if not api_key or not base_url:
        raise ValueError(f"判分模型：{provider} 缺少 api_key 或 base_url")
    judge_timeout = float(
        config.get("observability.langsmith.rag_eval_judge_timeout", 180) or 180
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        timeout=judge_timeout,
        max_retries=2,
        max_tokens=1024,
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


def _strip_llm_json_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```\s*$", "", text).strip()
    return text


_DUAL_HEAD_SCORE_KEYS = (
    "analysis_coverage",
    "analysis_technical_depth",
    "analysis_cluster_consistency",
    "analysis_insight",
    "writing_faithfulness",
    "writing_coherence",
    "writing_evidence_usage",
    "writing_style",
)


WORKFLOW_DUAL_HEAD_JUDGE_PROMPT = """
你是严格但务实的学术调研审稿人。根据下列材料，为「分析阶段」与「写作阶段」分别打出 **8 个 0~1 之间的小数分**（可保留两位小数），并各给一句不超过 200 字的短评。

## 输出要求
- **只输出一段合法 JSON**（不要 Markdown 代码围栏，不要前后解释）。
- 字段名必须完全一致（英文 snake_case），数值必须在 [0, 1]。
- 字段：analysis_coverage, analysis_technical_depth, analysis_cluster_consistency, analysis_insight,
  writing_faithfulness, writing_coherence, writing_evidence_usage, writing_style,
  analysis_note, writing_note

## 分析阶段 — 四维判分锚点（请据此选分，勿虚高）

**1. analysis_coverage（覆盖度）**：分析是否覆盖应分析的输入材料（主要主题、各簇、关键方法路线），而非只挑少数文献展开。
- 0.90–1.00：主要主题、主要簇、关键方法路线覆盖充分，无明显遗漏
- 0.70–0.89：大体完整，少量簇或关键路线展开不足
- 0.40–0.69：只覆盖部分主题，有明显缺口
- 0.00–0.39：大量遗漏，难以支撑后续写作

**2. analysis_technical_depth（技术深度）**：是否真正做方法拆解、比较、归纳（不只堆字数）。重点：方法差异比较、适用场景、指标/实验现象、局限总结。
- 0.90–1.00：能比较路线差异、实验表现、适用边界与局限
- 0.70–0.89：有分析与比较，深度一般或证据偏稀
- 0.40–0.69：偏摘要式复述，缺少实质分析
- 0.00–0.39：流水账或事实堆砌

**3. analysis_cluster_consistency（结构一致性/聚类合理性）**：聚类是否有逻辑；簇内是否同类、簇间是否区分（对应 cluster / deep / global 结构）。
- 0.90–1.00：主题边界清晰，簇内一致，簇间区分明显
- 0.70–0.89：大体合理，个别簇混杂
- 0.40–0.69：勉强可用，组织混乱
- 0.00–0.39：看不出清晰结构

**4. analysis_insight（洞见价值）**：是否超越摘要层：趋势判断、关键矛盾、有启发的未来方向。
- 0.90–1.00：跨论文趋势、关键矛盾或清晰未来方向
- 0.70–0.89：有总结提升但不够深
- 0.40–0.69：以复述为主，少量泛化
- 0.00–0.39：几乎无高层洞见

## 写作阶段 — 四维判分锚点

**5. writing_faithfulness（忠实性）**：章节相对下方「分析材料」是否忠实；是否引入分析未支持的新事实或结论（最重要）。
- 0.90–1.00：忠实于分析输出，无明显杜撰
- 0.70–0.89：基本忠实，少量扩写未明显偏离
- 0.40–0.69：若干未经支持的扩展
- 0.00–0.39：明显新事实或偏离原分析

**6. writing_coherence（结构与连贯性）**：段落组织、节间过渡、重复与跳跃。
- 0.90–1.00：结构清晰，过渡自然，重复少
- 0.70–0.89：整体顺畅，局部跳跃
- 0.40–0.69：松散，重复或断裂明显
- 0.00–0.39：阅读困难，结构失控

**7. writing_evidence_usage（证据使用/RAG）**：关键论断是否有分析或检索证据支撑（结合「RAG 检索摘录」判断；无日志时保守给分，勿编造检索）。
- 0.90–1.00：关键论断多有依据
- 0.70–0.89：多数有依据，少量泛化
- 0.40–0.69：证据零散，空泛总结多
- 0.00–0.39：几乎无证据支撑

**8. writing_style（表达/学术规范）**：术语、客观性、口语与空话（权重在总分中较低，勿因文采过度加分）。
- 0.90–1.00：术语准确，克制、专业
- 0.70–0.89：基本专业，偶有冗余
- 0.40–0.69：一般，规范性不足
- 0.00–0.39：口语化或逻辑混乱

## 分析材料（JSON/正文摘录，用于分析四维 + 写作忠实性对照）
{analyse_material}

## 章节正文摘录（用于写作四维）
{sections_text}

## RAG 检索摘录（用于写作 evidence 维度；可能为空）
{rag_excerpt}

{fmt}
"""


def _pick_rag_excerpt(run: Run) -> str:
    out = run.outputs or {}
    if not isinstance(out, dict):
        return ""
    v = out.get("retrieved_context_preview")
    if v is None:
        return ""
    return str(v).strip()


def _parse_dual_head_judge(raw: str) -> tuple[float | None, dict[str, Any]]:
    """解析双阶段 JSON；返回 (八维均值分数, 解析出的字段子集)。"""
    text = _strip_llm_json_fences(raw)
    if not text:
        return None, {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if not m:
            return None, {}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, {}

    if not isinstance(obj, dict):
        return None, {}

    scores: list[float] = []
    detail: dict[str, Any] = {}
    for k in _DUAL_HEAD_SCORE_KEYS:
        if k not in obj:
            continue
        try:
            s = float(obj[k])
        except (TypeError, ValueError):
            continue
        s = max(0.0, min(1.0, s))
        scores.append(s)
        detail[k] = s
    for note_key in ("analysis_note", "writing_note"):
        if note_key in obj and obj[note_key] is not None:
            detail[note_key] = str(obj[note_key])[:220]

    if len(scores) != len(_DUAL_HEAD_SCORE_KEYS):
        return None, detail

    return sum(scores) / len(scores), detail


@traceable(run_type="llm", name="Local LLM Correctness", tags=["evaluation", "rag", "judge"])
async def local_llm_correctness(run: Run, example: Example) -> dict[str, Any]:
    """在本机用小模型按「分析 / 写作」八维量表判分；聚合均值写入 ``correctness_local``（与 LangSmith 云端 Correctness 区分）。

    数据集侧「参考答案」填入 ``analyse_material`` 作为分析对照；模型输出填入 ``sections_text``；
    ``run.outputs.retrieved_context_preview`` 填入 ``rag_excerpt``（无则空串，由 prompt 要求保守给分）。
    """
    pred = _pick_prediction(run)
    ref = _pick_reference(example)
    if not ref:
        return {"key": "correctness_local", "score": None, "comment": "无参考答案"}

    max_c = max(500, config.get_int("observability.langsmith.rag_eval_judge_max_chars", 2400))
    ref_t = ref[:max_c]
    pred_t = pred[:max_c]
    rag_t = _pick_rag_excerpt(run)[:max_c]

    judge_prompt = (
        WORKFLOW_DUAL_HEAD_JUDGE_PROMPT.replace("{analyse_material}", ref_t)
        .replace("{sections_text}", pred_t)
        .replace("{rag_excerpt}", rag_t or "（无检索摘录）")
        .replace("{fmt}", "")
    )
    try:
        chat = _get_lc_judge_chat()
        msg = await chat.ainvoke([HumanMessage(content=judge_prompt)])
        raw = getattr(msg, "content", None) or ""
        if isinstance(raw, list):
            raw = str(raw)
        raw_s = str(raw)
        score, detail = _parse_dual_head_judge(raw_s)
        comment_obj: dict[str, Any] = (
            {"parse": "incomplete_json", "detail": detail, "raw": raw_s[:800]}
            if score is None
            else {"detail": detail, "raw": raw_s[:800]}
        )
        comment_max = max(400, config.get_int("observability.langsmith.rag_eval_judge_comment_max_chars", 2000))
        comment = json.dumps(comment_obj, ensure_ascii=False)[:comment_max]
        return {"key": "correctness_local", "score": score, "comment": comment}
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
