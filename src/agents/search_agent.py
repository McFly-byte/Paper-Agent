from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import re
import ast

from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearcher
from langgraph.runtime import Runtime

from src.core.state_models import State, ExecutionState, PaperRunContext
from src.core.prompts import search_agent_prompt
from src.core.state_models import BackToFrontData
from src.services.run_tmp_state_store import ensure_run_tmp_kb, put_json, KEY_SEARCH_RESULTS

from src.core.model_client import create_search_model_client
from src.core.node_gates import gate_search, record_gate
from src.core.workflow_recovery import (
    clear_recovery_feedback_for,
    format_recovery_user_block,
    set_recovery_target,
)

logger = setup_logger(__name__)

# LLM 路由键：search-model（见 models.yaml llm-routing.client_policies；DEFAULT_LLM_PROVIDER 可切百炼）
model_client = create_search_model_client()

# 创建一个查询条件类，包括查询内容、主题、时间范围等信息，用于存储用户的查询需求
class SearchQuery(BaseModel):
    """查询条件类，存储用户查询需求"""

    querys: List[str] = Field(default_factory=list, description="arXiv 英文布尔子式列表（程序会包 all:）")
    start_date: Optional[str] = Field(default=None, description="开始时间, 格式: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束时间, 格式: YYYY-MM-DD")

    @field_validator("querys", mode="before")
    @classmethod
    def _coerce_querys(cls, v):
        if v is None:
            return []
        return v


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _english_fallback_querys(user_request: str) -> List[str]:
    """从用户原文中提取英文词/短语，拼成一条安全子式（arXiv 英文索引几乎不含中文）。"""
    text = user_request or ""
    # 连续拉丁词（含连字符），长度>=3
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
    seen: set[str] = set()
    uniq: List[str] = []
    for raw in tokens:
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(raw)
    if len(uniq) >= 2:
        inner = " OR ".join(f'"{t}"' if " " in t or "-" in t else t for t in uniq[:8])
        return [f"({inner})"]
    if len(uniq) == 1:
        t0 = uniq[0]
        return [f'("{t0}")' if len(t0) <= 40 else f"({t0})"]
    return ['("diffusion" OR "super-resolution" OR "image restoration")']


def sanitize_arxiv_querys(querys: List[str], user_request: str) -> List[str]:
    """
    去掉含中文或占位语的子查询；若全部被丢弃则用英文回退，避免 export.arxiv.org 收到无效 query。
    """
    cleaned: List[str] = []
    for q in querys or []:
        s = (q or "").strip()
        if not s:
            continue
        if _contains_cjk(s):
            logger.warning(
                "[工作流·检索] 丢弃含非英文（CJK）的子查询（arXiv 索引不适用）: %s",
                s[:120] + ("…" if len(s) > 120 else ""),
            )
            continue
        cleaned.append(s)
    if cleaned:
        return cleaned
    fb = _english_fallback_querys(user_request)
    logger.warning(
        "[工作流·检索] 有效子查询为空，已用用户原文中的英文词回退生成 %s 条: %s",
        len(fb),
        fb,
    )
    return fb

search_agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    system_message=search_agent_prompt,
    output_content_type=SearchQuery
)

def parse_search_query(s: str) -> SearchQuery:
    """将前端传回的字符串转为 SearchQuery 对象"""
    # 提取 querys（使用 ast.literal_eval 保证安全）
    querys_match = re.search(r"querys\s*=\s*(\[[^\]]*\])", s)
    start_match = re.search(r"start_date\s*=\s*'([^']*)'", s)
    end_match = re.search(r"end_date\s*=\s*'([^']*)'", s)

    querys = []
    if querys_match:
        try:
            querys = ast.literal_eval(querys_match.group(1))
        except Exception:
            querys = []

    start_date = start_match.group(1) if start_match else None
    end_date = end_match.group(1) if end_match else None

    return SearchQuery(querys=querys, start_date=start_date, end_date=end_date)

async def search_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    """搜索论文节点"""
    try:
        state_queue = runtime.context.state_queue
        current_state = state["value"]
        current_state.current_step = ExecutionState.SEARCHING
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="initializing",data=None)) # 将初始状态推送到队列

        _rec = format_recovery_user_block(current_state.config or {}, "search")
        prompt = f"""
        请根据用户查询需求，生成检索查询条件；按系统说明输出 **json** 结构化结果（querys / start_date / end_date）。
        用户查询需求：{current_state.user_request}
        {_rec}
        """
        logger.info("[工作流·检索] 调用 LLM 生成 arXiv 检索条件（可能需数十秒）…")
        # 调用search_agent生成查询条件
        response = await search_agent.run(task = prompt) 
        search_query = response.messages[-1].content
        # 将查询条件推送到队列，等待前端人工审核
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="user_review",data=f"{search_query}"))
        logger.info("[工作流·检索] 已推送查询条件到前端，等待人工确认/修改（未确认前会停在这里）…")
        # 等待前端人工审核
        user_proxy = runtime.context.user_proxy
        result = await user_proxy.on_messages(
            [TextMessage(content="请人工审核：查询条件是否符合？", source="AI")],
            cancellation_token=CancellationToken()
        )
        search_query = parse_search_query(result.content)

        safe_querys = sanitize_arxiv_querys(search_query.querys, current_state.user_request or "")

        # 调用检索服务
        paper_searcher = PaperSearcher()
        logger.info("[工作流·检索] 开始 arXiv 检索论文（网络请求，条数上限与查询复杂度影响耗时）…")
        # 调用检索服务搜索论文
        results = await paper_searcher.search_papers(
            querys=safe_querys,
            max_results=current_state.max_papers,
            start_date=search_query.start_date,
            end_date=search_query.end_date,
        )
        current_state.search_results = results
        raw_n = len(results)
        dedup_keys: list[str] = []
        for i, p in enumerate(results):
            pid = (p or {}).get("paper_id") or (p or {}).get("id") or f"row-{i}"
            dedup_keys.append(str(pid))
        dedup_n = len(set(dedup_keys)) if dedup_keys else 0
        top_titles = [str((p or {}).get("title") or "")[:160] for p in results[:5]]
        top_abstracts = [str((p or {}).get("summary") or (p or {}).get("abstract") or "")[:500] for p in results[:5]]
        gate_ctx = {
            "querys": safe_querys,
            "start_date": search_query.start_date,
            "end_date": search_query.end_date,
            "raw_result_count": raw_n,
            "dedup_count": dedup_n,
            "top_titles": top_titles,
            "top_abstracts": top_abstracts,
            "user_request": current_state.user_request,
            "requested_max_results": current_state.max_papers,
        }
        current_state.config["search_gate_context"] = gate_ctx
        sg = gate_search(gate_ctx)
        current_state.boundary_checks = record_gate(current_state.boundary_checks, "search", sg)
        if not sg.passed:
            detail = "；".join(sg.reasons) if sg.reasons else "检索门禁未通过"
            if not current_state.error.search_node_error:
                current_state.error.search_node_error = detail
            set_recovery_target(current_state, "search")
            await state_queue.put(
                BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=detail)
            )
            current_state.search_results = results
            return {"value": current_state}

        await ensure_run_tmp_kb(current_state)
        await put_json(current_state, KEY_SEARCH_RESULTS, results)
        current_state.search_results = []
        clear_recovery_feedback_for(current_state, "search")
        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.SEARCHING,
                state="completed",
                data=f"论文搜索完成，共找到 {len(results)} 篇论文",
            )
        )
        return {"value": current_state}
            
    except Exception as e:
        err_msg = f"检索节点异常：{type(e).__name__}: {str(e)}"
        vs = state["value"]
        vs.error.search_node_error = err_msg
        set_recovery_target(vs, "search")
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="error",data=err_msg))
        return {"value": vs}