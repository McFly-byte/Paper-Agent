from autogen_agentchat.agents import AssistantAgent
# from pydantic import BaseModel, Field
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional,Dict,Any
from src.utils.log_utils import setup_logger
from src.core.prompts import reading_agent_prompt
from src.core.model_client import create_default_client, create_reading_model_client
from src.core.state_models import BackToFrontData
from langgraph.runtime import Runtime

from src.core.state_models import State, ExecutionState, PaperAgentState, PaperRunContext
from src.services.chroma_client import ChromaClient
from src.knowledge.knowledge import knowledge_base
from src.core.config import config
from src.rag.llamaindex.config import llamaindex_ingestion_enabled
from src.utils.llm_api_throttle import get_throttler_for_client_type, reading_throttle_token_estimate
from src.services.run_tmp_state_store import (
    ensure_run_tmp_kb,
    get_json,
    put_json,
    KEY_SEARCH_RESULTS,
    KEY_EXTRACTED_DATA,
)
from src.core.node_gates import gate_reading, record_gate
from src.core.workflow_recovery import (
    clear_recovery_feedback_for,
    format_recovery_user_block,
    set_recovery_target,
)
from openai import RateLimitError
from httpx import ReadTimeout
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_attempt, before_sleep_log
import re, json, ast
import asyncio
import logging

logger = setup_logger(__name__)

class KeyMethodology(BaseModel):
    name: Optional[str] = Field(default=None, description="方法名称（如“Transformer-based Sentiment Classifier”）")
    principle: Optional[str] = Field(default=None, description="核心原理")
    novelty: Optional[str] = Field(default=None, description="创新点（如“首次引入领域自适应预训练”）")


class ExtractedPaperData(BaseModel):
    """单篇结构化抽取；忽略模型多带的 paper_id 等字段。"""
    model_config = ConfigDict(extra="ignore")

    # paper_id: str = Field(default=None, description="论文ID")
    core_problem: str = Field(default=None, description="核心问题")
    key_methodology: KeyMethodology = Field(default=None, description="关键方法")
    datasets_used: List[str] = Field(default=[], description="使用的数据集")
    evaluation_metrics: List[str] = Field(default=[], description="评估指标")
    main_results: str = Field(default="", description="主要结果")
    limitations: str = Field(default="", description="局限性")
    contributions: List[str] = Field(default=[], description="贡献")
    # author_institutions: Optional[str]  # 如“Stanford University, Department of CS”
    
    # 清理空字符串和列表，统一格式
    @field_validator("datasets_used", "evaluation_metrics", "contributions", mode="before")
    @classmethod
    def _validate_list_fields(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("core_problem", "main_results", "limitations", mode="before")
    @classmethod
    def _validate_str_fields(cls, v):
        if v is None:
            return ""
        return str(v)

# 创建一个新的Pydantic模型来包装列表
class ExtractedPapersData(BaseModel):
    papers: List[ExtractedPaperData] = Field(default=[], description="提取的论文数据列表")

model_client = create_reading_model_client()

read_agent = AssistantAgent(
    name="read_agent",
    model_client=model_client,
    system_message=reading_agent_prompt,
    # 不用 output_content_type：部分模型会返回根级 JSON 数组或 papers 包装，与单对象 schema
    # 冲突会在 OpenAI 流式解析阶段直接抛错；改由下方 reading_node 统一解包 + model_validate。
    model_client_stream=True,
)

def sanitize_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    """把论文Dict清洗、标准化，防止存入Chroma时不稳定"""
    new_meta = {}
    for k, v in paper.items():
        if v is None: # None 值直接丢弃
            continue
        if isinstance(v, list): # list 转成逗号拼接字符串
            new_meta[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, dict): # dict 转成 JSON 字符串
            new_meta[k] = json.dumps(v, ensure_ascii=False)
        else:
            new_meta[k] = v
    return new_meta


async def add_papers_to_kb(
    papers: Optional[List[Dict[str, Any]]],
    extracted_papers: ExtractedPapersData,
    current_state: PaperAgentState,
) -> None:
    """将提取的论文数据添加到知识库；依赖 ensure_run_tmp_kb 已写入的 tmp_db_id。"""
    await ensure_run_tmp_kb(current_state)

    # 把论文数据转成JSON字符串
    documents=[json.dumps(paper.model_dump(),ensure_ascii=False) for paper in extracted_papers.papers]
    # 把论文数据转成适合存入Chroma的结构
    sanitized_metadatas = []
    if papers:
        for paper in papers:
           # new_meta = {}
           # for k, v in paper.items():
            #     if isinstance(v, list):
            #         new_meta[k] = ", ".join(str(x) for x in v)
            #     else:
            #         new_meta[k] = v
            # sanitized_metadatas.append(new_meta)
            sanitized_metadatas.append(sanitize_metadata(paper))          
    metadatas = sanitized_metadatas
    
    ids = [str(i) for i in range(len(documents))] 
    
    data = {
        "documents": documents, # 提炼后的论文信息，用于embedding后相似度检索
        "metadatas": metadatas, # 元数据，用于检索时过滤
        "ids": ids,
    }

    db_id = (current_state.config or {}).get("tmp_db_id")
    if not db_id:
        raise ValueError("add_papers_to_kb: 缺少 tmp_db_id")
    await knowledge_base.add_processed_content(db_id, data)


async def reading_node(state: State, runtime: Runtime[PaperRunContext]) -> State:
    """阅读论文节点"""
    state_queue = runtime.context.state_queue
    current_state = state["value"]
    current_state.current_step = ExecutionState.READING
    # 将初始状态推送到队列
    await state_queue.put(BackToFrontData(step=ExecutionState.READING,state="initializing",data=None))

    await ensure_run_tmp_kb(current_state)
    if not current_state.search_results:
        loaded = await get_json(current_state, KEY_SEARCH_RESULTS)
        if loaded is not None:
            current_state.search_results = loaded

    papers = list(current_state.search_results or [])
    input_paper_count = len(papers)
    _reading_recovery = format_recovery_user_block(current_state.config or {}, "reading")

    read_sem = config.get_int("REALTIME_HIGH_PRIORITY_MAX_CONCURRENCY", 0)
    if read_sem <= 0:
        read_sem = config.get_int("llm_concurrency_pools.realtime_high_priority_max", 0)
    if read_sem <= 0:
        read_sem = config.get_int("llm_remote_rate_limit.max_concurrent_reading_tasks", 0)
    if read_sem <= 0:
        read_sem = config.get_int("llm_remote_rate_limit.max_concurrent_llm_tasks", 2)
    sem_n = max(1, read_sem)
    semaphore = asyncio.Semaphore(sem_n)
    llm_throttle = get_throttler_for_client_type("reading-model")

    def _retryable_llm_error(exc: BaseException) -> bool:
        return isinstance(exc, (RateLimitError, ReadTimeout))

    _read_retry_max = max(60, config.get_int("llm_remote_rate_limit.reading_retry_wait_max_seconds", 240))
    _read_retry_attempts = max(3, config.get_int("llm_remote_rate_limit.reading_retry_stop_attempts", 12))

    @retry(
        retry=retry_if_exception(_retryable_llm_error),
        wait=wait_exponential(multiplier=2, min=8, max=_read_retry_max),
        stop=stop_after_attempt(_read_retry_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def read_single_paper(paper):
        body = str(paper)
        if _reading_recovery:
            body = _reading_recovery + "\n\n" + body
        return await read_agent.run(task=body)

    n = len(papers)

    async def read_with_limit(paper, index: int):
        async with semaphore:
            logger.info(
                "[工作流·阅读] 进行中：第 %s/%s 篇（并发槽已占用）",
                index + 1,
                n,
            )
            try:
                if llm_throttle:
                    est = reading_throttle_token_estimate(str(paper))
                    await llm_throttle.acquire(1, est)
                result = await read_single_paper(paper)
                logger.info("[工作流·阅读] 完成：第 %s/%s 篇", index + 1, n)
                return result
            except Exception:
                logger.exception("[工作流·阅读] 失败：第 %s/%s 篇", index + 1, n)
                raise

    logger.info(
        "[工作流·阅读] 开始对 %s 篇论文做 LLM 结构化提取（并发上限 %s，整体可能很慢）…",
        n,
        sem_n,
    )
    # 并行阅读多篇论文
    results = await asyncio.gather(
        *[read_with_limit(paper, i) for i, paper in enumerate(papers)]
    )

    # 合并结果
    extracted_papers = ExtractedPapersData()
    # for result in results:
    #     if result.messages[-1].content:
    #         parsed_paper = result.messages[-1].content
    #         extracted_papers.papers.append(parsed_paper)   
    
    # 清洗、预处理获取的数据    
    successful_papers = [] 
    for i, result in enumerate(results):
        raw_content = result.messages[-1].content
        # logger.info(f"Reading Agent Raw Output: {raw_content}") # 打印原始输出
        
        if isinstance(raw_content, ExtractedPaperData):
            extracted_papers.papers.append(raw_content)
            successful_papers.append(papers[i])
            continue
        if isinstance(raw_content, dict):
            data = raw_content
        elif isinstance(raw_content, str):
            clean_content = raw_content.strip()
            if clean_content.startswith("```"):
                clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
                clean_content = re.sub(r"\s*```$", "", clean_content)
            try:
                data = json.loads(clean_content)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(clean_content)
                except Exception:
                    logger.error(f"Failed to parse content as JSON or Python dict: {clean_content}")
                    continue
        else:
            logger.error(f"Unsupported content type: {type(raw_content)}")
            continue

        # 清理 Markdown 代码块
        # 3. 数据结构修正（处理列表包裹或 {"papers": ...} 包裹）
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0] # 取第一个
            else:
                logger.warning("Parsed content is an empty list.")
                continue
        
        if isinstance(data, dict):
            # 如果被包裹在 "papers" 键中
            if "papers" in data and isinstance(data["papers"], list):
                if len(data["papers"]) > 0:
                    data = data["papers"][0]
            # 如果被包裹在 "paper" 键中
            elif "paper" in data and isinstance(data["paper"], dict):
                data = data["paper"]
        
        try:
            # 4. 验证并转换
            parsed_paper = ExtractedPaperData.model_validate(data)
            extracted_papers.papers.append(parsed_paper)
            successful_papers.append(papers[i])
        except Exception as e:
            logger.error(f"Validation failed for data: {data}. Error: {e}")
            # extracted_papers.papers.append(ExtractedPaperData()) 


     # 还得存入向量数据库中
    # await add_papers_to_kb(papers,extracted_papers)
    logger.info(
        "[工作流·阅读] 将 %s 篇成功提取的论文写入临时向量库…",
        len(extracted_papers.papers),
    )
    await add_papers_to_kb(successful_papers, extracted_papers, current_state)

    if llamaindex_ingestion_enabled():
        try:
            # 惰性导入：未安装 llama-index 时仅在关闭 LLAMAINDEX 时避免加载整个 LlamaIndex 栈
            from src.rag.llamaindex.service import get_llamaindex_rag_service

            db_id = (current_state.config or {}).get("tmp_db_id")
            if db_id and successful_papers:
                extracted_dicts = [p.model_dump() for p in extracted_papers.papers]
                n_nodes = await get_llamaindex_rag_service().ingest_extracted_papers(
                    db_id=db_id,
                    run_id=current_state.run_id,
                    paper_meta_list=successful_papers,
                    extracted_list=extracted_dicts,
                )
                logger.info(
                    "[工作流·阅读] LlamaIndex 多节点入库完成 run_id=%s nodes=%s",
                    current_state.run_id,
                    n_nodes,
                )
        except Exception as li_exc:  # noqa: BLE001
            logger.warning(
                "[工作流·阅读] LlamaIndex 入库失败（已跳过，不影响 legacy）: %s",
                li_exc,
            )

    await put_json(current_state, KEY_EXTRACTED_DATA, extracted_papers.model_dump())
    current_state.extracted_data = ExtractedPapersData(papers=[])

    rg = gate_reading(
        input_paper_count=input_paper_count,
        papers_parsed=extracted_papers.papers,
        kb_write_count=len(extracted_papers.papers),
    )
    current_state.boundary_checks = record_gate(current_state.boundary_checks, "reading", rg)
    if not rg.passed:
        detail = "；".join(rg.reasons) if rg.reasons else "阅读门禁未通过"
        current_state.error.reading_node_error = detail
        set_recovery_target(current_state, "reading")
        await state_queue.put(
            BackToFrontData(step=ExecutionState.READING, state="error", data=detail)
        )
    else:
        clear_recovery_feedback_for(current_state, "reading")
        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.READING,
                state="completed",
                data=f"论文阅读完成，共阅读 {len(extracted_papers.papers)} 篇论文",
            )
        )
    current_state.search_results = []
    return {"value": current_state}


if __name__ == "__main__":
    paper = {
        'core_problem': 'Despite the rapid introduction of autonomous vehicles, public misunderstanding and mistrust are prominent issues hindering their acceptance.'
    }
    chroma_client = ChromaClient()
    chroma_client.add_documents(
        documents=[paper],
        metadatas=[paper],
    )   
