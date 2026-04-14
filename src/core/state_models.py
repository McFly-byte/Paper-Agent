from asyncio import Queue
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from enum import Enum


class BackToFrontData(BaseModel):
    step: str
    state: str
    data: Any


class ExecutionState(str, Enum):
    """工作流执行状态枚举"""
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    READING = "reading"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    WRITING_DIRECTOR = "writing_director"
    SECTION_WRITING = "section_writing"
    WRITING = "writing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"
    FINISHED = "finished"

class KeyMethodology(BaseModel):
    name: str  # 方法名称（如“Transformer-based Sentiment Classifier”）
    principle: str  # 核心原理
    novelty: str  # 创新点（如“首次引入领域自适应预训练”）


class ExtractedPaperData(BaseModel):
    paper_id: str  # 关联搜索结果的paper_id
    core_problem: str
    key_methodology: KeyMethodology
    datasets_used: List[str]  # 如["IMDB Dataset (50k reviews)", "SST-2"]
    evaluation_metrics: List[str]
    main_results: str  # 含关键数值，如“在IMDB上Accuracy达92.5%，优于BERT的89.3%”
    limitations: str
    contributions: List[str]
    # author_institutions: Optional[str]  # 如“Stanford University, Department of CS”
    # extract_source: dict  # 溯源：记录每个维度的提取章节，如{"core_problem": "Abstract, Introduction"}

# 创建一个新的Pydantic模型来包装列表
class ExtractedPapersData(BaseModel):
    papers: List[ExtractedPaperData]

class AnalysisResults(BaseModel):
    """分析模块产生的结构化结果"""
    topic_clusters: Optional[Dict[str, List[str]]] = Field(default=None, description="主题聚类, key: 主题名, value: 相关paper_id列表")
    trend_analysis: Optional[Dict[int, int]] = Field(default=None, description="趋势分析, key: 年份, value: 论文数量")
    method_comparison: Optional[List[Dict[str, Any]]] = Field(default=None, description="方法对比表格数据")
    influential_authors: Optional[List[str]] = Field(default=None, description="高产作者列表")
    influential_institutions: Optional[List[str]] = Field(default=None, description="核心机构列表")

class NodeError(BaseModel):
    search_node_error: Optional[str] = Field(default=None, description="搜索节点错误信息")
    reading_node_error: Optional[str] = Field(default=None, description="阅读节点错误信息")
    analyse_node_error: Optional[str] = Field(default=None, description="分析节点错误信息")
    writing_node_error: Optional[str] = Field(default=None, description="写作节点错误信息")
    report_node_error: Optional[str] = Field(default=None, description="报告生成节点错误信息")
    error: Optional[str] = Field(default=None, description="错误信息")

class PaperAgentState(BaseModel):
    """LangGraph工作流的全局状态对象"""
    run_id: str = Field(description="本次调研执行 ID，与 SSE / HITL / 日志对齐")
    user_id: Optional[str] = Field(default=None, description="可选：调用方用户标识")
    # 用户输入
    frontend_data: Optional[BackToFrontData] = Field(default=None, description="前端展示数据")
    agent_logs: Dict[str, str] = Field(default_factory=dict, description="各智能体执行日志，key为智能体名称")
    user_request: str = Field(description="用户的原始输入请求")
    max_papers: int = Field(default=50, description="最大论文数量")
    
    # 执行状态
    current_step: ExecutionState = Field(default=ExecutionState.INITIALIZING, description="当前执行步骤")
    error: Optional[NodeError] = Field(default=None, description="错误信息")
    
    # 数据流
    # search_results: List[PaperMetadata] = Field(default_factory=list, description="检索到的论文元数据列表")
    search_results: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="检索结果；节点间可清空，完整列表在临时向量库 workflow_search_results")
    paper_contents: Optional[Dict[str, str]] = Field(default_factory=dict, description="解析后的论文全文字典（大字段可外置向量库，勿在 state 长期堆积全文）")
    extracted_data: Optional[ExtractedPapersData] = Field(default=None, description="结构化抽取；阅读节点后可清空，完整数据在 workflow_extracted_data")
    analyse_results: Optional[str] = Field(default=None, description="分析结果 JSON 字符串；分析节点后可清空，正文在 workflow_analyse_results")
    outline: Optional[str] = Field(default=None, description="报告大纲")
    writted_sections: Optional[List[str]] = Field(default=None, description="已写章节；写作节点后可清空，列表在 workflow_writted_sections")
    report_markdown: Optional[str] = Field(default=None, description="最终生成的Markdown报告内容")
    
    # 配置与上下文
    llm_provider: Any = Field(default=None, description="LLM提供者实例", exclude=True)  # 排除序列化
    config: Dict[str, Any] = Field(default_factory=dict, description="运行时配置")
    boundary_checks: Dict[str, Any] = Field(
        default_factory=dict,
        description="五主节点确定性门禁结果（node_gates），与 LangSmith quality_eval 分层",
    )
    rag_retrieval_logs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="写作阶段 RAG 调用摘要，供 LangSmith 评估与 backend 对比",
    )

@dataclass
class PaperRunContext:
    """每 run 注入的运行时依赖（SSE 队列、HITL 代理），不参与 LangGraph checkpoint 序列化。"""

    state_queue: Queue
    user_proxy: Any


class State(TypedDict):
    """LangGraph 可 checkpoint 的状态：仅持久化业务快照；队列与代理见 PaperRunContext。"""

    value: PaperAgentState