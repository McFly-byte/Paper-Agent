"""五个主节点的职责边界与验收说明（契约文档化，供编排与面试对齐）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NodeContract(BaseModel):
    """单节点契约：责任、非职责、输入/输出期望、验收口径、失败处置原则。"""

    node_name: str
    responsibility: str = Field(description="本节点负责什么")
    out_of_scope: str = Field(description="本节点明确不负责什么")
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    acceptance_checks: list[str] = Field(
        default_factory=list,
        description="与 node_gates 中确定性检查对应的自然语言条目",
    )
    on_failure: str = Field(
        description="失败时原则：写 node error、SSE、是否阻断主流程等"
    )


NODE_CONTRACTS: dict[str, NodeContract] = {
    "search": NodeContract(
        node_name="search",
        responsibility="将用户意图转为可执行的 arXiv 检索条件（英文子式、日期），并拉取候选论文列表；触发 HITL 审核。",
        out_of_scope="不做全文阅读与结构化抽取；不做跨论文综合结论或报告终稿。",
        required_inputs=["user_request", "HITL 确认后的 SearchQuery"],
        expected_outputs=["search_gate_context（审计）", "workflow_search_results 持久化", "search_results 可清空"],
        acceptance_checks=[
            "querys 非空且子式为英文/布尔友好（无 CJK）",
            "日期若存在则格式合法",
            "检索去重后结果数 > 0 且在合理上限内",
            "记录 top 标题用于审计",
        ],
        on_failure="写入 search_node_error；SSE state=error；条件边进入 handle_error_node；原因需可解释（非笼统 search failed）。",
    ),
    "reading": NodeContract(
        node_name="reading",
        responsibility="对单篇论文做结构化信息抽取，Pydantic 校验后写入临时知识库（含可选 LlamaIndex 入库）。",
        out_of_scope="不做跨论文聚类/趋势结论；不负责写作与终稿拼装。",
        required_inputs=["search 阶段论文列表（或 tmp 中 KEY_SEARCH_RESULTS）"],
        expected_outputs=["workflow_extracted_data", "向量库 documents/metadatas"],
        acceptance_checks=[
            "解析成功篇数 / 输入篇数 ≥ 配置阈值",
            "关键字段非全空（core_problem、methodology、main_results、contributions）",
            "落库前统一 model_validate",
            "输出 metrics：解析率、字段完整率、入库篇数",
        ],
        on_failure="写入 reading_node_error 并 SSE；主流程在 orchestrator 侧终止。",
    ),
    "analyse": NodeContract(
        node_name="analyse",
        responsibility="跨论文组织：聚类主题、按簇深度分析、全局整合分析（JSON + global_analyse 正文）。",
        out_of_scope="不负责章节级长文写作与 Markdown 终稿流式交付。",
        required_inputs=["ExtractedPapersData"],
        expected_outputs=["workflow_analyse_results（文本/JSON）"],
        acceptance_checks=[
            "聚类/簇摘要非空；每簇有关键词与主题描述",
            "深度分析条目数与簇数量一致",
            "global_analyse 非空且覆盖若干预期模块关键词",
            "metrics：cluster_count、deep_count、global_len",
        ],
        on_failure="写入 analyse_node_error；子流程已通过 SSE 推送错误时同样落 error；进入 handle_error_node。",
    ),
    "writing": NodeContract(
        node_name="writing",
        responsibility="基于全局分析生成大纲、拆分 section，并行完成章节写作（含检索与审查闭环）。",
        out_of_scope="不负责最终整篇 Markdown 拼装与流式报告交付。",
        required_inputs=["global_analysis 文本", "user_request", "tmp_db_id 供 RAG"],
        expected_outputs=["sections 计划", "writted_sections", "workflow_writted_sections", "rag_retrieval_logs"],
        acceptance_checks=[
            "sections 非空",
            "已写章节数与计划匹配度 ≥ 阈值；每节最小长度",
            "review_agent 产出结构化 verdict（pass/revise/fail）并计入通过率",
            "metrics：planned_sections、written_sections、review_pass_rate",
        ],
        on_failure="写入 writing_node_error；SSE 章节级 error 已由子图推送；主流程终止。",
    ),
    "report": NodeContract(
        node_name="report",
        responsibility="将已写章节整合为一份连贯的 Markdown 调研报告（润色与过渡，不发明新结论）。",
        out_of_scope="不再引入新的研究论断或替代 analyse/writing 的结论生产。",
        required_inputs=["workflow_writted_sections 或等价章节列表"],
        expected_outputs=["report_markdown", "可选历史落库"],
        acceptance_checks=[
            "report_markdown 非空且含基本 Markdown 结构",
            "覆盖已写章节（标题/关键词重叠率）",
            "长度 ≥ 最低阈值；metrics：final_length、section_coverage、has_intro/has_conclusion 启发式",
        ],
        on_failure="写入 report_node_error；SSE；进入 handle_error_node。",
    ),
}


def get_node_contract(name: str) -> NodeContract | None:
    return NODE_CONTRACTS.get(name)
