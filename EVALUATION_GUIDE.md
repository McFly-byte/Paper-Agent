# Paper-Agent LangSmith 评估指导

基于 [LangSmith 官网教程](https://langsmith.langchain.ac.cn/observability/how_to_guides/trace_with_langgraph) 为 LLM Agent 部分添加了**丰富、必要的评估体系**。

## 1. 已实现的核心评估能力（已修复）

### 1.1 LangSmith Tracing (已增强并修复)
- 使用 `@traceable` 装饰 `PaperAgent Full Workflow` 和所有 LLM Judge 函数
- LangGraph 所有节点 (search/read/analyse/write/report) 自动纳入 trace tree
- **已移除废弃的 `wrap_openai` 导入**，改为依赖 `LANGCHAIN_TRACING_V2=true` 环境变量 + LangSmith 原生追踪
- 完全兼容本地 Ollama（qwen3.5:9b 等），追踪失败不影响主流程
- 在 LangSmith Dashboard 可看到完整的调用链、token 消耗、延迟、LLM-as-Judge 评分

### 1.2 LLM-as-Judge 评估 (核心亮点)
- **分析质量评估** (`evaluate_analysis_quality`): 技术深度、聚类一致性、洞见新颖性
- **写作质量评估** (`evaluate_writing_quality`): 连贯性、学术规范、RAG利用率、洞见深度
- **报告完整性评估**: 结构、覆盖度、实用性
- **RAG 检索评估**: 相关性、多样性、精确率、召回估算 (heuristic + LLM)
- 全部使用 `PydanticOutputParser` 实现结构化输出，便于后续自动化

### 1.3 集成点
- `orchestrator.py`: 主工作流 traceable + 结束时统一调用 `run_evaluation`
- `report_agent.py`: 报告生成后立即执行评估并通过 SSE 推送给前端
- 评估结果自动写入 `state.config.evaluation` 和历史报告元数据

## 2. 如何使用

### 步骤 1: 配置 LangSmith
1. 复制 `.env.example` 为 `.env`
2. 访问 [https://smith.langchain.com/](https://smith.langchain.com/) 注册并获取 `LANGCHAIN_API_KEY`
3. 填入 `.env`:
   ```env
   LANGCHAIN_API_KEY=lsv2_xxxxxxxx
   LANGCHAIN_PROJECT=paper-agent
   ```
4. 确保 `system_params.yaml` 中 `observability.langsmith.tracing: true`

### 步骤 2: 运行并查看
```bash
# 启动服务
poetry run uvicorn src.api.main:app --reload

# 前端发起一次完整调研任务
# 完成后会在日志中看到：
# [Evaluation][run_id=xxx] LangSmith 评估完成，整体得分: 0.78
```

### 步骤 3: 在 LangSmith Dashboard 查看
1. 打开 [LangSmith](https://smith.langchain.com/)
2. 切换到你的 `paper-agent` 项目
3. 查看 **Traces**:
   - 点击任意 run → 看到完整 LangGraph 树状结构
   - 展开 "Paper Analysis Evaluator"、"Writing Quality Evaluator" 等 LLM Judge 调用
   - 查看详细 reasoning、score、suggestions
4. 查看 **Datasets & Experiments** (后续可批量评估历史报告)

## 3. 评估指标详情 (面试/比赛亮点)

- **技术深度**：是否包含方法对比、趋势脉络、量化证据
- **洞见新颖性**：是否超出简单总结，提出跨主题关联或未来方向
- **RAG 利用率**：写作中是否有效引用检索到的论文证据
- **报告完整性**：是否覆盖聚类所有主题、是否有可操作建议
- **整体分数**：加权平均，用于报告质量排序和 A/B 测试

## 4. 未来可扩展方向 (符合架构改进要求)
- 集成 `langsmith.evaluate()` 批量离线评估历史数据集
- 添加 Human Feedback 回路 (前端点赞/修改 → 存为 LangSmith feedback)
- 支持自定义 Rubric (针对特定领域如「医疗」「自动驾驶」)
- 构建 Evaluation Leaderboard，自动选最优 prompt/模型
- 添加 LangSmith Online Evaluation (实时打分)

## 5. 代码位置
- `src/evaluation/evaluators.py`：核心评估逻辑 + traceable 装饰
- `src/agents/orchestrator.py`：主流程集成
- `src/agents/report_agent.py`：报告节点评估
- `src/core/config.py` & `system_params.yaml`：配置增强
- `src/core/prompts.py`：为 Judge 优化 prompt

**现在你的 Paper-Agent 已经具备生产级可观测性和系统化评估能力**，完全符合大厂后端/大模型应用开发实习的简历亮点要求！

如需进一步优化某个评估维度 (例如增加引用准确性检查、知识图谱一致性评估)，请告诉我具体方向。
