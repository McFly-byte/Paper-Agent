

# Paper-Agent：智能学术调研报告生成系统

语言： [English](../README.md) · 简体中文（与根目录 [README.md](../README.md) 同步维护）

[License](../LICENSE)
[Python](https://www.python.org/)

## 简介

**Paper-Agent** 面向科研人员，将「结构化检索 → 并行阅读与抽取 → 多阶段分析 → RAG 辅助写作 → Markdown 报告 → 前端 SSE 进度」串成一条流水线。后端为 **FastAPI**，工作流由 **LangGraph** 编排；分析与写作子步骤中采用 **AutoGen** 风格的多智能体群聊。

更完整的架构说明见仓库根目录 **[design.md](../design.md)**。阶段性重构说明见 **[refactor/](refactor/)** 目录。

## 项目预览

点击放大截图


| 截图 1 | 截图 2 | 截图 3 |
| ---- | ---- | ---- |
|      |      |      |
| 截图 4 | 截图 5 | 截图 6 |
|      |      |      |


## 核心特性

- **LangGraph 工作流**：协调器 → 背景调研 → 规划器 → 计划审核 → arXiv 检索（与计划联动）→ 去重/过滤 → 并行阅读（PDF/文本）→ **证据索引** → 聚类 + 深度 + 全局分析 → 并行写作子会话 → **忠实度审查** → 终稿报告；另有可选的 **工作流恢复** 与终端 **错误** 节点。
- **人机协同**：`WebUserProxyAgent` 处理审核类输入；接口按 **run** 划分（`POST /api/research/runs/{run_id}/input`）。旧版 `POST /send_input` 已废弃，返回 **410**。
- **RAG**：写作阶段默认通过 **LlamaIndex** 基于 Chroma 检索（HyDE、可选重排/多查询、元数据过滤等），见 `[../src/core/system_params.yaml](../src/core/system_params.yaml)` 与 `[.env.example](../.env.example)` 中的 `RAG_BACKEND`。部分路径仍兼容旧版 Chroma 工具。
- **知识库 HTTP API**：在 `**/knowledge/...`** 下提供库管理与文档入库（`[../src/knowledge/knowledge_router.py](../src/knowledge/knowledge_router.py)`）。
- **报告历史 API**：在 `**/api/reports/...`** 下列表/查看/删除已保存报告（`[../src/api/reports_router.py](../src/api/reports_router.py)`）。
- **可观测与评估**：对接 LangSmith 的追踪习惯；在配置齐全时，流程结束后可通过 SSE 推送 **评估** 结果（`[../src/evaluation/](../src/evaluation/)`）。
- **并发与韧性**：可配置远端 LLM 限流、重试与图级恢复（`[../src/core/workflow_recovery.py](../src/core/workflow_recovery.py)`、`system_params.yaml`）。

## 系统架构

### LangGraph 主路径节点


| 顺序  | 节点                              | 职责                                              |
| --- | ------------------------------- | ----------------------------------------------- |
| 1   | `coordinator_node`              | 澄清意图，为下游准备上下文。                                  |
| 2   | `background_investigation_node` | 轻量网络/背景信息收集，辅助规划。                               |
| 3   | `planner_node`                  | 产出结构化调研/检索计划。                                   |
| 4   | `plan_review_node`              | 校验或修订计划（可含人工输入）。                                |
| 5   | `search_node`                   | 按计划执行检索（如 arXiv，`search_plan_adapter`）。         |
| 6   | `paper_filter_node`             | 去重/相关性过滤，减轻后续阅读压力。                              |
| 7   | `reading_node`                  | 并行从论文中抽取结构化字段。                                  |
| 8   | `evidence_index_node`           | 构建证据片段/索引，供分析与写作使用。                             |
| 9   | `analyse_node`                  | 聚类 → 单簇深度分析 → 全局综合（子智能体在 `sub_analyse_agent/`）。 |
| 10  | `writing_node`                  | 写作总监 + 并行子写手 + 检索 + 审查（`sub_writing_agent/`）。   |
| 11  | `faithfulness_review_node`      | 汇总前做 groundedness/引用向检查。                        |
| 12  | `report_node`                   | 组装 Markdown，经运行队列推送进度。                          |


辅助节点：`workflow_recovery_node`（重试路由）、`handle_error_node`（失败终态 + SSE）。当前图使用进程内 **内存 checkpoint**（`InMemorySaver`）。

### 子智能体模块

- **分析**（`[../src/agents/sub_analyse_agent/](../src/agents/sub_analyse_agent/)`）：嵌入 + sklearn **KMeans** 聚类、单簇深度分析、全局「六大块」式综述。
- **写作**（`[../src/agents/sub_writing_agent/](../src/agents/sub_writing_agent/)`）：写作总监、并行写作组、检索智能体、审查智能体、共享写作状态模型。

### HTTP 接口（`[../main.py](../main.py)`）


| 方法   | 路径                                   | 说明                                                                      |
| ---- | ------------------------------------ | ----------------------------------------------------------------------- |
| POST | `/api/research/runs`                 | 启动一次运行；请求体含 `query`，可选 `kb_label`、`user_id`；返回 `run_id`（亦作 `trace_id`）。 |
| GET  | `/api/research/runs/{run_id}/stream` | **SSE**，推送 `BackToFrontData` 的 JSON，直至 `finished`。                      |
| POST | `/api/research/runs/{run_id}/input`  | 向该次运行的用户代理提交人机协同文本。                                                     |


路由前缀：`**/knowledge/...`**、`**/api/reports/...**`。默认 ASGI 监听 `**0.0.0.0:8001**`。

## 端到端流程（用户视角）

1. 用户提交自然语言调研需求（可选知识库标签 `kb_label`）。
2. 协调器与背景/规划阶段形成 **可审核的计划**。
3. **检索** 与 **过滤** 降噪；**阅读** 产出结构化证据。
4. **证据索引** 支撑 **分析**（聚类 + 深度 + 全局）。
5. **写作** 并行完成各章并走 **RAG**；**忠实度** 关口抑制无依据表述。
6. **报告** 合并为 Markdown；SSE 展示进度；末尾可选 **评估** 事件。

## 仓库目录（精简）

```text
Paper-Agent/
├── main.py                      # FastAPI：调研运行、SSE、路由挂载
├── pyproject.toml               # Poetry 依赖（package-mode = false）
├── design.md                    # 系统设计（主架构文档）
├── docs/                        # 补充文档（refactor、面试题、本中文 README）
├── src/
│   ├── agents/                  # LangGraph 各节点、检索/阅读/写作/报告、规划、调研、审查
│   ├── api/                     # reports 等 HTTP 模块
│   ├── core/                    # config、models.yaml、system_params.yaml、状态、LLM 基础设施、恢复策略
│   ├── knowledge/               # 知识库工厂、Chroma 实现、索引、路由
│   ├── rag/                     # LlamaIndex RAG（检索器、重排、入库等）
│   ├── services/                # arXiv、Chroma、运行临时态、报告历史等
│   ├── domain/                  # 论文领域：过滤、证据、忠实度、引用
│   ├── evaluation/              # 运行后评估与 LangSmith 辅助
│   ├── plugins/                 # OCR、护栏等可选路径
│   ├── tasks/                   # 论文检索任务辅助
│   └── utils/                   # 日志、限流、工具函数
├── web/                         # Vue 3 + Vite 5（将 /api、/knowledge 代理到 :8001）
├── test/                        # pytest（工作流、RAG、规划、忠实度等）
├── data/                        # 运行时 KB / Chroma 路径（见 system_params 中 SAVE_DIR）
└── output/log/                  # 常见日志目录（见 logging 配置）
```

## 快速开始

1. **环境**：Python **3.12**（上限见 `pyproject.toml`）、[Poetry](https://python-poetry.org/)、前端需 Node.js。
2. **安装后端**：`poetry install`
3. **配置**：
  - 将 `**[.env.example](../.env.example)`** 复制为 `**.env**`，至少配置一家 LLM 服务商密钥（如 `SILICONFLOW_API_KEY`，其余见 `[../src/core/models.yaml](../src/core/models.yaml)` 与 `.env.example` 注释）。
  - 按需修改 `**[../src/core/models.yaml](../src/core/models.yaml)**`（厂商、`llm-routing`、各 client 类型模型）。
  - 按需修改 `**[../src/core/system_params.yaml](../src/core/system_params.yaml)**`（RAG 后端、并发、限流、日志路径等）。
4. **启动 API**：`poetry run python main.py` → **[http://0.0.0.0:8001](http://0.0.0.0:8001)**
5. **启动前端**（另一终端）：`cd web && npm install && npm run dev` → **[http://localhost:5173](http://localhost:5173)**（Vite 将 `/api`、`/knowledge` 代理到 8001）。

## 配置说明

- **环境变量（`.env`）**：API 密钥、可选 `DEFAULT_LLM_PROVIDER`、LangSmith（`LANGCHAIN_API_KEY` / `LANGSMITH_*`）、`RAG_BACKEND`、DashScope / 火山等。变量名以 `**[.env.example](../.env.example)`** 为准。
- `**src/core/models.yaml**`：厂商注册、嵌入模型、各模块客户端、**llm-routing** 与限流桶名。
- `**src/core/system_params.yaml`**：路径（`SAVE_DIR`）、`KB_TYPE`、**rag** 段（LlamaIndex 开关、HyDE、重排、分块）、并发、日志、恢复相关默认项。

请勿将真实密钥提交到 Git；`.env` 已被忽略。

## 技术栈


| 层次       | 技术                                                                                   |
| -------- | ------------------------------------------------------------------------------------ |
| 运行时      | Python 3.12、Poetry                                                                   |
| API      | FastAPI、Uvicorn、sse-starlette                                                        |
| 编排       | LangGraph、LangSmith（可选追踪/评估）                                                         |
| 智能体      | pyautogen、autogen-agentchat、autogen-ext                                              |
| LLM 调用   | LangChain 对话集成（`langchain-openai`、`langchain-community`），厂商路由在仓库内实现                  |
| 检索与 RAG  | arXiv、ChromaDB、**LlamaIndex**（`llama-index-*`）                                       |
| NLP / ML | scikit-learn（KMeans、肘部法则）、pandas、sentence-transformers（可选 extra `rag-cross-encoder`） |
| 文档       | PyMuPDF、python-docx、markdownify、beautifulsoup4                                       |
| 前端       | Vue 3.4、Vue Router 4、Vite 5、axios、marked                                             |


## 参与贡献

欢迎通过 Issue 与 Pull Request 贡献代码。请勿提交密钥；较大行为变更请与 `**design.md`** 或 `**docs/refactor/**` 中的设计叙述保持一致，便于评审。

## 许可证

MIT — 见 [LICENSE](../LICENSE)。

## 联系方式

- **GitHub Issues**：报告缺陷与功能建议的首选渠道  
- 项目主页：[https://github.com/Tswoen/paper-agent](https://github.com/Tswoen/paper-agent)

---

⭐ 若本项目对你有帮助，欢迎在仓库上点 Star 以便他人发现。

## Star 历史

[Star History Chart](https://www.star-history.com/?repos=Tswoen%2FPaper-Agent&type=date&logscale=&legend=top-left)