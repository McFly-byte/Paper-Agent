<!-- <h1 align="center">基于多智能体和工作流的大模型的调研报告生成系统</h1> -->
<h1 align="center">Paper-Agent: Intelligent Academic Survey Report Generation System</h1>

<p align="center">
  Languages:
  English ·
  <a href="./docs/README_cn.md">简体中文</a>（若与英文不一致，以本页为准）
</p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

## Introduction

**Paper-Agent** automates end-to-end academic survey reports for researchers: structured retrieval, parallel reading and extraction, multi-stage analysis, RAG-assisted writing, Markdown report output, and SSE progress to the UI. It is built as a **FastAPI** service orchestrating **LangGraph** with **AutoGen**-style multi-agent chats in the writing and analysis substeps.

For deeper architecture notes, see **[design.md](design.md)** (repository root). Supplementary refactor notes live under **[docs/refactor/](docs/refactor/)**.

## Project preview

<summary>Click to enlarge screenshots</summary>

| Screenshot 1 | Screenshot 2 | Screenshot 3 |
|-------|-------|-------|
| <img width="400" src="https://github.com/user-attachments/assets/b3617fee-ab47-4aac-9be7-0cb543fd706a" /> | <img width="400" src="https://github.com/user-attachments/assets/a27882fb-3bd8-4f44-b18f-8161bb0d44a6" /> | <img width="400" src="https://github.com/user-attachments/assets/18f2f0bc-6d2c-4b5f-a2b9-a87d16fcd6be" /> |
| Screenshot 4 | Screenshot 5 | Screenshot 6 |
| <img width="400" src="https://github.com/user-attachments/assets/21e5dc93-1c8b-46e3-b33c-f359d94cf2db" /> | <img width="400" src="https://github.com/user-attachments/assets/1e21162d-e083-40bc-93de-08302f28b08b" /> | <img width="400" src="https://github.com/user-attachments/assets/77738e3d-7d80-4d8c-9ea4-61c45e3db5d6" /> |

## Core features

- **LangGraph workflow**: Coordinator → background research → planner → plan review → arXiv search (plan-aware) → dedup/filter → parallel reading (PDF/text) → **evidence index** → clustering + deep + global analysis → parallel writing sub-chat → **faithfulness review** → final report; optional **workflow recovery** and terminal **error** node.
- **Human-in-the-loop**: `WebUserProxyAgent` resolves review prompts; API is run-scoped (`POST /api/research/runs/{run_id}/input`). Legacy `POST /send_input` returns **410**.
- **RAG**: Default writing-time retrieval uses **LlamaIndex** over Chroma (HyDE, optional rerank / multi-query, metadata filters)—see `src/core/system_params.yaml` and `RAG_BACKEND` in `.env.example`. Legacy Chroma tooling remains where still referenced.
- **Knowledge base API**: CRUD and document ingest under **`/knowledge/...`** (`src/knowledge/knowledge_router.py`).
- **Report history API**: List/get/delete saved reports under **`/api/reports/...`** (`src/api/reports_router.py`).
- **Observability & evaluation**: LangSmith-friendly tracing; post-run **evaluation** payload pushed over SSE when LangSmith / eval dependencies are configured (`src/evaluation/`).
- **Concurrency & resilience**: Configurable remote LLM rate limits, retries, and graph-level recovery (`src/core/workflow_recovery.py`, `system_params.yaml`).

## System architecture

### LangGraph nodes (main path)

| Order | Node | Role |
|------:|------|------|
| 1 | `coordinator_node` | Clarifies intent and prepares downstream context. |
| 2 | `background_investigation_node` | Lightweight web/background gathering to inform the plan. |
| 3 | `planner_node` | Produces a structured research/search plan. |
| 4 | `plan_review_node` | Validates or adjusts the plan (can involve human input). |
| 5 | `search_node` | Executes retrieval (e.g. arXiv) using the adapted plan (`search_plan_adapter`). |
| 6 | `paper_filter_node` | Deduplication / relevance filtering before heavy reading. |
| 7 | `reading_node` | Parallel extraction of structured fields from papers. |
| 8 | `evidence_index_node` | Builds evidence snippets / index for analysis and writing. |
| 9 | `analyse_node` | Cluster → per-cluster deep analysis → global synthesis (sub-agents in `sub_analyse_agent/`). |
| 10 | `writing_node` | Director + parallel sub-writers + retrieval + review (`sub_writing_agent/`). |
| 11 | `faithfulness_review_node` | Groundedness / citation-oriented check before assembly. |
| 12 | `report_node` | Assembles Markdown and streams progress via the run queue. |

Supporting nodes: `workflow_recovery_node` (retry routing), `handle_error_node` (terminal failure + SSE). The graph is compiled with an **in-memory checkpointer** (`InMemorySaver`) for the current process.

### Sub-agent modules

- **Analysis** (`src/agents/sub_analyse_agent/`): clustering (embeddings + sklearn **KMeans**), deep per-cluster analysis, global six-module-style summary.
- **Writing** (`src/agents/sub_writing_agent/`): writing director, parallel writing group, retrieval agent, review agent, shared writing state models.

### HTTP surface (`main.py`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/research/runs` | Start a run; body: `query`, optional `kb_label`, `user_id`. Returns `run_id` (also used as `trace_id`). |
| GET | `/api/research/runs/{run_id}/stream` | **SSE** stream of `BackToFrontData` JSON events until `finished`. |
| POST | `/api/research/runs/{run_id}/input` | Submit HITL text for the run’s user proxy. |

Routers: **`/knowledge/...`**, **`/api/reports/...`**. Default ASGI bind: **`0.0.0.0:8001`**.

## End-to-end workflow (user-visible)

1. User submits a natural-language research request (and optionally a knowledge-base label).
2. Coordinator and background/planner stages shape a **reviewable plan**.
3. Search and **filter** reduce noise; **reading** extracts structured evidence.
4. **Evidence index** feeds **analysis** (clusters + deep + global).
5. **Writing** runs parallel chapter tasks with **RAG**; **faithfulness** gate reduces unsupported claims.
6. **Report** merges sections to Markdown; SSE shows progress; optional **evaluation** event at the end.

## Repository layout (concise)

```text
Paper-Agent/
├── main.py                      # FastAPI app: research runs, SSE, router includes
├── pyproject.toml               # Poetry dependencies (package-mode = false)
├── design.md                    # System design (primary architecture doc)
├── docs/                        # Extra docs (refactor phases, interview notes, README_cn)
├── src/
│   ├── agents/                  # LangGraph nodes, search/reading/writing/report, planner, researcher, reviewer
│   ├── api/                     # reports_router and other HTTP modules
│   ├── core/                    # config, models.yaml, system_params.yaml, state, LLM infra, workflow recovery
│   ├── knowledge/               # Knowledge base factory, Chroma impl, indexing, router
│   ├── rag/                     # LlamaIndex RAG service (retriever, rerank, ingestion, …)
│   ├── services/                # arXiv, Chroma client, run temp store, report history, …
│   ├── domain/                  # Paper domain: filter, evidence, faithfulness, citations
│   ├── evaluation/              # Post-run evaluators + LangSmith helpers
│   ├── plugins/                 # OCR / guards (optional paths)
│   ├── tasks/                   # Paper search task helpers
│   └── utils/                   # Logging, throttling, helpers
├── web/                         # Vue 3 + Vite 5 SPA (proxies /api and /knowledge to :8001)
├── test/                        # pytest modules (workflow, RAG, planning, faithfulness, …)
├── data/                        # Runtime KB / Chroma paths (see SAVE_DIR in system_params)
└── output/log/                  # Typical log output location (see logging config)
```

## Quick start

1. **Prerequisites**: Python **3.12** (see `pyproject.toml` upper bound), [Poetry](https://python-poetry.org/), Node.js for the web UI.
2. **Install backend**: `poetry install`
3. **Configure**:
   - Copy **`.env.example`** to **`.env`** and set at least one LLM provider key (e.g. `SILICONFLOW_API_KEY`, or keys described in `src/core/models.yaml` / comments in `.env.example`).
   - Adjust **`src/core/models.yaml`** (providers, `llm-routing`, per–client-type models).
   - Adjust **`src/core/system_params.yaml`** (RAG backend, concurrency, rate limits, logging paths).
4. **Run API**: `poetry run python main.py` → listens on **http://0.0.0.0:8001**
5. **Run web** (another terminal): `cd web && npm install && npm run dev` → **http://localhost:5173** (Vite proxies `/api` and `/knowledge` to port 8001).

## Configuration

- **Environment (`.env`)**: API keys, optional `DEFAULT_LLM_PROVIDER`, LangSmith (`LANGCHAIN_API_KEY` / `LANGSMITH_*`), `RAG_BACKEND`, DashScope / Volcengine keys as needed. See **`.env.example`** for authoritative variable names.
- **`src/core/models.yaml`**: Provider registry, embedding model, module-specific clients, **llm-routing** policies and rate-limit bucket names.
- **`src/core/system_params.yaml`**: Paths (`SAVE_DIR`), `KB_TYPE`, **rag** block (LlamaIndex toggles, HyDE, reranker, chunking), concurrency, logging, recovery-related defaults.

Never commit real secrets; `.env` is gitignored.

## Tech stack

| Layer | Technologies |
|--------|----------------|
| Runtime | Python 3.12, Poetry |
| API | FastAPI, Uvicorn, sse-starlette |
| Orchestration | LangGraph, LangSmith (optional tracing/eval) |
| Agents | pyautogen, autogen-agentchat, autogen-ext |
| LLM stack | LangChain chat integrations (`langchain-openai`, `langchain-community`), provider routing in-repo |
| Retrieval & RAG | arXiv, ChromaDB, **LlamaIndex** (`llama-index-*` packages) |
| NLP / ML | scikit-learn (KMeans, elbow), pandas, sentence-transformers (optional extra `rag-cross-encoder`) |
| Documents | PyMuPDF, python-docx, markdownify, beautifulsoup4 |
| Frontend | Vue 3.4, Vue Router 4, Vite 5, axios, marked |

## Contributing

Issues and pull requests are welcome. Please keep secrets out of git and align large behavior changes with **`design.md`** or **`docs/refactor/`** so reviewers can follow intent.

## License

MIT — see [LICENSE](LICENSE).

## Contact

- **GitHub Issues**: preferred for bugs and features  
- Upstream listing: https://github.com/Tswoen/paper-agent

---

⭐ If this project is useful, a star on the repo helps others discover it.

## Star history

[![Star History Chart](https://api.star-history.com/image?repos=Tswoen/Paper-Agent&type=date&legend=top-left)](https://www.star-history.com/?repos=Tswoen%2FPaper-Agent&type=date&logscale=&legend=top-left)
