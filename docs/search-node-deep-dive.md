# 【检索】节点：整体流程与重要类/方法

## 一、整体流程（入口 → 出口）

### 1. 入口与输入

- **图入口**：LangGraph 的 `START` 唯一指向 `search_node`（见 `orchestrator.py` 中 `set_entry_point("search_node")`、`add_edge(START, "search_node")`）。
- **输入**：`State`（TypedDict），其中：
  - `state["value"]`：`PaperAgentState`，检索节点只用其 **`user_request`**（用户自然语言需求，如「帮我写一篇有关 llm 在无人驾驶方面的调研报告」）；
  - `state["state_queue"]`：`asyncio.Queue`，用于向 SSE 推送 `BackToFrontData`，供前端展示进度与人工审核界面。

### 2. 中间主要步骤（按执行顺序）

| 步骤 | 说明 | 代码位置 |
|------|------|----------|
| 1 | 从 `state` 取出 `state_queue`、`current_state`，将 `current_step` 设为 `ExecutionState.SEARCHING`，并向队列推送「初始化」状态，通知前端进入检索阶段 | `search_agent.py` 59–62 行 |
| 2 | 用 `user_request` 拼 prompt，调用 **search_agent**（LLM）生成结构化查询条件（关键词列表 + 可选时间范围） | 64–69 行 |
| 3 | 将 LLM 返回的查询条件通过 `state_queue` 推送给前端，状态为 `user_review`，**等待人工审核** | 69–70 行 |
| 4 | 调用 **userProxyAgent.on_messages()**，协程在此挂起，直到前端通过 `/send_input` 调用 **userProxyAgent.set_user_input()** 传入用户确认/修改后的字符串 | 72–76 行 |
| 5 | 用 **parse_search_query()** 把前端返回的字符串解析为 **SearchQuery**（querys / start_date / end_date） | 76 行 |
| 6 | 实例化 **PaperSearcher**，调用 **search_papers(querys, start_date, end_date)** 执行 arXiv 检索 | 79–85 行 |
| 7 | 将结果写入 **current_state.search_results**；若有结果则推送 `completed`，若无则推送 `error` 并设置 **current_state.error.search_node_error** | 86–99 行 |
| 8 | 返回状态更新 **`{"value": current_state}`**，供 LangGraph 合并到全局 state | 99 行 |

异常时在 `except` 中设置 `state["value"].error.search_node_error` 并推送 `error`，仍返回 `state`，保证图可路由到错误节点。

### 3. 输出与前后衔接

- **输出**：
  - 对图的「输出」：返回 **`{"value": current_state}`**，其中 `current_state.search_results` 为论文元数据列表（每项含 `paper_id`、`title`、`authors`、`summary`、`url`、`pdf_url` 等），若有错误则 `current_state.error.search_node_error` 被置位。
  - 对前端的输出：通过 **state_queue** 持续推送 `BackToFrontData`（步骤、状态、文案/数据），包括初始化、待审核、完成/错误等。
- **与前一节点**：检索是图的**起点**，无前一业务节点；初始 state 由 **PaperAgentOrchestrator.run()** 在 `main.py` 的 `/api/research` 中构造并传入图。
- **与后一节点**：**condition_handler**（`orchestrator.py` 48–67 行）在 `search_node` 完成后检查：若 `err.search_node_error is None` 且 `current_step == ExecutionState.SEARCHING`，则下一跳为 **reading_node**；否则进入 **handle_error_node**。即检索节点只负责把「可用的论文列表」写入 `search_results`，阅读节点从同一 `state["value"]` 读取 `search_results` 进行后续解析与抽取。

---

## 二、重要类与方法

### 1. `search_node`（函数）

- **文件**：`src/agents/search_agent.py`
- **功能**：检索节点的**唯一入口**，完成「LLM 生成查询 → 人工审核 → arXiv 检索 → 写回状态并推送前端」的全流程。
- **实现要点**：
  - 签名 `async def search_node(state: State) -> State`，符合 LangGraph 节点约定；返回的 dict 会与现有 state 合并。
  - 依赖：`State`/`PaperAgentState`/`ExecutionState`/`BackToFrontData`、`search_agent`、`userProxyAgent`、`PaperSearcher`、`parse_search_query`。
  - 与状态：只读 `state["value"].user_request`，写 `state["value"].current_step`、`state["value"].search_results`、`state["value"].error.search_node_error`；与 `state["state_queue"]` 的交互见上表。
- **设计考量**：Human-in-the-loop 放在「LLM 出结果之后、调用 arXiv 之前」，避免错误查询浪费请求；异常统一捕获并写入 `NodeError`，由编排器条件边统一路由到错误节点，不抛异常以保证图可结束。

---

### 2. `SearchQuery`（Pydantic 模型）

- **文件**：`src/agents/search_agent.py`
- **功能**：表示「检索条件」的结构化输出：关键词列表 + 可选日期范围，供 LLM 结构化输出和 **parse_search_query** 解析前端回传字符串。
- **实现要点**：
  - `querys: List[str]`、`start_date`、`end_date`（均为可选，格式 YYYY-MM-DD）。
  - 被 **search_agent** 用作 `output_content_type`，LLM 按此结构返回；人工审核后前端传回的是**字符串**，需经 **parse_search_query** 转成 `SearchQuery` 再传给 **PaperSearcher.search_papers**。
- **设计考量**：用 Pydantic 做 schema，既约束 LLM 输出又便于校验；与前端约定用字符串形式回传，解析集中在一个函数内，便于维护和兼容前端格式变化。

---

### 3. `search_agent`（AssistantAgent）

- **文件**：`src/agents/search_agent.py`
- **功能**：根据用户自然语言需求生成结构化检索条件（关键词 + 时间范围），是「语义 → 检索条件」的转换层。
- **实现要点**：
  - 使用 **create_search_model_client()** 得到的 `model_client`（配置键 `search-model`，见 `model_client.py`；未配置则走 **create_default_client()**）。
  - **system_message** 使用 **search_agent_prompt**（`src/core/prompts.py`）：角色为论文查询助手，要求做语义分析并输出精确英文检索条件，含示例（如「近三年 Transformer 机器翻译」→ 关键词 + 年份范围）。
  - 调用方式：`response = await search_agent.run(task=prompt)`，prompt 中仅包含 `user_request`；取 `response.messages[-1].content` 作为结构化查询字符串，再经 **parse_search_query** 或直接作为展示/审核用字符串推给前端。
- **设计考量**：单一职责「自然语言 → 查询条件」，便于换模型或换 prompt；结构化输出（SearchQuery）减少解析错误；模型配置与默认回退集中在 **model_client**，便于多环境与可观测性。

---

### 4. `parse_search_query(s: str) -> SearchQuery`

- **文件**：`src/agents/search_agent.py`
- **功能**：把前端/LLM 返回的**字符串**解析为 **SearchQuery** 对象，供 **PaperSearcher.search_papers** 使用。
- **实现要点**：
  - 用正则从字符串中提取 `querys=` 的列表（`ast.literal_eval` 安全解析）、`start_date=`、`end_date=` 的字符串。
  - 若匹配失败则对应字段为默认值（如 `querys=[]`、日期为 None）。
- **设计考量**：前端可能传回「类 repr」的格式，正则 + literal_eval 在保证安全的前提下兼容这种格式；解析集中在一处，便于后续改为 JSON 等格式时只改此函数。

---

### 5. `WebUserProxyAgent` / `userProxyAgent`

- **文件**：`src/agents/userproxy_agent.py`
- **功能**：在检索节点中实现 **Human-in-the-loop**：LLM 给出查询条件后，挂起等待用户在前端确认/修改，再继续执行 arXiv 检索。
- **实现要点**：
  - 继承 AutoGen 的 **UserProxyAgent**；单例 **userProxyAgent** 在 `search_agent` 中被使用。
  - **on_messages**：创建 `asyncio.Future` 并 `await self.waiting_future`，协程挂起；**set_user_input(user_input)** 由外部（如 FastAPI 的 **/send_input**）调用，对 `waiting_future.set_result(user_input)`，使 **on_messages** 返回 **TextMessage(content=user_input, source="human")**。
  - 在 **search_node** 中的调用链：`result = await userProxyAgent.on_messages(...)` → 前端 POST `/send_input` 触发 **userProxyAgent.set_user_input(data["input"])** → **search_node** 拿到 `result.content` 再交给 **parse_search_query**。
- **设计考量**：用 Future 解耦「HTTP 请求」与「图执行」的线程/协程，避免轮询；同一队列 + 同一 userProxyAgent 保证「谁发起调研，谁收到对应的人工输入」，适合单会话 SSE 长连接场景。

---

### 6. `PaperSearcher`（类）与 `search_papers`（方法）

- **文件**：`src/tasks/paper_search.py`
- **功能**：封装 arXiv 检索：根据关键词列表与可选日期范围构造查询、调用 arxiv API、将结果格式化为统一字典列表（paper_id、title、authors、summary、url、pdf_url 等）。
- **实现要点**：
  - **search_papers**：将 `querys` 拼成 `all:"term1" OR all:"term2" ...`；若有 `start_date`/`end_date` 则用 **\_format_date** 转成 arXiv 的 `YYYYMMDDHHMM` 并加 `submittedDate:[... TO ...]`；用 **arxiv.Search** 创建搜索、**search.results()** 取迭代器，再经 **format_papers_list** → **\_parse_paper_result** 转成 `List[Dict]`。
  - **\_format_date**：支持多种 str 与 datetime 格式，统一成 arXiv 要求的字符串；解析失败时有 fallback（如 dateutil 或当前日期）。
  - 注意：**search.results()** 为同步阻塞调用，当前 **search_papers** 为 async 但内部未用 `run_in_executor`，在 IO 密集场景下可能阻塞事件循环；若需高并发可考虑将 arxiv 调用丢到线程池。
- **设计考量**：检索逻辑与节点解耦，放在 **tasks** 下便于单测和扩展（如增加 Semantic Scholar）；日期、排序、max_results 等参数可配置，便于后续做策略扩展；异常在 **search_papers** 内记录并 re-raise，由 **search_node** 统一捕获并写入 **NodeError**。

---

### 7. `State` / `PaperAgentState` / `BackToFrontData` / `ExecutionState` / `NodeError`

- **文件**：`src/core/state_models.py`
- **功能**：定义检索节点所读写的「图状态」与「对前端的推送结构」。
- **实现要点**：
  - **State**：TypedDict，`state_queue` + `value`（PaperAgentState）；LangGraph 按此类型在各节点间传递状态。
  - **PaperAgentState**：检索节点读 **user_request**，写 **current_step**、**search_results**、**error**（NodeError）；**search_results** 为 `List[Dict[str, Any]]`，与 **PaperSearcher** 返回格式一致。
  - **BackToFrontData**：Pydantic，`step`（如 ExecutionState.SEARCHING）、`state`（如 "initializing" / "user_review" / "completed" / "error"）、`data`（文案或 payload）；检索节点多次 **state_queue.put(BackToFrontData(...))**，由 **main.py** 的 SSE **event_generator** 从 **state_queue** 取并推给前端。
  - **NodeError.search_node_error**：检索节点在「无结果」或异常时赋值，**condition_handler** 据此决定下一跳为 **reading_node** 或 **handle_error_node**。
- **设计考量**：状态与推送结构集中定义，保证图与前端约定一致；错误按节点分字段，便于定位和条件路由；**search_results** 用通用 Dict 便于与下游阅读节点和后续扩展字段兼容。

---

### 8. `PaperAgentOrchestrator` 与 `condition_handler`

- **文件**：`src/agents/orchestrator.py`
- **功能**：编排器构建 LangGraph，**condition_handler** 在 **search_node** 执行完后决定下一节点（reading_node / handle_error_node）。
- **实现要点**：
  - **search_node** 注册为普通节点，**START → search_node**，**search_node** 后接 **add_conditional_edges("search_node", self.condition_handler)**。
  - **condition_handler** 读取 `state["value"].current_step` 与 `state["value"].error`；若 `search_node_error is None` 且 `current_step == ExecutionState.SEARCHING` 则返回 `"reading_node"`，否则返回 `"handle_error_node"`。
- **设计考量**：检索作为起点、无前驱，出口仅两种（正常进阅读、错误进错误节点），逻辑清晰；与其它节点共用同一 **condition_handler**，用顺序 if-elif 统一管理整图路由，便于维护。

---

### 9. `create_search_model_client` / `create_model_client("search-model")`

- **文件**：`src/core/model_client.py`
- **功能**：为 **search_agent** 提供 LLM 客户端（OpenAIChatCompletionClient），用于「用户需求 → 检索条件」的调用。
- **实现要点**：**create_search_model_client()** 内部调用 **create_model_client("search-model")**，从 **config**（如 models.yaml）读取 `search-model` 的 `model-provider`、`model`；若未配置则 **create_default_client()**。**ModelClient.create_client** 会设置 **model_info**（含 structured_output 等）和 **max_retries**、**timeout**。
- **设计考量**：检索模型可单独配置与降级，不影响阅读/写作等节点；与其它节点共用同一配置抽象，便于统一换模型或做可观测性（如按 client_type 打点）。

---

## 三、调用链小结（检索节点主路径）

```
main: GET /api/research
  → asyncio.create_task(orchestrator.run(query))
  → graph.ainvoke({ state_queue, value: PaperAgentState(user_request=query, ...) })

orchestrator: START → search_node(state)

search_agent.search_node(state):
  → state_queue.put(BackToFrontData(SEARCHING, "initializing", None))
  → search_agent.run(task=prompt(user_request))
  → state_queue.put(BackToFrontData(SEARCHING, "user_review", search_query_str))
  → userProxyAgent.on_messages(...)   # 挂起
       [前端展示审核界面 → 用户点击确认 → POST /send_input → userProxyAgent.set_user_input(input)]
  → parse_search_query(result.content) → SearchQuery
  → PaperSearcher().search_papers(querys, start_date, end_date)
       → arxiv.Search(...).results() → format_papers_list → _parse_paper_result
  → current_state.search_results = results
  → state_queue.put(BackToFrontData(SEARCHING, "completed"/"error", ...))
  → return {"value": current_state}

orchestrator.condition_handler(state)
  → 若无 search_node_error 且 current_step==SEARCHING → "reading_node"
  → 否则 → "handle_error_node"
```

以上内容可直接对照 `src/agents/search_agent.py`、`src/tasks/paper_search.py`、`src/agents/userproxy_agent.py`、`src/agents/orchestrator.py`、`src/core/state_models.py`、`src/core/model_client.py` 和 `main.py` 阅读与调试。

---

## 四、面试追问（检索节点）

以下问题结合本节点真实实现设计，用于区分「做过 vs 只跑过 demo」，回答时需结合代码或设计说明。

---

**Q1.** 人工审核为什么放在「LLM 生成查询条件之后、调 arXiv 之前」？如果用户一直不点确认，协程会怎样？超时或取消要怎么加？

- **考察点**：Human-in-the-loop 的放置理由、对 asyncio 挂起与取消的理解、边界与健壮性设计。
- **简要回答要点**：① 先审核再请求 arXiv，避免错误/模糊查询浪费外部 API 且结果难复用。② `userProxyAgent.on_messages()` 里 `await self.waiting_future` 会一直挂起，没有超时；可给 `asyncio.wait_for(..., timeout=...)` 或配合 `CancellationToken` 做取消。③ 超时后可向队列推送「已超时」状态并设 `search_node_error`，由 condition_handler 进入错误节点。

---

**Q2.** `PaperSearcher.search_papers` 声明为 `async`，但内部 `arxiv.Search(...).results()` 是同步阻塞的，这会带来什么问题？你会怎么改？

- **考察点**：对事件循环与阻塞调用的理解、异步封装实践。
- **简要回答要点**：① 同步网络/IO 会阻塞整个事件循环，同一进程里其他请求（如其它 SSE、/send_input）会被拖慢。② 把 arxiv 的「构建 Search + results() + 解析」放到 `run_in_executor(thread_pool, sync_search_fn)` 里执行，`search_papers` 仍为 async，内部 `await loop.run_in_executor(...)`，不阻塞事件循环。③ 若后续支持多数据源或重试，可把「单次 arxiv 调用」封装成同步函数再交给 executor。

---

**Q3.** 当前 `state_queue` 和 `userProxyAgent` 都是全局单例，两个用户同时发起调研（两个 SSE 连接、两个图在跑）会怎样？如何改才能支持多会话互不干扰？

- **考察点**：对「单例 + 共享队列」在多会话下的问题分析、会话隔离设计。
- **简要回答要点**：① 两个图的 BackToFrontData 都进同一个 queue，两个前端会互相收到对方的进度/审核态。② 人工输入只认「最后一次」谁调了 `set_user_input`，无法区分是哪个会话，会导致串会话。③ 改造思路：每个请求（或每个 SSE 连接）单独建一个 `state_queue` 和单独的 `WebUserProxyAgent`（或给 agent 绑定 session_id），`orchestrator.run()` 用该请求的 queue 和 agent；或在一个 agent 内用 `session_id -> Future` 的 map，`/send_input` 带 session_id 唤醒对应 Future。

---

**Q4.** `parse_search_query` 用正则 + `ast.literal_eval` 解析前端传回的字符串，若前端传了畸形或恶意输入（如超长字符串、畸形括号、非列表内容）可能出什么情况？`ast.literal_eval` 在这里主要防住了什么？

- **考察点**：输入校验与安全、对 literal_eval 边界与风险的了解。
- **简要回答要点**：① 正则可匹配失败导致 querys/日期为默认值，业务上表现为「用空列表或错误日期去搜」；极端输入可能触发正则性能问题。② `ast.literal_eval` 只允许字面量（列表、字符串、数字等），不能执行任意代码，相比 `eval` 更安全。③ 仍可能解析出超大列表或超长字符串，建议对 `querys` 做长度/数量上限校验，对日期做格式与范围校验，解析异常时打日志并设 `search_node_error` 或返回明确错误给前端。

---

**Q5.** 检索阶段「无结果」和「调用 arXiv 抛异常」在代码里是怎么区分的？错误信息是怎么一路传到前端、并让图进入错误节点的？

- **考察点**：错误分类、状态与错误在节点与编排间的传递、前后端约定。
- **简要回答要点**：① 无结果：`len(results) == 0` 时设 `current_state.error.search_node_error` 并推送 `state="error"` 的 BackToFrontData，仍 `return {"value": current_state}`。② 异常：`except` 里给 `state["value"].error.search_node_error` 赋值并 push error 的 BackToFrontData，再 `return state`。③ 两种情况下 condition_handler 都会看到 `search_node_error` 非空，下一跳为 `handle_error_node` 而非 reading_node。④ 前端通过 SSE 收到的 BackToFrontData 里 `step`/`state`/`data` 展示「错误」和文案（如「没有找到相关论文」或异常信息）。

---

**Q6.** 当前检索没有对 arXiv 做重试和限流，若你要加「失败重试」和「请求限流」，会放在哪一层、怎么设计（结合现有 PaperSearcher 与 search_node）？

- **考察点**：容错与限流的设计层级、对重试/限流实现方式的掌握。
- **简要回答要点**：① 重试：放在 PaperSearcher 内更合适，在 `search_papers` 里对「构建 Search + results()」包一层重试（如 tenacity 或手写 for + except），只对可重试异常（网络超时、5xx）重试，避免把「无结果」当失败重试。② 限流：若多会话共享同一 arXiv 调用，可在 PaperSearcher 或封装层用 asyncio.Semaphore 限制并发数；若用 run_in_executor，可限制线程池大小 + 全局 Semaphore。③ 重试与限流都尽量封装在「调用 arXiv」这一层，search_node 只处理「成功/业务失败/最终异常」，保持节点逻辑简单。

---

**Q7.** 为什么用 `BackToFrontData(step, state, data)` 推给前端，而不是直接把 `PaperAgentState` 序列化推送？这样设计有什么利弊？

- **考察点**：前后端状态契约设计、最小暴露与演进性。
- **简要回答要点**：① 只推送「当前步骤 + 展示态 + 本步要展示的数据」，前端按 step/state 渲染 UI（如 loading、待审核、完成、错误），data 只带本步需要的文案或 payload，避免把整份 state（含 search_results 等大对象）全量推给前端。② 好处：前端不依赖全局 state 结构、带宽更小、前后端可独立演进（新增步骤只需约定新的 step/state/data）。③ 代价：前端若需要历史步骤的详细数据需自行在客户端累积或由后端在 data 里按需带一点；设计时要约定好 step/state 枚举和 data 的语义，避免歧义。

---

**Q8.** 若要把「检索」扩展成多数据源（例如再加 Semantic Scholar 或校内库），在不大改 search_node 的前提下，你会怎么抽象？PaperSearcher 和状态（如 search_results）要怎么设计？

- **考察点**：扩展性、抽象边界、数据源异构与统一状态。
- **简要回答要点**：① 抽象一层「检索接口」：例如 `SearchBackend.search(querys, start_date, end_date) -> List[Dict]`，PaperSearcher 实现该接口并内部只调 arXiv；再增加 SemanticScholarBackend 等，统一返回格式（至少含 paper_id、title、url、summary 等下游阅读节点依赖的字段）。② search_node 里通过配置或策略选择 backend，或组合多个 backend 再合并/去重。③ search_results 保持 `List[Dict]`，在 Dict 里用统一字段名（或加 source 字段区分来源），阅读节点只依赖公共字段；若不同来源的 id 冲突，可加前缀或命名空间（如 arxiv:xxx、semantic:yyy）。
