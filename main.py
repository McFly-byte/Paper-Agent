import os

# 避免 AutoGen + OpenTelemetry 在 run_stream 被提前关闭（break / 中断）时上下文 detach 报错刷屏
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# 尽早加载 Config：写入 .env、合并 YAML，并应用 LangSmith LANGCHAIN_*（须在首次 LangGraph 执行前完成）
from src.core.config import config  # noqa: F401

from src.utils.log_utils import setup_logger
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from src.agents.userproxy_agent import WebUserProxyAgent, userProxyAgent
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.knowledge.knowledge_router import knowledge
from src.api.reports_router import reports_router

import asyncio
from src.core.state_models import BackToFrontData, ExecutionState
# 设置日志
logger = setup_logger(name='main', log_file='project.log')

app = FastAPI()
app.include_router(knowledge)
app.include_router(reports_router, prefix="/api")
# === CORS 配置（开发时可用 "*"，生产需限定具体域名） ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_queue = asyncio.Queue() # 全局队列，用于存储状态

# agent = WebUserProxyAgent("user_proxy")


async def _run_research_workflow(query: str, knowledge_base_label: str | None = None) -> None:
    """后台执行工作流；异常时写入队列，避免 create_task 吞掉异常导致 SSE 永不结束。"""
    from src.agents.orchestrator import PaperAgentOrchestrator

    logger.info("[工作流] 已接收调研请求，后台任务启动（控制台将按阶段打印进度）…")
    orchestrator = PaperAgentOrchestrator(state_queue=state_queue)
    try:
        await orchestrator.run(
            user_request=query,
            knowledge_base_label=knowledge_base_label,
        )
    except Exception as e:
        logger.exception("调研工作流执行失败: %s", e)
        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.FAILED.value,
                state="error",
                data=f"工作流异常（如 LLM 超时）: {e!s}",
            )
        )
        await state_queue.put(
            BackToFrontData(step=ExecutionState.FINISHED.value, state="finished", data=None)
        )

# ---------------------------------------------------------------------------
# 接口：接收前端的人工输入（Human-in-the-loop）
# 场景：搜索阶段 LLM 生成查询条件后，前端展示给用户确认/修改，用户点击确认后
#       前端把最终内容 POST 到这里，本接口唤醒 userProxyAgent，工作流继续往下执行
# ---------------------------------------------------------------------------
@app.post("/send_input")
async def send_input(data: dict):
    user_input = data.get("input")
    userProxyAgent.set_user_input(user_input)
    return JSONResponse({"status": 200, "msg": "已收到人工输入"})


# ---------------------------------------------------------------------------
# 接口：发起一次调研并订阅 SSE 流，持续接收进度与最终报告
# 流程：前端 GET /api/research?query=xxx → 本函数返回 SSE 响应 → 后台启动 orchestrator
#       orchestrator 各节点往 state_queue 里 put 状态 → event_generator 从 queue 取并 yield → 前端通过 EventSource 收到
# ---------------------------------------------------------------------------
@app.get("/api/research")
async def research_stream(query: str, kb_label: str | None = None):
    """kb_label：可选，用于历史报告中展示关联知识库名称。"""
    # 异步生成器：SSE 的“数据源”，每次 yield 一条会变成一次 SSE 事件推给前端
    async def event_generator():
        while True:
            state = await state_queue.get()
            yield {"data": f"{state.model_dump_json()}"}

    # 用 sse-starlette 把异步生成器包装成 SSE 响应；
    event_source = EventSourceResponse(event_generator(), media_type="text/event-stream")

    asyncio.create_task(_run_research_workflow(query, knowledge_base_label=kb_label))

    return event_source

if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
    