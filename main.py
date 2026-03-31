from time import sleep
from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from src.agents.userproxy_agent import WebUserProxyAgent, userProxyAgent
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.knowledge.knowledge_router import knowledge
from fastapi import APIRouter

import asyncio
from src.core.state_models import BackToFrontData
# 设置日志
logger = setup_logger(name='main', log_file='project.log')

app = FastAPI() 
app.include_router(knowledge)
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
@app.get('/api/research')
async def research_stream(query: str):
    from src.agents.orchestrator import PaperAgentOrchestrator
    from src.core.state_models import State, ExecutionState

    # 异步生成器：SSE 的“数据源”，每次 yield 一条会变成一次 SSE 事件推给前端
    async def event_generator():
        while True:
            state = await state_queue.get()
            yield {"data": f"{state.model_dump_json()}"}

    # 用 sse-starlette 把异步生成器包装成 SSE 响应；
    event_source = EventSourceResponse(event_generator(), media_type="text/event-stream")

    orchestrator = PaperAgentOrchestrator(state_queue=state_queue)

    asyncio.create_task(orchestrator.run(user_request=query)) 

    
    return event_source

if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
    