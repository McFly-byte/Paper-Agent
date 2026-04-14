import os
import uuid
import asyncio
from dataclasses import dataclass

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from src.core.config import config  # noqa: F401

from src.utils.log_utils import setup_logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from src.agents.userproxy_agent import WebUserProxyAgent
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.knowledge.knowledge_router import knowledge
from src.api.reports_router import reports_router

from src.core.state_models import BackToFrontData, ExecutionState

logger = setup_logger(name="main", log_file="project.log")

app = FastAPI()
app.include_router(knowledge)
app.include_router(reports_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class RunRecord:
    run_id: str
    queue: asyncio.Queue
    user_proxy: WebUserProxyAgent


class RunRegistry:
    """进程内：run_id → 队列与用户代理（单 worker MVP）。"""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def register(self, record: RunRecord) -> None:
        async with self._lock:
            self._runs[record.run_id] = record

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def mark_completed(self, run_id: str) -> None:
        async with self._lock:
            self._runs.pop(run_id, None)


run_registry = RunRegistry()


def _agent_name_for_run(run_id: str) -> str:
    """AutoGen 要求 agent name 为合法 Python 标识符；UUID 含 '-' 非法，需替换。"""
    return f"user_proxy_{run_id.replace('-', '_')}"


def _new_run() -> RunRecord:
    run_id = str(uuid.uuid4())
    return RunRecord(
        run_id=run_id,
        queue=asyncio.Queue(),
        user_proxy=WebUserProxyAgent(_agent_name_for_run(run_id)),
    )


def _sse_generator(queue: asyncio.Queue):
    async def event_generator():
        while True:
            msg = await queue.get()
            yield {"data": msg.model_dump_json()}
            if msg.state == "finished":
                break

    return event_generator()


class CreateResearchRunBody(BaseModel):
    query: str = Field(min_length=1)
    kb_label: str | None = None
    user_id: str | None = None


class HitlInputBody(BaseModel):
    input: str = Field(default="")


async def _run_research_workflow(
    *,
    run_id: str,
    state_queue: asyncio.Queue,
    user_proxy: WebUserProxyAgent,
    query: str,
    knowledge_base_label: str | None = None,
    user_id: str | None = None,
) -> None:
    from src.agents.orchestrator import PaperAgentOrchestrator

    logger.info("[工作流][run_id=%s] 后台任务启动", run_id)
    orchestrator = PaperAgentOrchestrator(state_queue=state_queue)
    try:
        await orchestrator.run(
            user_request=query,
            run_id=run_id,
            user_proxy=user_proxy,
            knowledge_base_label=knowledge_base_label,
            user_id=user_id,
        )
    except Exception as e:
        logger.exception("[工作流][run_id=%s] 执行失败: %s", run_id, e)
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
    finally:
        await run_registry.mark_completed(run_id)


@app.post("/api/research/runs", status_code=201)
async def create_research_run(body: CreateResearchRunBody):
    rec = _new_run()
    await run_registry.register(rec)
    asyncio.create_task(
        _run_research_workflow(
            run_id=rec.run_id,
            state_queue=rec.queue,
            user_proxy=rec.user_proxy,
            query=body.query,
            knowledge_base_label=body.kb_label,
            user_id=body.user_id,
        )
    )
    return JSONResponse(
        status_code=201,
        content={"run_id": rec.run_id, "trace_id": rec.run_id},
    )


@app.get("/api/research/runs/{run_id}/stream")
async def research_run_stream(run_id: str):
    rec = run_registry.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown_run")

    return EventSourceResponse(_sse_generator(rec.queue), media_type="text/event-stream")


@app.post("/api/research/runs/{run_id}/input")
async def research_run_input(run_id: str, body: HitlInputBody):
    rec = run_registry.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown_run")

    code = rec.user_proxy.set_user_input(body.input)
    if code == "ok":
        return JSONResponse({"status": 200, "msg": "已收到人工输入"})
    if code == "already_resolved":
        raise HTTPException(status_code=409, detail="hitl_already_resolved")
    raise HTTPException(status_code=409, detail="no_pending_hitl")


@app.post("/send_input")
async def send_input_deprecated(data: dict):
    """已废弃：请使用 POST /api/research/runs/{run_id}/input。"""
    raise HTTPException(
        status_code=410,
        detail="deprecated_use_run_scoped_input",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
