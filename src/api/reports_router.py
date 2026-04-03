"""历史调研报告 REST API。"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.report_history_store import delete_report, get_detail, list_summaries

reports_router = APIRouter(prefix="/reports", tags=["reports"])


class ReportSummary(BaseModel):
    id: str
    title: str = ""
    query: str = ""
    status: str = "completed"
    createdAt: str = ""
    knowledgeBase: Optional[str] = None


class ReportDetail(ReportSummary):
    content: str = Field(default="", description="完整 Markdown 正文")


def _to_summary(entry: dict[str, Any]) -> ReportSummary:
    return ReportSummary(
        id=str(entry.get("id", "")),
        title=str(entry.get("title", "") or ""),
        query=str(entry.get("query", "") or ""),
        status=str(entry.get("status", "completed") or "completed"),
        createdAt=str(entry.get("createdAt", "") or ""),
        knowledgeBase=entry.get("knowledgeBase"),
    )


@reports_router.get("", response_model=list[ReportSummary])
async def list_reports():
    entries = await list_summaries()
    return [_to_summary(e) for e in entries]


@reports_router.get("/{report_id}", response_model=ReportDetail)
async def get_report(report_id: str):
    detail = await get_detail(report_id)
    if not detail:
        raise HTTPException(status_code=404, detail="报告不存在")
    content = str(detail.get("content", "") or "")
    meta = {k: v for k, v in detail.items() if k != "content"}
    summary = _to_summary(meta)
    return ReportDetail(**summary.model_dump(), content=content)


@reports_router.delete("/{report_id}", status_code=204)
async def remove_report(report_id: str):
    ok = await delete_report(report_id)
    if not ok:
        raise HTTPException(status_code=404, detail="报告不存在")
