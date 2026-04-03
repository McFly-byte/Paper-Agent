"""report_history_store 单元测试（临时目录，不污染 data/reports）。

手动联调建议：
1. 启动后端 uvicorn 与前端 vite（/api 已代理）。
2. 跑通一次完整工作流后访问 GET /api/reports，应至少有一条 completed。
3. GET /api/reports/{id} 的 content 与 data/reports/{id}.md 一致。
4. 历史页删除后，index.json 与对应 .md 均被移除。
"""

import asyncio
import tempfile
from pathlib import Path

from src.services import report_history_store as store


async def _append_list_get_delete(root: Path) -> None:
    rid = await store.append_completed(
        "# Hello\n\n正文",
        "调研查询",
        knowledge_base="测试知识库",
        reports_root=root,
    )
    assert rid is not None
    lst = await store.list_summaries(reports_root=root)
    assert len(lst) == 1
    assert lst[0]["title"] == "Hello"
    assert lst[0]["query"] == "调研查询"
    assert lst[0]["knowledgeBase"] == "测试知识库"
    assert lst[0]["status"] == "completed"

    d = await store.get_detail(rid, reports_root=root)
    assert d is not None
    assert "正文" in d["content"]

    ok = await store.delete_report(rid, reports_root=root)
    assert ok is True
    assert await store.list_summaries(reports_root=root) == []


def test_append_list_get_delete() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_append_list_get_delete(Path(tmp)))


def test_extract_title_fallback() -> None:
    assert store.extract_title_from_markdown("无标题\n\n段落", "备用标题") == "备用标题"


def test_append_empty_skips() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rid = await store.append_completed("", "q", reports_root=root)
            assert rid is None
            assert await store.list_summaries(reports_root=root) == []

    asyncio.run(_run())


def test_concurrent_append() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r1, r2 = await asyncio.gather(
                store.append_completed("# A\n", "qa", reports_root=root),
                store.append_completed("# B\n", "qb", reports_root=root),
            )
            assert r1 and r2 and r1 != r2
            lst = await store.list_summaries(reports_root=root)
            assert len(lst) == 2

    asyncio.run(_run())


def test_delete_missing_returns_false() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ok = await store.delete_report("00000000-0000-0000-0000-000000000000", reports_root=Path(tmp))
            assert ok is False

    asyncio.run(_run())
