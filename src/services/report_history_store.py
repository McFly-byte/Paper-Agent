"""历史调研报告持久化：index.json 元数据 + 每份报告独立 .md 文件。"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.config import config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_index_lock = asyncio.Lock()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_reports_root() -> Path:
    """报告存储根目录（相对路径相对于项目根）。"""
    rel = config.get("paths.reports_dir", "data/reports")
    p = Path(str(rel))
    if p.is_absolute():
        return p
    return _project_root() / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _index_path(root: Path) -> Path:
    return root / "index.json"


def _load_index_sync(root: Path) -> list[dict[str, Any]]:
    idx = _index_path(root)
    if not idx.exists():
        return []
    try:
        with open(idx, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取报告索引失败: %s", e)
        return []


def _save_index_sync(root: Path, entries: list[dict[str, Any]]) -> None:
    ensure_dir(root)
    final = _index_path(root)
    tmp = final.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    tmp.replace(final)


def extract_title_from_markdown(markdown: str, fallback: str, max_len: int = 120) -> str:
    """从正文取首个 Markdown 标题，否则截断 fallback。"""
    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            t = line.lstrip("#").strip()
            if t:
                return t[:max_len]
    t = fallback.strip().replace("\n", " ")
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


async def append_completed(
    markdown: str,
    query: str,
    knowledge_base: Optional[str] = None,
    *,
    reports_root: Optional[Path] = None,
) -> Optional[str]:
    """
    写入一条已完成报告。成功返回 report_id，空正文或 IO 失败返回 None（不抛异常）。
    """
    if not markdown or not markdown.strip():
        logger.info("跳过历史保存：报告正文为空")
        return None
    root = reports_root or get_reports_root()
    report_id = str(uuid.uuid4())
    title = extract_title_from_markdown(markdown, query)
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    entry: dict[str, Any] = {
        "id": report_id,
        "title": title,
        "query": query,
        "status": "completed",
        "createdAt": created,
        "knowledgeBase": knowledge_base,
    }
    try:
        ensure_dir(root)
        md_path = root / f"{report_id}.md"
        md_path.write_text(markdown, encoding="utf-8")
        async with _index_lock:
            entries = _load_index_sync(root)
            entries.append(entry)
            entries.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
            _save_index_sync(root, entries)
        logger.info("已保存历史报告: id=%s", report_id)
        return report_id
    except OSError as e:
        logger.exception("保存历史报告失败: %s", e)
        return None


async def list_summaries(
    reports_root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    root = reports_root or get_reports_root()
    if not root.exists():
        return []
    entries = _load_index_sync(root)
    entries.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    return entries


async def get_detail(
    report_id: str,
    reports_root: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    root = reports_root or get_reports_root()
    entries = _load_index_sync(root)
    meta = next((e for e in entries if e.get("id") == report_id), None)
    if not meta:
        return None
    md_path = root / f"{report_id}.md"
    if not md_path.is_file():
        return None
    try:
        content = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    out = dict(meta)
    out["content"] = content
    return out


async def delete_report(report_id: str, reports_root: Optional[Path] = None) -> bool:
    root = reports_root or get_reports_root()
    async with _index_lock:
        entries = _load_index_sync(root)
        new_entries = [e for e in entries if e.get("id") != report_id]
        if len(new_entries) == len(entries):
            return False
        md_path = root / f"{report_id}.md"
        try:
            if md_path.is_file():
                md_path.unlink()
        except OSError as e:
            logger.warning("删除报告 md 文件失败: %s", e)
        _save_index_sync(root, new_entries)
    return True
