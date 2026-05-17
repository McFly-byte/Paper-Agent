"""Run audit 查看与下载（仅允许读取 workflow_v2.run_audit_dir 下内容）。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from src.core.config import config
from src.observability.run_audit import PROJECT_ROOT

router = APIRouter(tags=["run_audit"])


def _audit_root() -> Path:
    raw = config.get("workflow_v2.run_audit_dir", "artifacts/run_audits") or "artifacts/run_audits"
    p = Path(str(raw))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _run_dir(run_id: str) -> Path:
    if not run_id or ".." in run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="invalid_run_id")
    return (_audit_root() / run_id).resolve()


_BLOCKED_SEGMENTS = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        "id_rsa",
        "id_ed25519",
        "credentials.json",
        "secrets.json",
    }
)


def _safe_target(run_root: Path, file_path: str) -> Path:
    """防止 path traversal；禁止敏感文件名。"""
    if not file_path or file_path.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="invalid_path")
    norm = file_path.replace("\\", "/").lstrip("/")
    if ".." in norm.split("/"):
        raise HTTPException(status_code=400, detail="invalid_path")
    parts = [p for p in Path(norm).parts if p not in (".",)]
    for seg in parts:
        low = seg.lower()
        if seg.startswith(".") and low != ".gitkeep":
            raise HTTPException(status_code=403, detail="hidden_path_not_allowed")
        if low in _BLOCKED_SEGMENTS or low.endswith(".pem") or low.endswith("_rsa"):
            raise HTTPException(status_code=403, detail="sensitive_file_blocked")
    target = (run_root / Path(*parts)).resolve()
    try:
        target.relative_to(run_root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path_traversal") from exc
    return target


@router.get("/research/runs/{run_id}/audit")
async def get_run_audit_manifest(run_id: str):
    root = _run_dir(run_id)
    mf = root / "manifest.json"
    exists = mf.is_file()
    payload: dict = {"run_id": run_id, "exists": exists, "path": str(root), "manifest": None}
    if exists:
        try:
            payload["manifest"] = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail="manifest_corrupt") from exc
    return JSONResponse(payload)


@router.get("/research/runs/{run_id}/audit/files/{file_path:path}")
async def get_run_audit_file(run_id: str, file_path: str):
    run_root = _run_dir(run_id)
    target = _safe_target(run_root, file_path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file_not_found")
    text = target.read_text(encoding="utf-8", errors="replace")
    suf = target.suffix.lower()
    if suf == ".json":
        try:
            return JSONResponse(json.loads(text))
        except json.JSONDecodeError:
            return PlainTextResponse(text, media_type="text/plain; charset=utf-8")
    if suf == ".jsonl":
        return PlainTextResponse(text, media_type="application/x-ndjson; charset=utf-8")
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")
