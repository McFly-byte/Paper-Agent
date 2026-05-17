"""audit_router 路径安全与 manifest 读取。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    root = tmp_path / "run_audits"
    run_id = "11111111-1111-1111-1111-111111111111"
    d = root / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": ["manifest.json"], "summary": {}}),
        encoding="utf-8",
    )
    (d / "note.md").write_text("hello", encoding="utf-8")

    import src.api.audit_router as ar

    monkeypatch.setattr(ar, "_audit_root", lambda: root.resolve())

    from main import app

    return TestClient(app)


def test_audit_manifest(client: TestClient) -> None:
    rid = "11111111-1111-1111-1111-111111111111"
    r = client.get(f"/api/research/runs/{rid}/audit")
    assert r.status_code == 200
    data = r.json()
    assert data["exists"] is True
    assert data["manifest"]["run_id"] == rid


def test_audit_manifest_missing(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api.audit_router as ar

    monkeypatch.setattr(ar, "_audit_root", lambda: (tmp_path / "empty").resolve())
    (tmp_path / "empty").mkdir(exist_ok=True)
    from main import app

    c = TestClient(app)
    r = c.get("/api/research/runs/does-not-exist/audit")
    assert r.status_code == 200
    assert r.json()["exists"] is False


def test_path_traversal_rejected(client: TestClient) -> None:
    rid = "11111111-1111-1111-1111-111111111111"
    r = client.get(f"/api/research/runs/{rid}/audit/files/../../note.md")
    assert r.status_code in (400, 404)


def test_audit_file_note(client: TestClient) -> None:
    rid = "11111111-1111-1111-1111-111111111111"
    r = client.get(f"/api/research/runs/{rid}/audit/files/note.md")
    assert r.status_code == 200
