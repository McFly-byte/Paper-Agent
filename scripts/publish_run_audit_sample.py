"""Publish a reviewed run audit directory into examples/run_audits.

The script refuses incomplete audit packages by default. It copies only files
inside the source audit directory and never follows arbitrary external paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.observability.audit_analyzer import analyze_run_audit
from src.observability.run_audit import PROJECT_ROOT


def _resolve_source(run_id_or_path: str) -> Path:
    raw = Path(run_id_or_path)
    if raw.exists():
        return raw.resolve()
    return (PROJECT_ROOT / "artifacts" / "run_audits" / run_id_or_path).resolve()


def _copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id_or_path", help="run id under artifacts/run_audits or an audit directory path")
    parser.add_argument("--dest-root", default=str(PROJECT_ROOT / "examples" / "run_audits"))
    parser.add_argument("--allow-warnings", action="store_true")
    args = parser.parse_args()

    src = _resolve_source(args.run_id_or_path)
    if not src.is_dir():
        print(json.dumps({"ok": False, "error": f"source_not_found: {src}"}, ensure_ascii=False, indent=2))
        return 2

    report = analyze_run_audit(src)
    blocking = report["error_count"] > 0 or (report["warning_count"] > 0 and not args.allow_warnings)
    if blocking:
        print(json.dumps({"ok": False, "analysis": report}, ensure_ascii=False, indent=2))
        return 1

    dest_root = Path(args.dest_root)
    run_id = src.name
    dest = (dest_root / run_id).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    _copytree_clean(src, dest)
    print(json.dumps({"ok": True, "source": str(src), "dest": str(dest), "analysis": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
