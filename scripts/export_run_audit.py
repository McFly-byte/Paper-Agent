#!/usr/bin/env python3
"""从保存的 PaperAgentState JSON 手动导出审计包（不依赖 Chroma tmp store）。

示例：

  python scripts/export_run_audit.py --state-json ./state.json --out ./artifacts/run_audits/manual_export

说明：未外置在 JSON 中的大字段（如 workflow_search_results）将按 ``not_available`` 出现在审计包中；
若需完整包请在 run 进程内开启 ``workflow_v2.enable_run_audit_dump`` 由 orchestrator 自动导出。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _async_main() -> None:
    parser = argparse.ArgumentParser(description="从 state JSON 导出 run audit 目录")
    parser.add_argument("--state-json", required=True, help="PaperAgentState 的 JSON 文件路径")
    parser.add_argument("--out", required=True, help="输出目录（将直接写入该路径，不再追加 run_id）")
    parser.add_argument("--max-text-chars", type=int, default=4000)
    parser.add_argument("--max-papers", type=int, default=20)
    parser.add_argument("--max-evidence-items", type=int, default=120)
    parser.add_argument("--no-full-report", action="store_true")
    parser.add_argument("--no-written-body", action="store_true")
    parser.add_argument("--no-evidence-claims", action="store_true")
    parser.add_argument("--no-redact", action="store_true")
    args = parser.parse_args()

    raw = Path(args.state_json).read_text(encoding="utf-8")
    data = json.loads(raw)
    from src.core.state_models import PaperAgentState
    from src.observability.run_audit import RunAuditExporter

    state = PaperAgentState.model_validate(data)
    out_dir = Path(args.out).resolve()
    exporter = RunAuditExporter(
        str(Path.cwd()),
        max_text_chars=args.max_text_chars,
        max_papers=args.max_papers,
        max_evidence_items=args.max_evidence_items,
        include_full_report=not args.no_full_report,
        include_written_sections=not args.no_written_body,
        include_evidence_claims=not args.no_evidence_claims,
        redact_secrets=not args.no_redact,
    )
    path = await exporter.export(state, run_dir=out_dir)
    print(f"OK: audit written to {path}")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
