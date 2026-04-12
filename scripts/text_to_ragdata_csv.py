#!/usr/bin/env python3
"""
将「query,answer」格式的 CSV 文本转为 UTF-8 的 RAGdata.csv（两列：query, answer）。

用法：
  poetry run python scripts/text_to_ragdata_csv.py rag_pairs.txt
  poetry run python scripts/text_to_ragdata_csv.py -o data/csv/custom.csv rag_pairs.txt
  type rag_pairs.txt | poetry run python scripts/text_to_ragdata_csv.py

默认输出：项目根目录下 data/csv/RAGdata.csv（不存在则创建目录）。
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_header(name: str) -> str:
    return name.strip().lstrip("\ufeff").strip().lower()


def _rows_from_csv_text(raw: str) -> list[dict[str, str]]:
    text = raw.lstrip("\ufeff")
    if not text.strip():
        return []

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        return []

    key_map: dict[str, str] = {}
    for orig in reader.fieldnames:
        if orig is None:
            continue
        norm = _normalize_header(orig)
        if norm in ("query", "answer"):
            key_map[norm] = orig

    if "query" not in key_map or "answer" not in key_map:
        raise ValueError(
            "首行表头必须包含 query 与 answer 两列（可带 BOM、大小写不敏感）。"
            f"当前表头: {reader.fieldnames!r}"
        )

    q_col, a_col = key_map["query"], key_map["answer"]
    out: list[dict[str, str]] = []
    for i, row in enumerate(reader, start=2):
        if row is None:
            continue
        q = (row.get(q_col) or "").strip()
        a = (row.get(a_col) or "").strip()
        if not q and not a:
            continue
        if not q or not a:
            raise ValueError(f"第 {i} 行存在空字段：query 与 answer 均不能为空（可删全空行）。")
        out.append({"query": q, "answer": a})
    return out


def main() -> int:
    default_out = _project_root() / "data" / "csv" / "RAGdata.csv"
    parser = argparse.ArgumentParser(description="CSV 文本 → data/csv/RAGdata.csv")
    parser.add_argument(
        "input_file",
        nargs="?",
        help="输入文件（UTF-8）；省略则从标准输入读取",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_out,
        help=f"输出路径（默认: {default_out})",
    )
    args = parser.parse_args()

    if args.input_file:
        path = Path(args.input_file)
        if not path.is_file():
            print(f"错误：找不到文件 {path}", file=sys.stderr)
            return 1
        raw = path.read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()

    try:
        rows = _rows_from_csv_text(raw)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    out_path: Path = args.output
    if not out_path.is_absolute():
        out_path = (_project_root() / out_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "answer"],
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"已写入 {len(rows)} 条 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
