#!/usr/bin/env python3
"""Merge train.csv and val.csv into a final training CSV.

The held-out test.csv is never included. Use this after validation has already
served its model-selection role.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/train_val.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/processed/train_val_summary.json"))
    args = parser.parse_args()

    train_rows, train_fields = read_csv(args.split_dir / "train.csv")
    val_rows, val_fields = read_csv(args.split_dir / "val.csv")
    test_rows, _ = read_csv(args.split_dir / "test.csv")

    if train_fields != val_fields:
        raise SystemExit("train.csv and val.csv have different columns.")

    train_ids = {row["sample_id"] for row in train_rows}
    val_ids = {row["sample_id"] for row in val_rows}
    test_ids = {row["sample_id"] for row in test_rows}

    duplicate_train_val = sorted(train_ids & val_ids)
    overlap_with_test = sorted((train_ids | val_ids) & test_ids)
    if duplicate_train_val:
        raise SystemExit(f"Duplicate sample_id across train/val: {duplicate_train_val[:10]}")
    if overlap_with_test:
        raise SystemExit(f"train+val overlaps with held-out test: {overlap_with_test[:10]}")

    combined_rows = train_rows + val_rows
    write_csv(args.output, combined_rows, train_fields)
    write_json(
        args.summary_output,
        {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_val_rows": len(combined_rows),
            "test_rows_held_out": len(test_rows),
            "output": str(args.output),
            "test_included": False,
            "columns": train_fields,
        },
    )
    print(f"Wrote {args.output}")
    print(f"Rows: train={len(train_rows)}, val={len(val_rows)}, train_val={len(combined_rows)}")
    print(f"Held-out test rows not included: {len(test_rows)}")


if __name__ == "__main__":
    main()
