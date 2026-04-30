#!/usr/bin/env python3
"""Build train/val/test CSVs from a finalized human annotation file."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import division_columns, load_schema, polarity_columns, read_csv, write_csv, write_json


def polarity_key(row: dict[str, str], pol_cols: list[str]) -> str:
    active = [col for col in pol_cols if str(row.get(col, "")).strip() == "1"]
    return active[0] if active else "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/annotation/cz_final_annotations.csv"))
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.68)
    parser.add_argument("--val-ratio", type=float, default=0.17)
    parser.add_argument("--seed", type=int, default=646)
    parser.add_argument("--exclude-not-nba", action="store_true")
    args = parser.parse_args()

    schema = load_schema(args.schema)
    rows, fieldnames = read_csv(args.input)
    div_cols = division_columns(schema, "human")
    pol_cols = polarity_columns(schema, "human")
    required = ["sample_id", "image_path", "text", *div_cols, *pol_cols]
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    if args.exclude_not_nba and "human_flag__not_nba_relevant" in fieldnames:
        rows = [row for row in rows if str(row.get("human_flag__not_nba_relevant", "")).strip() != "1"]

    rng = random.Random(args.seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[polarity_key(row, pol_cols)].append(row)
    for group_rows in grouped.values():
        rng.shuffle(group_rows)

    splits = {"train": [], "val": [], "test": []}
    for group_rows in grouped.values():
        n = len(group_rows)
        n_train = round(n * args.train_ratio)
        n_val = round(n * args.val_ratio)
        splits["train"].extend(group_rows[:n_train])
        splits["val"].extend(group_rows[n_train : n_train + n_val])
        splits["test"].extend(group_rows[n_train + n_val :])

    for split_rows in splits.values():
        rng.shuffle(split_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        write_csv(args.output_dir / f"{split}.csv", split_rows, fieldnames)

    summary = {
        "input": str(args.input),
        "num_rows": {split: len(split_rows) for split, split_rows in splits.items()},
        "division_counts": {
            split: {col: sum(int(row[col]) for row in split_rows) for col in div_cols}
            for split, split_rows in splits.items()
        },
        "polarity_counts": {
            split: {col: sum(int(row[col]) for row in split_rows) for col in pol_cols}
            for split, split_rows in splits.items()
        },
    }
    write_json(args.output_dir / "split_summary.json", summary)
    print(f"Wrote splits to {args.output_dir}")


if __name__ == "__main__":
    main()
