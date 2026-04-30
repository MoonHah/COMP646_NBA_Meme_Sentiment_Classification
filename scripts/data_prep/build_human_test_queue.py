#!/usr/bin/env python3
"""Sample a roughly balanced human verification queue from weak labels."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import division_columns, flag_columns, label_columns, load_schema, read_csv, write_csv, write_json


def bucket_for(row: dict[str, str], div_cols: list[str]) -> str:
    active = [col for col in div_cols if str(row.get(col, "")).strip() == "1"]
    if not active:
        return "no_clear_division"
    return active[0].replace("weak_division__", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-input", type=Path, default=Path("data/weak/full_ai_labeling_queue_labeled.csv"))
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--seed", type=int, default=646)
    parser.add_argument("--output", type=Path, default=Path("data/annotation/human_test_annotation_queue.csv"))
    parser.add_argument("--train-pool-output", type=Path, default=Path("data/weak/weak_train_pool.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/annotation/stratified_subset_summary.json"))
    args = parser.parse_args()

    schema = load_schema(args.schema)
    rows, fieldnames = read_csv(args.weak_input)
    weak_div_cols = division_columns(schema, "weak")
    human_cols = label_columns(schema, "human", include_meme_type=True)
    human_flags = flag_columns(schema, "human")

    rng = random.Random(args.seed)
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[bucket_for(row, weak_div_cols)].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    selected = []
    bucket_names = sorted(buckets)
    while len(selected) < args.size and any(buckets.values()):
        for name in bucket_names:
            if len(selected) >= args.size:
                break
            if buckets[name]:
                selected.append(buckets[name].pop())

    selected_ids = {row["sample_id"] for row in selected}
    train_pool = [row for row in rows if row.get("sample_id") not in selected_ids]

    queue_fields = list(dict.fromkeys(fieldnames + human_cols + human_flags + ["annotation_status", "annotator", "notes"]))
    queue_rows = []
    for row in selected:
        out = dict(row)
        out.update({col: "" for col in human_cols + human_flags})
        out.update({"annotation_status": "unlabeled", "annotator": "", "notes": ""})
        queue_rows.append(out)

    write_csv(args.output, queue_rows, queue_fields)
    write_csv(args.train_pool_output, train_pool, fieldnames)
    write_json(
        args.summary_output,
        {
            "weak_input": str(args.weak_input),
            "num_input_rows": len(rows),
            "requested_size": args.size,
            "num_selected": len(selected),
            "bucket_counts_before_sampling": {
                name: sum(1 for row in rows if bucket_for(row, weak_div_cols) == name) for name in bucket_names
            },
            "bucket_counts_selected": {
                name: sum(1 for row in selected if bucket_for(row, weak_div_cols) == name) for name in bucket_names
            },
            "human_queue": str(args.output),
            "weak_train_pool": str(args.train_pool_output),
        },
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.train_pool_output}")


if __name__ == "__main__":
    main()
