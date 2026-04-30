#!/usr/bin/env python3
"""Build raw post and human-annotation queue CSVs from nbameme.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import (
    flag_columns,
    label_columns,
    load_schema,
    records_from_json,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("nbameme.json"))
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--raw-output", type=Path, default=Path("data/raw/raw_posts.csv"))
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=Path("data/annotation/full_human_annotation_queue.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/annotation/raw_queue_summary.json"),
    )
    args = parser.parse_args()

    schema = load_schema(args.schema)
    records = records_from_json(args.input)
    raw_fields = ["sample_id", "image_path", "text", "raw_order", "source_url"]
    write_csv(args.raw_output, records, raw_fields)

    human_cols = label_columns(schema, "human", include_meme_type=True)
    flags = flag_columns(schema, "human")
    queue_fields = raw_fields + human_cols + flags + ["annotation_status", "annotator", "notes"]
    queue_rows = []
    for record in records:
        row = dict(record)
        row.update({col: "" for col in human_cols + flags})
        row.update({"annotation_status": "unlabeled", "annotator": "", "notes": ""})
        queue_rows.append(row)
    write_csv(args.queue_output, queue_rows, queue_fields)

    missing_images = [row["sample_id"] for row in records if not Path(row["image_path"]).exists()]
    duplicate_count = len(records) - len({row["sample_id"] for row in records})
    write_json(
        args.summary_output,
        {
            "input": str(args.input),
            "num_posts": len(records),
            "num_missing_images": len(missing_images),
            "missing_images": missing_images[:50],
            "num_duplicate_sample_ids": duplicate_count,
            "raw_output": str(args.raw_output),
            "queue_output": str(args.queue_output),
        },
    )
    print(f"Wrote {args.raw_output}")
    print(f"Wrote {args.queue_output}")


if __name__ == "__main__":
    main()
