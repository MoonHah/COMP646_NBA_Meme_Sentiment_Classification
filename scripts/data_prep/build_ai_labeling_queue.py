#!/usr/bin/env python3
"""Build a queue for VLM/AI weak labeling from raw posts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import label_columns, load_schema, read_csv, records_from_json, write_csv, write_json


def load_records(raw_csv: Path, raw_json: Path) -> list[dict[str, str]]:
    if raw_csv.exists():
        rows, _ = read_csv(raw_csv)
        return rows
    return records_from_json(raw_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, default=Path("data/raw/raw_posts.csv"))
    parser.add_argument("--raw-json", type=Path, default=Path("nbameme.json"))
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--output", type=Path, default=Path("data/weak/full_ai_labeling_queue.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/weak/ai_labeling_queue_summary.json"))
    args = parser.parse_args()

    schema = load_schema(args.schema)
    records = load_records(args.raw_csv, args.raw_json)
    weak_cols = label_columns(schema, "weak", include_meme_type=True)
    fields = ["sample_id", "image_path", "text", "raw_order", "source_url"] + weak_cols + [
        "weak_label_note",
        "weak_label_status",
    ]
    rows = []
    for record in records:
        row = dict(record)
        row.update({col: "" for col in weak_cols})
        row.update({"weak_label_note": "", "weak_label_status": "pending"})
        rows.append(row)

    write_csv(args.output, rows, fields)
    write_json(args.summary_output, {"num_rows": len(rows), "output": str(args.output)})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
