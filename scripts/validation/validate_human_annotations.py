#!/usr/bin/env python3
"""Validate human annotation CSVs before modeling or reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import division_columns, flag_columns, label_columns, load_schema, polarity_columns, read_csv, write_json


VALID_BINARY = {"0", "1"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/annotation/human_test_annotation_queue.csv"))
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotation/human_annotation_validation.json"))
    parser.add_argument("--require-labeled", action="store_true")
    args = parser.parse_args()

    schema = load_schema(args.schema)
    rows, fieldnames = read_csv(args.input)
    human_cols = label_columns(schema, "human", include_meme_type=True)
    div_cols = division_columns(schema, "human")
    pol_cols = polarity_columns(schema, "human")
    flags = flag_columns(schema, "human")
    issues = []

    missing = [col for col in ["sample_id", "image_path", *human_cols] if col not in fieldnames]
    if missing:
        issues.append({"issue": "missing_columns", "detail": missing})

    for row in rows:
        sample_id = row.get("sample_id", "")
        status = row.get("annotation_status", "")
        if args.require_labeled and status not in {"labeled", "reviewed", "complete"}:
            issues.append({"sample_id": sample_id, "issue": "not_labeled", "detail": status})
        cols_to_check = [col for col in human_cols + flags if col in fieldnames and str(row.get(col, "")).strip()]
        invalid = [col for col in cols_to_check if str(row.get(col, "")).strip() not in VALID_BINARY]
        if invalid:
            issues.append({"sample_id": sample_id, "issue": "invalid_binary_label", "detail": invalid})
        if all(col in fieldnames for col in pol_cols):
            values = [str(row.get(col, "")).strip() for col in pol_cols]
            if all(value in VALID_BINARY for value in values) and sum(int(value) for value in values) != 1:
                issues.append({"sample_id": sample_id, "issue": "invalid_polarity_count", "detail": values})
        if row.get("image_path") and not Path(row["image_path"]).exists():
            issues.append({"sample_id": sample_id, "issue": "missing_image", "detail": row["image_path"]})

    write_json(
        args.output,
        {
            "input": str(args.input),
            "num_rows": len(rows),
            "division_counts": {
                col: sum(int(row.get(col, 0) or 0) for row in rows if str(row.get(col, "")).strip() in VALID_BINARY)
                for col in div_cols
                if col in fieldnames
            },
            "polarity_counts": {
                col: sum(int(row.get(col, 0) or 0) for row in rows if str(row.get(col, "")).strip() in VALID_BINARY)
                for col in pol_cols
                if col in fieldnames
            },
            "issues": issues,
        },
    )
    print(f"Wrote {args.output}")
    print(f"Issues: {len(issues)}")
    if issues:
        raise SystemExit("Human annotation validation found issues.")


if __name__ == "__main__":
    main()
