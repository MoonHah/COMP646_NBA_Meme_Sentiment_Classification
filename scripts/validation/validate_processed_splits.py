#!/usr/bin/env python3
"""Validate train/val/test CSV splits for model training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_SCHEMA = Path("configs/label_schema.json")
DEFAULT_OUTPUT = Path("data/processed/validated_split_summary.json")
VALID_BINARY_VALUES = {"0", "1"}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_schema(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def label_groups(schema: dict[str, object]) -> dict[str, list[str]]:
    return {
        "division": [f"human_division__{label}" for label in schema["division_labels"]],
        "polarity": [f"human_polarity__{label}" for label in schema["polarity_labels"]],
        "meme_type": [f"human_type__{label}" for label in schema["meme_type_labels"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    groups = label_groups(schema)
    label_cols = groups["division"] + groups["polarity"]

    splits = {}
    fieldnames_by_split = {}
    issues = []
    seen: dict[str, str] = {}

    for split in ["train", "val", "test"]:
        path = args.split_dir / f"{split}.csv"
        rows, fieldnames = read_csv(path)
        splits[split] = rows
        fieldnames_by_split[split] = fieldnames

        missing_cols = [col for col in ["sample_id", "image_path", "text", *label_cols] if col not in fieldnames]
        if missing_cols:
            issues.append({"split": split, "issue": "missing_columns", "detail": missing_cols})

        for row in rows:
            sample_id = row.get("sample_id", "")
            if sample_id in seen:
                issues.append(
                    {
                        "split": split,
                        "issue": "duplicate_sample_across_splits",
                        "detail": {"sample_id": sample_id, "previous_split": seen[sample_id]},
                    }
                )
            seen[sample_id] = split

            invalid_binary = [
                col for col in label_cols if row.get(col, "").strip() not in VALID_BINARY_VALUES
            ]
            polarity_count = sum(
                int(row.get(col, "0")) for col in groups["polarity"] if row.get(col, "") in VALID_BINARY_VALUES
            )
            missing_image = not Path(row.get("image_path", "")).exists()

            if invalid_binary:
                issues.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "issue": "invalid_binary_label",
                        "detail": invalid_binary,
                    }
                )
            if polarity_count != 1:
                issues.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "issue": "invalid_polarity_count",
                        "detail": polarity_count,
                    }
                )
            if missing_image:
                issues.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "issue": "missing_image",
                        "detail": row.get("image_path", ""),
                    }
                )

    summary = {
        "split_dir": str(args.split_dir),
        "num_rows": {split: len(rows) for split, rows in splits.items()},
        "num_unique_samples": len(seen),
        "label_counts": {
            split: {
                col: sum(int(row[col]) for row in rows)
                for col in label_cols
                if rows and col in rows[0]
            }
            for split, rows in splits.items()
        },
        "fieldnames_match": len({tuple(fields) for fields in fieldnames_by_split.values()}) == 1,
        "active_tasks": {
            "division": "multi-label sigmoid",
            "polarity": "3-class softmax",
        },
        "excluded_labels": groups["meme_type"],
        "issues": issues,
    }
    write_json(args.output, summary)
    print(f"Wrote {args.output}")
    print(f"Issues: {len(issues)}")
    if issues:
        raise SystemExit("Processed splits have validation issues.")


if __name__ == "__main__":
    main()
