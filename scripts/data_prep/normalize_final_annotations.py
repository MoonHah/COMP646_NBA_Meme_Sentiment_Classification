#!/usr/bin/env python3
"""Normalize a final annotation CSV into the project's current column schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import division_columns, image_path_for, load_schema, polarity_columns, read_csv, split_teams, team_divisions, write_csv, write_json


def copy_or_default(row: dict[str, str], key: str, default: str = "") -> str:
    return str(row.get(key, default) or default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/annotation/cz_final_annotations.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/all_human_labeled.csv"))
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--team-column", default="parsed_teams")
    parser.add_argument("--polarity-column", default="polarity")
    parser.add_argument("--summary-output", type=Path, default=Path("data/processed/normalization_summary.json"))
    args = parser.parse_args()

    schema = load_schema(args.schema)
    rows, fieldnames = read_csv(args.input)
    div_cols = division_columns(schema, "human")
    pol_cols = polarity_columns(schema, "human")
    base_fields = ["sample_id", "legacy_image_id", "image_path", "text", "raw_order", "parsed_teams"]
    output_fields = base_fields + div_cols + pol_cols + ["primary_team_or_context"]

    out_rows = []
    for idx, row in enumerate(rows):
        sample_id = copy_or_default(row, "sample_id") or copy_or_default(row, "id") or copy_or_default(row, "image_id")
        teams = split_teams(row.get(args.team_column, ""))
        divisions = team_divisions(teams, schema)
        out = {
            "sample_id": sample_id,
            "legacy_image_id": copy_or_default(row, "legacy_image_id"),
            "image_path": copy_or_default(row, "image_path", image_path_for(sample_id)),
            "text": copy_or_default(row, "text") or copy_or_default(row, "title"),
            "raw_order": copy_or_default(row, "raw_order", str(idx)),
            "parsed_teams": row.get(args.team_column, ""),
            "primary_team_or_context": copy_or_default(row, "primary_team_or_context") or (teams[0] if teams else ""),
        }
        for label, col in zip(schema["division_labels"], div_cols):
            out[col] = row.get(col, "1" if label in divisions else "0")
        polarity = row.get(args.polarity_column, "").strip().lower()
        for label, col in zip(schema["polarity_labels"], pol_cols):
            out[col] = row.get(col, "1" if polarity == label else "0")
        out_rows.append(out)

    write_csv(args.output, out_rows, output_fields)
    write_json(args.summary_output, {"input": str(args.input), "output": str(args.output), "num_rows": len(out_rows)})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
