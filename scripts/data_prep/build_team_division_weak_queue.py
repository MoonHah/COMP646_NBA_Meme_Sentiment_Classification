#!/usr/bin/env python3
"""Build a team-grounded weak-label queue using existing team annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import (
    division_columns,
    image_path_for,
    load_schema,
    polarity_columns,
    read_csv,
    split_teams,
    team_divisions,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("final_annotations.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/weak/team_division_ai_labeling_queue.csv"))
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--team-column", default="parsed_teams")
    parser.add_argument("--summary-output", type=Path, default=Path("data/weak/team_division_queue_summary.json"))
    args = parser.parse_args()

    schema = load_schema(args.schema)
    rows, fieldnames = read_csv(args.input)
    weak_div_cols = division_columns(schema, "weak")
    weak_pol_cols = polarity_columns(schema, "weak")
    out_fields = [
        "sample_id",
        "image_path",
        "text",
        "raw_order",
        "parsed_teams",
        "primary_team_or_context",
        *weak_div_cols,
        *weak_pol_cols,
        "weak_label_note",
        "weak_label_status",
    ]

    out_rows = []
    skipped_no_team = 0
    for idx, row in enumerate(rows):
        sample_id = row.get("sample_id") or row.get("id") or row.get("image_id") or row.get("legacy_image_id")
        sample_id = str(sample_id or "").strip()
        teams = split_teams(row.get(args.team_column, ""))
        divisions = team_divisions(teams, schema)
        if not divisions:
            skipped_no_team += 1
            continue
        out = {
            "sample_id": sample_id,
            "image_path": row.get("image_path") or image_path_for(sample_id),
            "text": row.get("text") or row.get("title") or "",
            "raw_order": row.get("raw_order") or idx,
            "parsed_teams": row.get(args.team_column, ""),
            "primary_team_or_context": row.get("primary_team_or_context") or (teams[0] if teams else ""),
            "weak_label_note": "division labels mapped from existing team annotations; polarity left blank",
            "weak_label_status": "team_mapped",
        }
        for label, col in zip(schema["division_labels"], weak_div_cols):
            out[col] = 1 if label in divisions else 0
        for col in weak_pol_cols:
            out[col] = ""
        out_rows.append(out)

    write_csv(args.output, out_rows, out_fields)
    write_json(
        args.summary_output,
        {
            "input": str(args.input),
            "output": str(args.output),
            "num_rows": len(out_rows),
            "skipped_no_team": skipped_no_team,
            "input_columns": fieldnames,
        },
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
