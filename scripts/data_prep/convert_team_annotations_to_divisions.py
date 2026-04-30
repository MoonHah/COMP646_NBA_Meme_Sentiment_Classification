#!/usr/bin/env python3
"""Convert team annotations into NBA division multi-label columns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import division_columns, load_schema, read_csv, split_teams, team_divisions, write_csv, write_json


TEAM_COLUMNS = ["parsed_teams", "teams", "team", "primary_team_or_context"]


def find_team_column(fieldnames: list[str], requested: str | None) -> str:
    if requested:
        return requested
    for col in TEAM_COLUMNS:
        if col in fieldnames:
            return col
    raise SystemExit(f"Could not find a team column. Tried: {', '.join(TEAM_COLUMNS)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/annotation/cz_annotations.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/annotation/cz_final_annotations.csv"))
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--team-column", default=None)
    parser.add_argument("--summary-output", type=Path, default=Path("data/annotation/team_to_division_summary.json"))
    args = parser.parse_args()

    schema = load_schema(args.schema)
    rows, fieldnames = read_csv(args.input)
    team_col = find_team_column(fieldnames, args.team_column)
    div_cols = division_columns(schema, "human")

    out_rows = []
    unknown_teams: dict[str, int] = {}
    for row in rows:
        teams = split_teams(row.get(team_col, ""))
        divisions = team_divisions(teams, schema)
        for team in teams:
            if team not in schema["team_to_division"]:
                unknown_teams[team] = unknown_teams.get(team, 0) + 1
        out = dict(row)
        for label, col in zip(schema["division_labels"], div_cols):
            out[col] = 1 if label in divisions else 0
        out_rows.append(out)

    output_fields = list(dict.fromkeys(fieldnames + div_cols))
    write_csv(args.output, out_rows, output_fields)
    write_json(
        args.summary_output,
        {
            "input": str(args.input),
            "output": str(args.output),
            "team_column": team_col,
            "num_rows": len(out_rows),
            "unknown_teams": unknown_teams,
            "division_counts": {col: sum(int(row.get(col, 0) or 0) for row in out_rows) for col in div_cols},
        },
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
