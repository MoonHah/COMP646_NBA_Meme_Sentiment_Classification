#!/usr/bin/env python3
"""Shared helpers for the NBA meme data-preparation scripts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SCHEMA = Path("configs/label_schema.json")
TEAM_SPLIT_RE = re.compile(r"\s*(?:[,;/|]|\band\b|&|\+)\s*", re.IGNORECASE)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_schema(path: Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    return read_json(path)


def division_columns(schema: dict[str, Any], prefix: str = "human") -> list[str]:
    return [f"{prefix}_division__{label}" for label in schema["division_labels"]]


def polarity_columns(schema: dict[str, Any], prefix: str = "human") -> list[str]:
    return [f"{prefix}_polarity__{label}" for label in schema["polarity_labels"]]


def meme_type_columns(schema: dict[str, Any], prefix: str = "human") -> list[str]:
    return [f"{prefix}_type__{label}" for label in schema.get("meme_type_labels", [])]


def flag_columns(schema: dict[str, Any], prefix: str = "human") -> list[str]:
    return [f"{prefix}_flag__{label}" for label in schema.get("quality_flags", [])]


def label_columns(schema: dict[str, Any], prefix: str = "human", include_meme_type: bool = True) -> list[str]:
    cols = division_columns(schema, prefix) + polarity_columns(schema, prefix)
    if include_meme_type:
        cols += meme_type_columns(schema, prefix)
    return cols


def normalize_sample_id(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("t3_"):
        return text[3:]
    return text


def image_path_for(sample_id: str, image_dir: Path = Path("images_jpg")) -> str:
    return str(image_dir / f"{sample_id}.jpg")


def post_id(post: dict[str, Any], fallback: int) -> str:
    for key in ("sample_id", "id", "post_id", "image_id", "legacy_image_id", "name"):
        if post.get(key):
            return normalize_sample_id(post[key])
    return str(fallback)


def post_text(post: dict[str, Any]) -> str:
    pieces = []
    for key in ("title", "selftext", "text", "caption", "body"):
        value = str(post.get(key, "") or "").strip()
        if value and value.lower() not in {"[removed]", "[deleted]"}:
            pieces.append(value)
    return "\n".join(dict.fromkeys(pieces))


def records_from_json(path: Path) -> list[dict[str, str]]:
    raw = read_json(path)
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            posts = raw["data"]
        elif isinstance(raw.get("posts"), list):
            posts = raw["posts"]
        else:
            posts = list(raw.values())
    elif isinstance(raw, list):
        posts = raw
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    records = []
    for idx, post in enumerate(posts):
        if not isinstance(post, dict):
            continue
        sample_id = post_id(post, idx)
        records.append(
            {
                "sample_id": sample_id,
                "image_path": str(post.get("image_path") or image_path_for(sample_id)),
                "text": post_text(post),
                "raw_order": idx,
                "source_url": str(post.get("url") or post.get("image_url") or post.get("permalink") or ""),
            }
        )
    return records


def split_teams(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value:
        return []
    return [part.strip() for part in TEAM_SPLIT_RE.split(value) if part.strip()]


def team_divisions(teams: list[str], schema: dict[str, Any]) -> list[str]:
    mapping = schema["team_to_division"]
    divisions = []
    for team in teams:
        if team in mapping and mapping[team] not in divisions:
            divisions.append(mapping[team])
    return divisions
