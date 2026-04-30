#!/usr/bin/env python3
"""Generate report-ready SVG visualizations for the NBA meme project.

The script intentionally uses only the Python standard library for plots so it
can run in Colab or locally without installing matplotlib/pandas.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_SCHEMA = Path("configs/label_schema.json")
DEFAULT_OUTPUT_DIR = Path("report_figures")

COLORS = {
    "train": "#4C78A8",
    "val": "#F58518",
    "test": "#54A24B",
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#8C8C8C",
    "grid": "#D9D9D9",
    "text": "#222222",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def short_division(label: str) -> str:
    return label.replace(" Division", "")


def col_name(prefix: str, label: str) -> str:
    return f"{prefix}__{label}"


def division_cols(schema: dict[str, Any]) -> list[str]:
    return [f"human_division__{label}" for label in schema["division_labels"]]


def polarity_cols(schema: dict[str, Any]) -> list[str]:
    return [f"human_polarity__{label}" for label in schema["polarity_labels"]]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#222}",
        ".title{font-size:22px;font-weight:700}",
        ".axis{font-size:12px}",
        ".label{font-size:11px}",
        ".legend{font-size:12px}",
        "</style>",
    ]


def save_svg(path: Path, parts: list[str]) -> None:
    write_text(path, "\n".join(parts + ["</svg>\n"]))


def grouped_bar_svg(
    path: Path,
    title: str,
    categories: list[str],
    series: dict[str, list[float]],
    y_label: str,
    width: int = 1050,
    height: int = 560,
) -> None:
    margin = {"left": 78, "right": 34, "top": 72, "bottom": 112}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_value = max([1.0] + [value for values in series.values() for value in values])
    max_value *= 1.12
    names = list(series)
    bar_group_w = plot_w / max(1, len(categories))
    bar_w = min(28, bar_group_w / (len(names) + 1.2))

    parts = svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')
    parts.append(f'<text x="20" y="{margin["top"] + plot_h/2}" transform="rotate(-90 20 {margin["top"] + plot_h/2})" class="axis">{esc(y_label)}</text>')

    for i in range(6):
        value = max_value * i / 5
        y = margin["top"] + plot_h - (value / max_value) * plot_h
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width-margin["right"]}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{margin["left"]-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{value:.0f}</text>')

    for c_idx, category in enumerate(categories):
        group_x = margin["left"] + c_idx * bar_group_w + bar_group_w / 2
        parts.append(f'<text x="{group_x:.1f}" y="{height-54}" text-anchor="end" transform="rotate(-35 {group_x:.1f} {height-54})" class="axis">{esc(category)}</text>')
        start_x = group_x - (len(names) * bar_w) / 2
        for s_idx, name in enumerate(names):
            value = series[name][c_idx]
            h = (value / max_value) * plot_h
            x = start_x + s_idx * bar_w
            y = margin["top"] + plot_h - h
            color = COLORS.get(name, [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]][s_idx % 4])
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + (bar_w-2)/2:.1f}" y="{y-4:.1f}" text-anchor="middle" class="label">{value:.0f}</text>')

    legend_x = margin["left"]
    legend_y = height - 24
    for idx, name in enumerate(names):
        x = legend_x + idx * 110
        color = COLORS.get(name, [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]][idx % 4])
        parts.append(f'<rect x="{x}" y="{legend_y-12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x+20}" y="{legend_y}" class="legend">{esc(name)}</text>')

    save_svg(path, parts)


def horizontal_bar_svg(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    x_label: str,
    color: str = COLORS["blue"],
    width: int = 900,
    height: int = 470,
    value_format: str = "{:.3f}",
) -> None:
    margin = {"left": 190, "right": 42, "top": 68, "bottom": 54}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_value = max([1.0] + values)
    row_h = plot_h / max(1, len(labels))

    parts = svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')
    for i in range(6):
        value = max_value * i / 5
        x = margin["left"] + (value / max_value) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"]+plot_h}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" class="axis">{value:.1f}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin["top"] + idx * row_h + row_h * 0.2
        h = row_h * 0.58
        w = (value / max_value) * plot_w
        parts.append(f'<text x="{margin["left"]-12}" y="{y+h*0.72:.1f}" text-anchor="end" class="axis">{esc(label)}</text>')
        parts.append(f'<rect x="{margin["left"]}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{margin["left"]+w+8:.1f}" y="{y+h*0.72:.1f}" class="label">{esc(value_format.format(value))}</text>')
    parts.append(f'<text x="{margin["left"] + plot_w/2}" y="{height-6}" text-anchor="middle" class="axis">{esc(x_label)}</text>')
    save_svg(path, parts)


def heatmap_svg(
    path: Path,
    title: str,
    rows: list[str],
    cols: list[str],
    matrix: list[list[int]],
    width: int = 700,
    height: int = 600,
) -> None:
    margin = {"left": 130, "right": 34, "top": 92, "bottom": 96}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    cell_w = plot_w / max(1, len(cols))
    cell_h = plot_h / max(1, len(rows))
    max_value = max([1] + [value for row in matrix for value in row])

    parts = svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')
    parts.append(f'<text x="{margin["left"] + plot_w/2}" y="62" text-anchor="middle" class="axis">Predicted label</text>')
    parts.append(f'<text x="20" y="{margin["top"] + plot_h/2}" transform="rotate(-90 20 {margin["top"] + plot_h/2})" class="axis">Gold label</text>')
    for c_idx, col in enumerate(cols):
        x = margin["left"] + c_idx * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{height-52}" text-anchor="end" transform="rotate(-35 {x:.1f} {height-52})" class="axis">{esc(col)}</text>')
    for r_idx, row_label in enumerate(rows):
        y = margin["top"] + r_idx * cell_h + cell_h / 2
        parts.append(f'<text x="{margin["left"]-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{esc(row_label)}</text>')
        for c_idx, value in enumerate(matrix[r_idx]):
            x = margin["left"] + c_idx * cell_w
            y0 = margin["top"] + r_idx * cell_h
            intensity = value / max_value
            blue = int(245 - intensity * 145)
            green = int(248 - intensity * 115)
            red = int(250 - intensity * 210)
            fill = f"rgb({red},{green},{blue})"
            parts.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="white"/>')
            if value:
                parts.append(f'<text x="{x+cell_w/2:.1f}" y="{y0+cell_h/2+5:.1f}" text-anchor="middle" class="label">{value}</text>')
    save_svg(path, parts)


def count_labels(rows: list[dict[str, str]], cols: list[str]) -> list[int]:
    return [sum(1 for row in rows if row.get(col, "").strip() == "1") for col in cols]


def load_splits(split_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {split: read_csv(split_dir / f"{split}.csv") for split in ("train", "val", "test")}


def find_split_results(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for split in ("test", "val"):
        if split in metrics:
            return split, metrics[split]
    raise ValueError("Metrics JSON must contain a 'test' or 'val' result.")


def per_label_f1(metrics_split: dict[str, Any]) -> tuple[list[str], list[float]]:
    labels = []
    scores = []
    for key, value in metrics_split["division"]["per_label"].items():
        if key.startswith("human_division__"):
            labels.append(short_division(key.replace("human_division__", "")))
            scores.append(float(value["f1-score"]))
    return labels, scores


def polarity_confusion(metrics_split: dict[str, Any], polarity_labels: list[str]) -> list[list[int]]:
    matrix = [[0 for _ in polarity_labels] for _ in polarity_labels]
    index = {label: idx for idx, label in enumerate(polarity_labels)}
    for item in metrics_split.get("generations", []):
        gold = item.get("gold_polarity")
        pred = item.get("pred_polarity")
        if gold in index and pred in index:
            matrix[index[gold]][index[pred]] += 1
    return matrix


def missing_extra_counts(metrics_split: dict[str, Any], division_labels: list[str]) -> tuple[list[int], list[int]]:
    missing = Counter()
    extra = Counter()
    for item in metrics_split.get("generations", []):
        gold = set(item.get("gold_divisions", []))
        pred = set(item.get("pred_divisions", []))
        for label in gold - pred:
            missing[label] += 1
        for label in pred - gold:
            extra[label] += 1
    return [missing[label] for label in division_labels], [extra[label] for label in division_labels]


def write_summary_markdown(
    path: Path,
    schema: dict[str, Any],
    splits: dict[str, list[dict[str, str]]],
    metrics: dict[str, Any] | None,
) -> None:
    lines = ["# Report Visualization Summary", ""]
    lines.append("## Split Sizes")
    for split, rows in splits.items():
        lines.append(f"- {split}: {len(rows)}")
    lines.append("")

    if metrics:
        split_name, result = find_split_results(metrics)
        lines.extend(
            [
                f"## Model Metrics ({split_name})",
                f"- Division macro-F1: {result['division']['macro_f1']:.3f}",
                f"- Division micro-F1: {result['division']['micro_f1']:.3f}",
                f"- Polarity accuracy: {result['polarity']['accuracy']:.3f}",
                f"- Polarity macro-F1: {result['polarity']['macro_f1']:.3f}",
                f"- Parse failures: {result.get('parse_failures', 'n/a')}",
                "",
                "## Division Per-Label F1",
            ]
        )
        for label, score in zip(*per_label_f1(result)):
            lines.append(f"- {label}: {score:.3f}")
        lines.append("")

    write_text(path, "\n".join(lines))


def write_qualitative_examples(
    path: Path,
    split_rows: list[dict[str, str]],
    metrics_split: dict[str, Any],
    max_examples: int,
) -> None:
    by_id = {row["sample_id"]: row for row in split_rows}
    generations = metrics_split.get("generations", [])
    errors = [
        item
        for item in generations
        if set(item.get("gold_divisions", [])) != set(item.get("pred_divisions", []))
        or item.get("gold_polarity") != item.get("pred_polarity")
    ]
    correct = [
        item
        for item in generations
        if set(item.get("gold_divisions", [])) == set(item.get("pred_divisions", []))
        and item.get("gold_polarity") == item.get("pred_polarity")
    ]
    selected = [("Correct", item) for item in correct[:max_examples]] + [
        ("Error", item) for item in errors[:max_examples]
    ]

    cards = []
    for kind, item in selected:
        row = by_id.get(item["sample_id"], {})
        image_path = row.get("image_path", "")
        text = row.get("text", "")
        cards.append(
            f"""
            <article class="card">
              <h2>{esc(kind)}: {esc(item['sample_id'])}</h2>
              <img src="../{esc(image_path)}" alt="{esc(item['sample_id'])}">
              <p><strong>Text:</strong> {esc(text[:260])}</p>
              <p><strong>Gold divisions:</strong> {esc(item.get('gold_divisions', []))}</p>
              <p><strong>Pred divisions:</strong> {esc(item.get('pred_divisions', []))}</p>
              <p><strong>Gold polarity:</strong> {esc(item.get('gold_polarity', ''))}</p>
              <p><strong>Pred polarity:</strong> {esc(item.get('pred_polarity', ''))}</p>
            </article>
            """
        )
    html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Qualitative Examples</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #ddd; padding: 12px; border-radius: 6px; }}
    .card h2 {{ font-size: 16px; margin: 0 0 10px; }}
    img {{ max-width: 100%; max-height: 260px; object-fit: contain; background: #f5f5f5; }}
    p {{ font-size: 13px; line-height: 1.35; }}
  </style>
</head>
<body>
  <h1>Qualitative Examples</h1>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    write_text(path, html_doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--examples-per-group", type=int, default=5)
    args = parser.parse_args()

    schema = read_json(args.schema)
    splits = load_splits(args.split_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    div_cols = division_cols(schema)
    pol_cols = polarity_cols(schema)
    div_labels = [short_division(label) for label in schema["division_labels"]]
    pol_labels = list(schema["polarity_labels"])

    grouped_bar_svg(
        args.output_dir / "split_sizes.svg",
        "Dataset Split Sizes",
        ["train", "val", "test"],
        {"rows": [len(splits[name]) for name in ("train", "val", "test")]},
        "Number of memes",
        width=720,
        height=460,
    )
    grouped_bar_svg(
        args.output_dir / "division_label_distribution.svg",
        "Division Label Counts by Split",
        div_labels,
        {split: count_labels(rows, div_cols) for split, rows in splits.items()},
        "Number of positive labels",
    )
    grouped_bar_svg(
        args.output_dir / "polarity_label_distribution.svg",
        "Polarity Label Counts by Split",
        pol_labels,
        {split: count_labels(rows, pol_cols) for split, rows in splits.items()},
        "Number of memes",
        width=760,
        height=460,
    )

    metrics = read_json(args.metrics) if args.metrics else None
    if metrics:
        split_name, metrics_split = find_split_results(metrics)
        labels, scores = per_label_f1(metrics_split)
        horizontal_bar_svg(
            args.output_dir / f"{split_name}_division_per_label_f1.svg",
            f"{split_name.title()} Division Per-Label F1",
            labels,
            scores,
            "F1 score",
            color=COLORS["blue"],
            value_format="{:.3f}",
        )

        missing, extra = missing_extra_counts(metrics_split, list(schema["division_labels"]))
        grouped_bar_svg(
            args.output_dir / f"{split_name}_division_missing_extra.svg",
            f"{split_name.title()} Division Error Counts",
            div_labels,
            {"missing": missing, "extra": extra},
            "Number of labels",
            width=1050,
            height=540,
        )

        heatmap_svg(
            args.output_dir / f"{split_name}_polarity_confusion.svg",
            f"{split_name.title()} Polarity Confusion Matrix",
            pol_labels,
            pol_labels,
            polarity_confusion(metrics_split, pol_labels),
        )
        write_qualitative_examples(
            args.output_dir / f"{split_name}_qualitative_examples.html",
            splits[split_name],
            metrics_split,
            args.examples_per_group,
        )

    write_summary_markdown(args.output_dir / "summary.md", schema, splits, metrics)
    print(f"Wrote report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
