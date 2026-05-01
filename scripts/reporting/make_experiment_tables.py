#!/usr/bin/env python3
"""Create report tables from Qwen LoRA evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = {
    "frequency_baseline": Path("data/results/frequency_baseline_test.json"),
    "qwen_zeroshot": Path("data/results/qwen_zeroshot_test.json"),
    "val_baseline": Path("data/results/qwen_lora_val_no_team_context_detailed.json"),
    "val_recall": Path("data/results/qwen_lora_val_recall_prompt.json"),
    "val_balanced": Path("data/results/qwen_lora_val_balanced_prompt.json"),
    "test_balanced_train_only": Path("data/results/qwen_lora_test_balanced_final.json"),
    "test_balanced_train_val": Path("data/results/qwen_lora_test_balanced_train_val.json"),
}

DIVISION_KEYS = [
    "human_division__Atlantic Division",
    "human_division__Central Division",
    "human_division__Southeast Division",
    "human_division__Northwest Division",
    "human_division__Pacific Division",
    "human_division__Southwest Division",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_metric_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def metric_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.metric:
        return dict(parse_metric_arg(value) for value in args.metric)
    return DEFAULT_METRICS


def result_splits(metrics: dict[str, Any]) -> list[str]:
    return [split for split in ("val", "test") if split in metrics]


def rounded(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def short_division(key: str) -> str:
    return key.replace("human_division__", "").replace(" Division", "")


def summary_row(name: str, path: Path, metrics: dict[str, Any], split: str) -> dict[str, Any]:
    result = metrics[split]
    config = metrics.get("adapter_training_config") or {}
    return {
        "experiment": name,
        "split": split,
        "prompt_style": metrics.get("prompt_style") or config.get("prompt_style", ""),
        "train_file": config.get("train_file", ""),
        "adapter": metrics.get("adapter", ""),
        "metrics_file": str(path),
        "division_macro_f1": result["division"]["macro_f1"],
        "division_micro_f1": result["division"]["micro_f1"],
        "polarity_accuracy": result["polarity"]["accuracy"],
        "polarity_macro_f1": result["polarity"]["macro_f1"],
        "parse_failures": result.get("parse_failures", ""),
    }


def division_row(name: str, path: Path, metrics: dict[str, Any], split: str) -> dict[str, Any]:
    result = metrics[split]
    row: dict[str, Any] = {
        "experiment": name,
        "split": split,
        "metrics_file": str(path),
    }
    per_label = result["division"]["per_label"]
    for key in DIVISION_KEYS:
        stats = per_label.get(key, {})
        label = short_division(key)
        row[f"{label}_precision"] = stats.get("precision", "")
        row[f"{label}_recall"] = stats.get("recall", "")
        row[f"{label}_f1"] = stats.get("f1-score", "")
    return row


def polarity_row(name: str, path: Path, metrics: dict[str, Any], split: str) -> dict[str, Any]:
    result = metrics[split]
    row: dict[str, Any] = {
        "experiment": name,
        "split": split,
        "metrics_file": str(path),
    }
    per_label = result["polarity"]["per_label"]
    for key in ["human_polarity__positive", "human_polarity__negative", "human_polarity__neutral"]:
        stats = per_label.get(key, {})
        label = key.replace("human_polarity__", "")
        row[f"{label}_precision"] = stats.get("precision", "")
        row[f"{label}_recall"] = stats.get("recall", "")
        row[f"{label}_f1"] = stats.get("f1-score", "")
    return row


def markdown_table(rows: list[dict[str, Any]], columns: list[str], display_names: dict[str, str] | None = None) -> str:
    display_names = display_names or {}
    header = [display_names.get(col, col) for col in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = rounded(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict[str, Any]], columns: list[str], caption: str, label: str) -> str:
    col_spec = "l" * len(columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(columns).replace("_", "\\_") + " \\\\",
        "\\hline",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = rounded(value)
            values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def find_row(rows: list[dict[str, Any]], experiment: str, split: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("experiment") == experiment and row.get("split") == split:
            return row
    return None


def metric(row: dict[str, Any] | None, key: str) -> str:
    if row is None:
        return "--"
    return rounded(row.get(key, ""))


def report_validation_prompt_table(summary_rows: list[dict[str, Any]]) -> str:
    experiments = [
        ("Baseline", "val_baseline"),
        ("Recall-oriented", "val_recall"),
        ("Balanced", "val_balanced"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Prompt & Div. Macro & Div. Micro & Pol. Acc. & Pol. Macro \\\\",
        "\\midrule",
    ]
    for label, experiment in experiments:
        row = find_row(summary_rows, experiment, "val")
        div_macro = metric(row, "division_macro_f1")
        div_micro = metric(row, "division_micro_f1")
        pol_acc = metric(row, "polarity_accuracy")
        pol_macro = metric(row, "polarity_macro_f1")
        if experiment == "val_balanced":
            div_macro = f"\\textbf{{{div_macro}}}"
            div_micro = f"\\textbf{{{div_micro}}}"
        lines.append(f"{label} & {div_macro} & {div_micro} & {pol_acc} & {pol_macro} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Validation results for prompt selection. Division is evaluated as a multi-label task using macro/micro F1; polarity is evaluated as a three-way classification task.}",
            "\\label{tab:validation_prompt_selection}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def report_heldout_test_table(summary_rows: list[dict[str, Any]]) -> str:
    experiments = [
        ("Frequency prior", "frequency_baseline"),
        ("Qwen zero-shot", "qwen_zeroshot"),
        ("LoRA, train only", "test_balanced_train_only"),
        ("LoRA, train+val", "test_balanced_train_val"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & Div. Macro & Div. Micro & Pol. Acc. & Pol. Macro \\\\",
        "\\midrule",
    ]
    for label, experiment in experiments:
        row = find_row(summary_rows, experiment, "test")
        div_macro = metric(row, "division_macro_f1")
        div_micro = metric(row, "division_micro_f1")
        pol_acc = metric(row, "polarity_accuracy")
        pol_macro = metric(row, "polarity_macro_f1")
        if experiment == "test_balanced_train_val":
            div_macro = f"\\textbf{{{div_macro}}}"
            div_micro = f"\\textbf{{{div_micro}}}"
        if experiment == "test_balanced_train_only":
            pol_acc = f"\\textbf{{{pol_acc}}}"
            pol_macro = f"\\textbf{{{pol_macro}}}"
        lines.append(f"{label} & {div_macro} & {div_micro} & {pol_acc} & {pol_macro} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Held-out test results on the human-verified test set. Fine-tuning improves division recognition over zero-shot Qwen; merging train and validation improves division F1 but hurts polarity.}",
            "\\label{tab:heldout_test_results}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def report_test_division_table(division_rows: list[dict[str, Any]]) -> str:
    experiments = [
        ("Qwen zero-shot", "qwen_zeroshot"),
        ("LoRA train", "test_balanced_train_only"),
        ("LoRA train+val", "test_balanced_train_val"),
    ]
    rows_by_experiment = {
        experiment: find_row(division_rows, experiment, "test") for _, experiment in experiments
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Division & Qwen zero-shot & LoRA train & LoRA train+val \\\\",
        "\\midrule",
    ]
    for key in DIVISION_KEYS:
        label = short_division(key)
        col = f"{label}_f1"
        values = [metric(rows_by_experiment[experiment], col) for _, experiment in experiments]
        numeric_values = []
        for value in values:
            try:
                numeric_values.append(float(value))
            except ValueError:
                numeric_values.append(-1.0)
        best = max(numeric_values)
        formatted = [
            f"\\textbf{{{value}}}" if score == best and score >= 0 else value
            for value, score in zip(values, numeric_values)
        ]
        lines.append(f"{label} & {formatted[0]} & {formatted[1]} & {formatted[2]} \\\\")
    macro_values = [
        metric(find_row(summary_rows_cache, experiment, "test"), "division_macro_f1")
        for _, experiment in experiments
    ]
    numeric_macro_values = []
    for value in macro_values:
        try:
            numeric_macro_values.append(float(value))
        except ValueError:
            numeric_macro_values.append(-1.0)
    best_macro = max(numeric_macro_values)
    formatted_macro = [
        f"\\textbf{{{value}}}" if score == best_macro and score >= 0 else value
        for value, score in zip(macro_values, numeric_macro_values)
    ]
    lines.extend(
        [
            "\\midrule",
            f"Macro avg. & {formatted_macro[0]} & {formatted_macro[1]} & {formatted_macro[2]} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Per-division F1 on the human-verified held-out test set. The train+val model has the strongest division macro-F1, while the train-only model is more balanced across the two tasks.}",
            "\\label{tab:test_division_f1}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def report_test_polarity_table(polarity_rows: list[dict[str, Any]]) -> str:
    experiments = [
        ("Qwen zero-shot", "qwen_zeroshot"),
        ("LoRA train", "test_balanced_train_only"),
        ("LoRA train+val", "test_balanced_train_val"),
    ]
    rows_by_experiment = {
        experiment: find_row(polarity_rows, experiment, "test") for _, experiment in experiments
    }
    labels = [("Positive", "positive_f1"), ("Negative", "negative_f1"), ("Neutral", "neutral_f1")]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Polarity & Qwen zero-shot & LoRA train & LoRA train+val \\\\",
        "\\midrule",
    ]
    for label, col in labels:
        values = [metric(rows_by_experiment[experiment], col) for _, experiment in experiments]
        numeric_values = []
        for value in values:
            try:
                numeric_values.append(float(value))
            except ValueError:
                numeric_values.append(-1.0)
        best = max(numeric_values)
        formatted = [
            f"\\textbf{{{value}}}" if score == best and score >= 0 else value
            for value, score in zip(values, numeric_values)
        ]
        lines.append(f"{label} & {formatted[0]} & {formatted[1]} & {formatted[2]} \\\\")
    macro_values = [
        metric(find_row(summary_rows_cache, experiment, "test"), "polarity_macro_f1")
        for _, experiment in experiments
    ]
    numeric_macro_values = []
    for value in macro_values:
        try:
            numeric_macro_values.append(float(value))
        except ValueError:
            numeric_macro_values.append(-1.0)
    best_macro = max(numeric_macro_values)
    formatted_macro = [
        f"\\textbf{{{value}}}" if score == best_macro and score >= 0 else value
        for value, score in zip(macro_values, numeric_macro_values)
    ]
    lines.extend(
        [
            "\\midrule",
            f"Macro avg. & {formatted_macro[0]} & {formatted_macro[1]} & {formatted_macro[2]} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Per-polarity F1 on the human-verified held-out test set. The train-only LoRA model gives the best polarity macro-F1, mainly by preserving stronger positive and neutral performance.}",
            "\\label{tab:test_polarity_f1}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


summary_rows_cache: list[dict[str, Any]] = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        action="append",
        help="Metric file as NAME=PATH. Can be repeated. If omitted, known default result paths are used.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("report_tables"))
    parser.add_argument(
        "--format",
        choices=["all", "latex", "markdown", "csv"],
        default="all",
        help="Which table format(s) to write.",
    )
    args = parser.parse_args()

    summary_rows: list[dict[str, Any]] = []
    division_rows: list[dict[str, Any]] = []
    polarity_rows: list[dict[str, Any]] = []
    skipped = []

    for name, path in metric_paths(args).items():
        if not path.exists():
            skipped.append(str(path))
            continue
        metrics = read_json(path)
        for split in result_splits(metrics):
            summary_rows.append(summary_row(name, path, metrics, split))
            division_rows.append(division_row(name, path, metrics, split))
            polarity_rows.append(polarity_row(name, path, metrics, split))

    if not summary_rows:
        raise SystemExit("No metrics files found. Use --metric NAME=PATH or place JSON files in data/results/.")

    global summary_rows_cache
    summary_rows_cache = summary_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_cols = [
        "experiment",
        "split",
        "prompt_style",
        "division_macro_f1",
        "division_micro_f1",
        "polarity_accuracy",
        "polarity_macro_f1",
        "parse_failures",
    ]
    division_cols = ["experiment", "split"] + [
        f"{short_division(key)}_f1" for key in DIVISION_KEYS
    ]
    polarity_cols = ["experiment", "split", "positive_f1", "negative_f1", "neutral_f1"]

    if args.format in {"all", "csv"}:
        write_csv(args.output_dir / "experiment_summary.csv", summary_rows, list(summary_rows[0].keys()))
        write_csv(args.output_dir / "division_per_label.csv", division_rows, list(division_rows[0].keys()))
        write_csv(args.output_dir / "polarity_per_label.csv", polarity_rows, list(polarity_rows[0].keys()))

    if args.format in {"all", "markdown"}:
        write_text(args.output_dir / "experiment_summary.md", markdown_table(summary_rows, summary_cols))
        write_text(args.output_dir / "division_per_label_f1.md", markdown_table(division_rows, division_cols))
        write_text(args.output_dir / "polarity_per_label_f1.md", markdown_table(polarity_rows, polarity_cols))

    if args.format in {"all", "latex"}:
        write_text(
            args.output_dir / "experiment_summary.tex",
            report_validation_prompt_table(summary_rows) + "\n" + report_heldout_test_table(summary_rows),
        )
        write_text(
            args.output_dir / "division_per_label_f1.tex",
            report_test_division_table(division_rows),
        )
        write_text(
            args.output_dir / "polarity_per_label_f1.tex",
            report_test_polarity_table(polarity_rows),
        )

    if skipped:
        write_text(args.output_dir / "skipped_metrics.txt", "\n".join(skipped) + "\n")

    print(f"Wrote experiment tables to {args.output_dir}")
    if skipped:
        print(f"Skipped missing metric files: {len(skipped)}")


if __name__ == "__main__":
    main()
