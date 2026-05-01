#!/usr/bin/env python3
"""Evaluate simple train-prior baselines for division and polarity labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_SCHEMA = Path("configs/label_schema.json")
DEFAULT_OUTPUT = Path("data/results/frequency_baseline_test.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def division_labels(schema: dict[str, Any]) -> list[str]:
    return list(schema["division_labels"])


def polarity_labels(schema: dict[str, Any]) -> list[str]:
    return list(schema["polarity_labels"])


def division_columns(schema: dict[str, Any]) -> list[str]:
    return [f"human_division__{label}" for label in division_labels(schema)]


def polarity_columns(schema: dict[str, Any]) -> list[str]:
    return [f"human_polarity__{label}" for label in polarity_labels(schema)]


def true_division_matrix(rows: list[dict[str, str]], schema: dict[str, Any]) -> list[list[int]]:
    cols = division_columns(schema)
    return [[int(row[col]) for col in cols] for row in rows]


def true_polarity_labels(rows: list[dict[str, str]], schema: dict[str, Any]) -> list[int]:
    cols = polarity_columns(schema)
    y_true = []
    for row in rows:
        positives = [idx for idx, col in enumerate(cols) if row[col] == "1"]
        if len(positives) != 1:
            raise ValueError(f"Invalid polarity labels for sample_id={row.get('sample_id')}")
        y_true.append(positives[0])
    return y_true


def gold_divisions(row: dict[str, str], schema: dict[str, Any]) -> list[str]:
    return [label for label in division_labels(schema) if row[f"human_division__{label}"] == "1"]


def gold_polarity(row: dict[str, str], schema: dict[str, Any]) -> str:
    for label in polarity_labels(schema):
        if row[f"human_polarity__{label}"] == "1":
            return label
    return "invalid"


def train_priors(rows: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
    divs = division_labels(schema)
    pols = polarity_labels(schema)
    div_counts = {label: sum(row[f"human_division__{label}"] == "1" for row in rows) for label in divs}
    pol_counts = {label: sum(row[f"human_polarity__{label}"] == "1" for row in rows) for label in pols}
    total_div_positives = sum(div_counts.values())
    avg_divisions_per_row = total_div_positives / max(len(rows), 1)
    return {
        "division_counts": div_counts,
        "polarity_counts": pol_counts,
        "avg_divisions_per_row": avg_divisions_per_row,
    }


def choose_divisions(priors: dict[str, Any], schema: dict[str, Any], strategy: str) -> list[str]:
    counts = priors["division_counts"]
    ranked = sorted(division_labels(schema), key=lambda label: (-counts[label], label))
    if strategy == "top1":
        return ranked[:1]
    if strategy == "top_k_mean":
        k = max(1, round(float(priors["avg_divisions_per_row"])))
        return ranked[:k]
    raise ValueError(f"Unsupported division strategy: {strategy}")


def choose_polarity(priors: dict[str, Any]) -> str:
    counts = priors["polarity_counts"]
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_from_pr(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def multilabel_report(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    target_names: list[str],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    f1_values = []
    weighted_f1_sum = 0.0
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0

    for idx, name in enumerate(target_names):
        tp = sum(true[idx] == 1 and pred[idx] == 1 for true, pred in zip(y_true, y_pred))
        fp = sum(true[idx] == 0 and pred[idx] == 1 for true, pred in zip(y_true, y_pred))
        fn = sum(true[idx] == 1 and pred[idx] == 0 for true, pred in zip(y_true, y_pred))
        support = sum(true[idx] == 1 for true in y_true)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = f1_from_pr(precision, recall)
        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": float(support),
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["support"] += support
        f1_values.append(f1)
        weighted_f1_sum += f1 * support
        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support

    micro_precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    micro_recall = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    micro_f1 = f1_from_pr(micro_precision, micro_recall)
    macro_f1 = safe_div(sum(f1_values), len(f1_values))
    macro_precision = safe_div(sum(item["precision"] for item in report.values()), len(target_names))
    macro_recall = safe_div(sum(item["recall"] for item in report.values()), len(target_names))
    support = float(totals["support"])
    report["micro avg"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1-score": micro_f1,
        "support": support,
    }
    report["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": support,
    }
    report["weighted avg"] = {
        "precision": safe_div(weighted_precision_sum, support),
        "recall": safe_div(weighted_recall_sum, support),
        "f1-score": safe_div(weighted_f1_sum, support),
        "support": support,
    }
    return report


def multiclass_report(
    y_true: list[int],
    y_pred: list[int],
    target_names: list[str],
) -> dict[str, dict[str, float]]:
    one_hot_true = [[1 if label == idx else 0 for idx in range(len(target_names))] for label in y_true]
    one_hot_pred = [[1 if label == idx else 0 for idx in range(len(target_names))] for label in y_pred]
    return multilabel_report(one_hot_true, one_hot_pred, target_names)


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    return safe_div(sum(true == pred for true, pred in zip(y_true, y_pred)), len(y_true))


def evaluate_split(
    rows: list[dict[str, str]],
    schema: dict[str, Any],
    predicted_divisions: list[str],
    predicted_polarity: str,
) -> dict[str, Any]:
    divs = division_labels(schema)
    pols = polarity_labels(schema)
    y_div_true = true_division_matrix(rows, schema)
    y_pol_true = true_polarity_labels(rows, schema)
    y_div_pred = [[1 if label in predicted_divisions else 0 for label in divs] for _ in rows]
    y_pol_pred = [pols.index(predicted_polarity) for _ in rows]
    div_report = multilabel_report(y_div_true, y_div_pred, division_columns(schema))
    pol_report = multiclass_report(y_pol_true, y_pol_pred, polarity_columns(schema))

    generations = []
    for row in rows:
        generations.append(
            {
                "sample_id": row["sample_id"],
                "raw_output": json.dumps(
                    {"divisions": predicted_divisions, "polarity": predicted_polarity}
                ),
                "parsed": {"divisions": predicted_divisions, "polarity": predicted_polarity},
                "parse_ok": True,
                "gold_divisions": gold_divisions(row, schema),
                "pred_divisions": predicted_divisions,
                "gold_polarity": gold_polarity(row, schema),
                "pred_polarity": predicted_polarity,
            }
        )

    return {
        "division": {
            "micro_f1": div_report["micro avg"]["f1-score"],
            "macro_f1": div_report["macro avg"]["f1-score"],
            "per_label": div_report,
        },
        "polarity": {
            "accuracy": accuracy(y_pol_true, y_pol_pred),
            "macro_f1": pol_report["macro avg"]["f1-score"],
            "per_label": pol_report,
        },
        "generations": generations,
        "parse_failures": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--division-strategy", choices=["top1", "top_k_mean"], default="top_k_mean")
    args = parser.parse_args()

    schema = read_json(args.schema)
    train_rows = read_csv(args.split_dir / f"{args.train_split}.csv")
    priors = train_priors(train_rows, schema)
    predicted_divisions = choose_divisions(priors, schema, args.division_strategy)
    predicted_polarity = choose_polarity(priors)

    results: dict[str, Any] = {
        "model": "frequency_baseline",
        "active_tasks": {"division": "multi-label", "polarity": "3-class"},
        "baseline_config": {
            "train_split": args.train_split,
            "division_strategy": args.division_strategy,
            "predicted_divisions": predicted_divisions,
            "predicted_polarity": predicted_polarity,
            "train_priors": priors,
        },
    }
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        rows = read_csv(args.split_dir / f"{split}.csv")
        results[split] = evaluate_split(rows, schema, predicted_divisions, predicted_polarity)

    write_json(args.output, results)
    print(f"Wrote {args.output}")
    for split in splits:
        result = results[split]
        print(
            split,
            "division macro-F1:",
            round(result["division"]["macro_f1"], 4),
            "division micro-F1:",
            round(result["division"]["micro_f1"], 4),
            "polarity accuracy:",
            round(result["polarity"]["accuracy"], 4),
            "polarity macro-F1:",
            round(result["polarity"]["macro_f1"], 4),
        )


if __name__ == "__main__":
    main()
