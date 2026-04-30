#!/usr/bin/env python3
"""Evaluate a Qwen2.5-VL LoRA adapter on division and polarity labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from finetune_qwen_lora import PROMPT_STYLES, load_schema, polarity_columns, user_prompt


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_ADAPTER = Path("models/qwen_lora_division_polarity")
DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_SCHEMA = Path("configs/label_schema.json")
DEFAULT_OUTPUT = Path("data/results/qwen_lora_metrics.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_adapter_metadata(adapter: Path) -> dict[str, Any]:
    path = adapter / "training_config.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_image(path: str, max_image_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_image_side > 0:
        image.thumbnail((max_image_side, max_image_side))
    return image


def division_labels(schema: dict[str, Any]) -> list[str]:
    return list(schema["division_labels"])


def division_columns(schema: dict[str, Any]) -> list[str]:
    return [f"human_division__{label}" for label in schema["division_labels"]]


def true_division_matrix(rows: list[dict[str, str]], schema: dict[str, Any]) -> list[list[int]]:
    cols = division_columns(schema)
    return [[int(row[col]) for col in cols] for row in rows]


def true_polarity_labels(rows: list[dict[str, str]], schema: dict[str, Any]) -> list[int]:
    cols = polarity_columns(schema)
    labels = []
    for row in rows:
        positives = [idx for idx, col in enumerate(cols) if row[col] == "1"]
        if len(positives) != 1:
            raise ValueError(f"Invalid polarity labels for sample_id={row.get('sample_id')}")
        labels.append(positives[0])
    return labels


def gold_divisions(row: dict[str, str], schema: dict[str, Any]) -> list[str]:
    return [label for label in division_labels(schema) if row[f"human_division__{label}"] == "1"]


def gold_polarity(row: dict[str, str], schema: dict[str, Any]) -> str:
    for label in schema["polarity_labels"]:
        if row[f"human_polarity__{label}"] == "1":
            return label
    return "invalid"


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def normalize_prediction(parsed: dict[str, Any], schema: dict[str, Any]) -> tuple[list[int], int]:
    divisions = parsed.get("divisions", [])
    if isinstance(divisions, str):
        divisions = [part.strip() for part in re.split(r"[,;/|]", divisions) if part.strip()]
    if not isinstance(divisions, list):
        divisions = []
    normalized_divisions = {str(item).strip() for item in divisions}
    y_div = [1 if label in normalized_divisions else 0 for label in division_labels(schema)]

    polarity_values = list(schema["polarity_labels"])
    polarity = str(parsed.get("polarity", "neutral")).strip().lower()
    if polarity not in polarity_values:
        polarity = "neutral"
    y_pol = polarity_values.index(polarity)
    return y_div, y_pol


def labels_from_prediction(y_div: list[int], y_pol: int, schema: dict[str, Any]) -> tuple[list[str], str]:
    pred_divisions = [label for label, value in zip(division_labels(schema), y_div) if value == 1]
    pred_polarity = list(schema["polarity_labels"])[y_pol]
    return pred_divisions, pred_polarity


def load_model(args: argparse.Namespace):
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    processor = AutoProcessor.from_pretrained(args.adapter)
    model.eval()
    return model, processor


def generate_one(
    model: Any,
    processor: Any,
    row: dict[str, str],
    max_image_side: int,
    max_new_tokens: int,
    include_team_context: bool,
    prompt_style: str,
) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": load_image(row["image_path"], max_image_side)},
                {"type": "text", "text": user_prompt(row, include_team_context, prompt_style)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def evaluate_split(model: Any, processor: Any, rows: list[dict[str, str]], schema: dict[str, Any], args: argparse.Namespace):
    y_div_true = true_division_matrix(rows, schema)
    y_pol_true = true_polarity_labels(rows, schema)
    y_div_pred = []
    y_pol_pred = []
    generations = []

    for row in tqdm(rows, desc="evaluating"):
        output = generate_one(
            model,
            processor,
            row,
            args.max_image_side,
            args.max_new_tokens,
            args.include_team_context,
            args.prompt_style,
        )
        parsed = parse_json(output)
        div_pred, pol_pred = normalize_prediction(parsed, schema)
        y_div_pred.append(div_pred)
        y_pol_pred.append(pol_pred)
        pred_divisions, pred_polarity = labels_from_prediction(div_pred, pol_pred, schema)
        generations.append(
            {
                "sample_id": row["sample_id"],
                "raw_output": output,
                "parsed": parsed,
                "parse_ok": bool(parsed) and "divisions" in parsed and "polarity" in parsed,
                "gold_divisions": gold_divisions(row, schema),
                "pred_divisions": pred_divisions,
                "gold_polarity": gold_polarity(row, schema),
                "pred_polarity": pred_polarity,
            }
        )

    div_names = division_columns(schema)
    pol_names = polarity_columns(schema)
    return {
        "division": {
            "micro_f1": f1_score(y_div_true, y_div_pred, average="micro", zero_division=0),
            "macro_f1": f1_score(y_div_true, y_div_pred, average="macro", zero_division=0),
            "per_label": classification_report(
                y_div_true,
                y_div_pred,
                target_names=div_names,
                output_dict=True,
                zero_division=0,
            ),
        },
        "polarity": {
            "accuracy": accuracy_score(y_pol_true, y_pol_pred),
            "macro_f1": f1_score(y_pol_true, y_pol_pred, average="macro", zero_division=0),
            "per_label": classification_report(
                y_pol_true,
                y_pol_pred,
                target_names=pol_names,
                output_dict=True,
                zero_division=0,
            ),
        },
        "generations": generations,
        "parse_failures": sum(1 for item in generations if not item["parse_ok"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows per split for smoke tests.")
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default=None)
    parser.add_argument(
        "--include-team-context",
        action="store_true",
        help="Use annotation-derived team context in the prompt. Debug only; this leaks division evidence.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    metadata = read_adapter_metadata(args.adapter)
    if args.prompt_style is None:
        args.prompt_style = metadata.get("prompt_style", "baseline")
        print(f"Using prompt_style={args.prompt_style}")

    schema = load_schema(args.schema)
    model, processor = load_model(args)
    results: dict[str, Any] = {
        "model": args.model,
        "adapter": str(args.adapter),
        "prompt_style": args.prompt_style,
        "adapter_training_config": metadata,
        "active_tasks": {"division": "multi-label", "polarity": "3-class"},
    }

    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        rows = read_csv(args.split_dir / f"{split}.csv")
        if args.limit > 0:
            rows = rows[: args.limit]
            print(f"Evaluating first {len(rows)} rows from {split}.")
        results[split] = evaluate_split(model, processor, rows, schema, args)

    write_json(args.output, results)
    print(f"Wrote {args.output}")
    if "test" in results:
        print(
            "Test division macro-F1:",
            round(results["test"]["division"]["macro_f1"], 4),
            "polarity accuracy:",
            round(results["test"]["polarity"]["accuracy"], 4),
            "polarity macro-F1:",
            round(results["test"]["polarity"]["macro_f1"], 4),
        )


if __name__ == "__main__":
    main()
