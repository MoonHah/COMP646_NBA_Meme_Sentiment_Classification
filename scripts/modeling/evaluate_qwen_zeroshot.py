#!/usr/bin/env python3
"""Evaluate base Qwen2.5-VL without LoRA fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from evaluate_qwen_lora import DEFAULT_MODEL, evaluate_split, load_schema, read_csv, write_json
from finetune_qwen_lora import PROMPT_STYLES


DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_SCHEMA = Path("configs/label_schema.json")
DEFAULT_OUTPUT = Path("data/results/qwen_zeroshot_test.json")


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows per split.")
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="balanced")
    parser.add_argument(
        "--include-team-context",
        action="store_true",
        help="Debug only; leaks annotation-derived team context into the prompt.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    schema = load_schema(args.schema)
    model, processor = load_model_and_processor(args)
    results: dict[str, Any] = {
        "model": args.model,
        "adapter": "",
        "prompt_style": args.prompt_style,
        "active_tasks": {"division": "multi-label", "polarity": "3-class"},
        "baseline_config": {"type": "zero_shot_qwen"},
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
