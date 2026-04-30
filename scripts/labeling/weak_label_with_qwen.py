#!/usr/bin/env python3
"""Use Qwen2.5-VL to weak-label an annotation queue.

This is a convenience script for Colab. Weak labels are suggestions for
annotation and data triage; they should not be treated as the final test-set
ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from data_utils import division_columns, label_columns, load_schema, polarity_columns, read_csv, write_csv


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_image(path: str, max_image_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_image_side > 0:
        image.thumbnail((max_image_side, max_image_side))
    return image


def prompt(row: dict[str, str], include_meme_type: bool) -> str:
    type_text = ""
    if include_meme_type:
        type_text = '\n- "types": array containing any of "humor", "highlight", or empty if neither applies.'
    return f"""
Classify this NBA meme using both the image and Reddit text.

Return only compact JSON:
- "divisions": array of NBA divisions referenced by the meme. Multiple divisions are allowed.
- "polarity": exactly one of "positive", "negative", or "neutral".{type_text}

NBA divisions:
- Atlantic Division: Celtics, Nets, Knicks, 76ers, Raptors.
- Central Division: Bulls, Cavaliers, Pistons, Pacers, Bucks.
- Southeast Division: Hawks, Hornets, Heat, Magic, Wizards.
- Northwest Division: Nuggets, Timberwolves, Thunder, Trail Blazers, Jazz.
- Pacific Division: Warriors, Clippers, Lakers, Suns, Kings.
- Southwest Division: Mavericks, Rockets, Grizzlies, Pelicans, Spurs.

Reddit text:
{row.get("text", "")}
""".strip()


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


def load_model(args: argparse.Namespace):
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


def generate(model: Any, processor: Any, row: dict[str, str], args: argparse.Namespace) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": load_image(row["image_path"], args.max_image_side)},
                {"type": "text", "text": prompt(row, args.include_meme_type)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(
        model.device
    )
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, outputs)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def normalize_to_row(row: dict[str, str], parsed: dict[str, Any], raw_output: str, schema: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    out = dict(row)
    divisions = parsed.get("divisions", [])
    if isinstance(divisions, str):
        divisions = [part.strip() for part in re.split(r"[,;/|]", divisions) if part.strip()]
    if not isinstance(divisions, list):
        divisions = []
    division_set = {str(item).strip() for item in divisions}
    for label, col in zip(schema["division_labels"], division_columns(schema, "weak")):
        out[col] = "1" if label in division_set else "0"

    polarity = str(parsed.get("polarity", "")).strip().lower()
    for label, col in zip(schema["polarity_labels"], polarity_columns(schema, "weak")):
        out[col] = "1" if polarity == label else "0"

    if args.include_meme_type:
        types = parsed.get("types", [])
        if isinstance(types, str):
            types = [part.strip() for part in re.split(r"[,;/|]", types) if part.strip()]
        type_set = {str(item).strip().lower() for item in types if item}
        for label in schema.get("meme_type_labels", []):
            out[f"weak_type__{label}"] = "1" if label in type_set else "0"

    if not args.no_notes:
        out["weak_label_note"] = raw_output.strip()
    out["weak_label_status"] = "labeled"
    return out


def save(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    write_csv(path, rows, fieldnames)
    print(f"Saved checkpoint to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/label_schema.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--include-meme-type", action="store_true")
    parser.add_argument("--no-notes", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    schema = load_schema(args.schema)
    input_rows, input_fields = read_csv(args.input)
    output_rows = input_rows
    if args.output.exists():
        output_rows, input_fields = read_csv(args.output)

    weak_cols = label_columns(schema, "weak", include_meme_type=args.include_meme_type)
    fieldnames = list(dict.fromkeys(input_fields + weak_cols + ["weak_label_note", "weak_label_status"]))

    model, processor = load_model(args)
    processed = 0
    for idx, row in enumerate(tqdm(output_rows, desc="weak-labeling"), start=1):
        if args.limit and processed >= args.limit:
            break
        if args.skip_existing and row.get("weak_label_status") == "labeled":
            continue
        try:
            raw = generate(model, processor, row, args)
            parsed = parse_json(raw)
            output_rows[idx - 1] = normalize_to_row(row, parsed, raw, schema, args)
            print(f"[{idx}/{len(output_rows)}] labeled sample_id={row.get('sample_id')}")
        except Exception as exc:
            row["weak_label_status"] = "failed"
            row["weak_label_note"] = str(exc)
            output_rows[idx - 1] = row
            print(f"[{idx}/{len(output_rows)}] FAILED sample_id={row.get('sample_id')}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        processed += 1
        if args.save_every > 0 and processed % args.save_every == 0:
            save(args.output, output_rows, fieldnames)

    save(args.output, output_rows, fieldnames)


if __name__ == "__main__":
    main()
