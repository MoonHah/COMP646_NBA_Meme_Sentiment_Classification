#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-VL to predict division and polarity labels.

This script is intended for Colab/GPU. It treats the task as supervised
generation: image + Reddit text -> compact JSON labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_SCHEMA = Path("configs/label_schema.json")
DEFAULT_OUTPUT_DIR = Path("models/qwen_lora_division_polarity")
PROMPT_STYLES = ("baseline", "recall", "balanced")
SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "labels",
    "token_type_ids",
    "input_token_type",
    "position_ids",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_image(path: str, max_image_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_image_side > 0:
        image.thumbnail((max_image_side, max_image_side))
    return image


def squeeze_sequence_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return a 1D sequence tensor even if the processor adds extra singleton dims."""
    while value.dim() > 1 and value.shape[0] == 1:
        value = value.squeeze(0)
    return value


def is_sequence_key(key: str) -> bool:
    return key in SEQUENCE_KEYS or "token_type" in key


def division_columns(schema: dict[str, Any]) -> list[str]:
    return [f"human_division__{label}" for label in schema["division_labels"]]


def polarity_columns(schema: dict[str, Any]) -> list[str]:
    return [f"human_polarity__{label}" for label in schema["polarity_labels"]]


def target_json(row: dict[str, str], schema: dict[str, Any]) -> str:
    divisions = [
        label
        for label in schema["division_labels"]
        if row[f"human_division__{label}"] == "1"
    ]
    polarity_labels = list(schema["polarity_labels"])
    polarity_values = [row[f"human_polarity__{label}"] for label in polarity_labels]
    if polarity_values.count("1") != 1:
        raise ValueError(f"Invalid polarity labels for sample_id={row.get('sample_id')}")
    polarity = polarity_labels[polarity_values.index("1")]
    return json.dumps({"divisions": divisions, "polarity": polarity}, ensure_ascii=False)


def prompt_instructions(prompt_style: str) -> str:
    if prompt_style == "recall":
        return """
Important labeling rules:
- First identify every NBA team, player-team association, logo, jersey, rivalry, matchup, or explicit team nickname visible in the image or Reddit text.
- Return every corresponding division. Do not return only the most salient team if multiple teams are referenced.
- Pay special attention to Atlantic Division teams: Celtics, Nets, Knicks, 76ers, and Raptors.
- Do not default to Southeast Division or Pacific Division unless there is visible or textual evidence.
- For polarity, criticism, mockery, disappointment, or sarcastic praise should usually be negative rather than neutral.
""".strip()
    if prompt_style == "balanced":
        return """
Important labeling rules:
- Identify all NBA teams, player-team associations, logos, jerseys, matchups, rivalries, or explicit team nicknames visible in the image or Reddit text.
- Include every corresponding division when there is clear visual or textual evidence.
- Do not output only the most salient division if another division is clearly referenced.
- Do not guess a division from general NBA context. If a team/division is not clearly supported, leave it out.
- For polarity, use negative for clear criticism, mockery, disappointment, or hostile sarcasm.
- Use neutral when the meme is mostly informational, ambiguous, or the sentiment target is unclear.
""".strip()
    return ""


def user_prompt(
    row: dict[str, str],
    include_team_context: bool = False,
    prompt_style: str = "baseline",
) -> str:
    teams = row.get("parsed_teams") or row.get("primary_team_or_context") or "unknown"
    team_context = ""
    if include_team_context:
        team_context = f"\nKnown team context from annotation: {teams}\n"
    style_instructions = prompt_instructions(prompt_style)
    if style_instructions:
        style_instructions = f"\n{style_instructions}\n"
    return f"""
Classify this NBA meme.

Use the image and Reddit text. Return only JSON with:
- "divisions": an array of NBA divisions referenced by the meme. It may contain multiple divisions.
- "polarity": exactly one of "positive", "negative", or "neutral".

NBA divisions:
- Atlantic Division: Celtics, Nets, Knicks, 76ers, Raptors.
- Central Division: Bulls, Cavaliers, Pistons, Pacers, Bucks.
- Southeast Division: Hawks, Hornets, Heat, Magic, Wizards.
- Northwest Division: Nuggets, Timberwolves, Thunder, Trail Blazers, Jazz.
- Pacific Division: Warriors, Clippers, Lakers, Suns, Kings.
- Southwest Division: Mavericks, Rockets, Grizzlies, Pelicans, Spurs.
{team_context}
{style_instructions}

Reddit text:
{row.get("text", "")}
""".strip()


class MemeSFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        schema: dict[str, Any],
        processor: Any,
        max_image_side: int,
        max_length: int,
        include_team_context: bool,
        prompt_style: str,
    ) -> None:
        self.rows = rows
        self.schema = schema
        self.processor = processor
        self.max_image_side = max_image_side
        self.max_length = max_length
        self.include_team_context = include_team_context
        self.prompt_style = prompt_style

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        row = self.rows[idx]
        image = load_image(row["image_path"], self.max_image_side)
        answer = target_json(row, self.schema)
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt(row, self.include_team_context, self.prompt_style)},
        ]
        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(prompt_messages)
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {}
        for key, value in full_inputs.items():
            if is_sequence_key(key):
                item[key] = squeeze_sequence_tensor(value)
            else:
                item[key] = value
        labels = item["input_ids"].clone()
        prompt_ids = squeeze_sequence_tensor(prompt_inputs["input_ids"])
        prompt_len = min(prompt_ids.shape[0], labels.shape[0])
        labels[:prompt_len] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        item["labels"] = labels
        return item


def collate_features(features: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    keys = features[0].keys()
    for key in keys:
        tensors = [feature[key] for feature in features]
        if is_sequence_key(key):
            tensors = [squeeze_sequence_tensor(tensor) for tensor in tensors]
            pad_value = -100 if key == "labels" else pad_token_id
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=pad_value
            )
        elif key.startswith("pixel_values") or key.endswith("grid_thw"):
            batch[key] = torch.cat(tensors, dim=0)
        else:
            batch[key] = torch.stack(tensors)
    return batch


def load_model_and_processor(args: argparse.Namespace):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def write_training_metadata(args: argparse.Namespace, num_train_rows: int) -> None:
    metadata = {
        "model": args.model,
        "split_dir": str(args.split_dir),
        "train_file": str(args.train_file) if args.train_file is not None else None,
        "schema": str(args.schema),
        "num_train_rows": num_train_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "max_image_side": args.max_image_side,
        "load_in_4bit": args.load_in_4bit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "include_team_context": args.include_team_context,
        "prompt_style": args.prompt_style,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="Optional explicit training CSV. Defaults to <split-dir>/train.csv.",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--save-every-steps", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=0, help="Use only the first N training rows for smoke tests.")
    parser.add_argument("--debug-first-batch", action="store_true", help="Print first batch tensor shapes and exit.")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="baseline")
    parser.add_argument(
        "--include-team-context",
        action="store_true",
        help="Leak annotation-derived team context into the prompt. Use only for debugging, not fair evaluation.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    train_path = args.train_file if args.train_file is not None else args.split_dir / "train.csv"
    train_rows = read_csv(train_path)
    print(f"Loaded training rows from {train_path}")
    if args.train_limit > 0:
        train_rows = train_rows[: args.train_limit]
        print(f"Using first {len(train_rows)} training rows for a smoke test.")
    model, processor = load_model_and_processor(args)
    dataset = MemeSFTDataset(
        train_rows,
        schema,
        processor,
        args.max_image_side,
        args.max_length,
        args.include_team_context,
        args.prompt_style,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda features: collate_features(features, processor.tokenizer.pad_token_id),
    )

    if args.debug_first_batch:
        first_batch = next(iter(loader))
        print("First batch tensor shapes:")
        for key, value in first_batch.items():
            print(f"  {key}: {tuple(value.shape)}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    global_step = 0
    total_steps = math.ceil(len(loader) * args.epochs / args.grad_accum)
    progress = tqdm(total=total_steps, desc="training")
    optimizer.zero_grad(set_to_none=True)

    for _epoch in range(args.epochs):
        pending_updates = 0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            pending_updates += 1
            if pending_updates == args.grad_accum:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_updates = 0
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")
                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
        if pending_updates:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)

    progress.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    write_training_metadata(args, len(train_rows))
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
