# Colab Qwen LoRA Training

Use this workflow to fine-tune Qwen2.5-VL on the current active tasks:

- NBA division context, multi-label.
- Sentiment polarity, 3-class.

`human_type__humor` and `human_type__highlight` are intentionally excluded from
training and evaluation because their distribution is too imbalanced for the
current project scope.

## 1. Setup

Use a GPU runtime.

```bash
pip install -U "transformers>=4.57.0" accelerate qwen-vl-utils pillow bitsandbytes peft tqdm scikit-learn
```

Make sure Colab has:

- `configs/label_schema.json`
- `images_jpg/`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `scripts/modeling/finetune_qwen_lora.py`
- `scripts/modeling/evaluate_qwen_lora.py`

## 2. Smoke Test

Run a tiny pass first. This checks that Colab can load Qwen, read the images,
format the labels, save an adapter, and evaluate generations. Treat these
numbers as pipeline diagnostics only while annotation review is still ongoing.
The default prompt does not include annotation-derived team context, because that
would leak the division label.

Optional shape check:

```bash
python3 scripts/modeling/finetune_qwen_lora.py \
  --split-dir data/processed \
  --train-limit 1 \
  --batch-size 1 \
  --load-in-4bit \
  --max-image-side 336 \
  --max-length 1024 \
  --debug-first-batch
```

```bash
python3 scripts/modeling/finetune_qwen_lora.py \
  --split-dir data/processed \
  --output-dir models/qwen_lora_smoke \
  --epochs 1 \
  --train-limit 16 \
  --batch-size 1 \
  --grad-accum 4 \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-image-side 336 \
  --max-length 1024
```

Then evaluate a few validation rows:

```bash
python3 scripts/modeling/evaluate_qwen_lora.py \
  --adapter models/qwen_lora_smoke \
  --split-dir data/processed \
  --split val \
  --limit 12 \
  --output data/results/qwen_lora_smoke_metrics.json \
  --load-in-4bit \
  --max-image-side 336
```

## 3. Pilot Training

If the smoke test works, run a short 1-epoch pilot on the current provisional
splits. The default batch size is 1 with gradient accumulation to fit Colab
GPUs.

```bash
python3 scripts/modeling/finetune_qwen_lora.py \
  --split-dir data/processed \
  --output-dir models/qwen_lora_division_polarity \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-image-side 448 \
  --max-length 1536
```

If Colab runs out of memory, reduce image and sequence size:

```bash
python3 scripts/modeling/finetune_qwen_lora.py \
  --split-dir data/processed \
  --output-dir models/qwen_lora_division_polarity \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-image-side 336 \
  --max-length 1024
```

## 4. Evaluation

After training:

```bash
python3 scripts/modeling/evaluate_qwen_lora.py \
  --adapter models/qwen_lora_division_polarity \
  --split-dir data/processed \
  --output data/results/qwen_lora_metrics.json \
  --load-in-4bit \
  --max-image-side 448
```

The output JSON contains validation/test metrics and raw generations for error
analysis.

## 5. Recall-Oriented Control Model

If the baseline prompt under-predicts multi-division labels, train a control
model with the recall-oriented prompt. Keep the other settings the same so the
comparison is interpretable.

```bash
python3 scripts/modeling/finetune_qwen_lora.py \
  --split-dir data/processed \
  --output-dir models/qwen_lora_recall_prompt \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-image-side 448 \
  --max-length 1536 \
  --prompt-style recall
```

Then evaluate on validation:

```bash
python3 scripts/modeling/evaluate_qwen_lora.py \
  --adapter models/qwen_lora_recall_prompt \
  --split-dir data/processed \
  --split val \
  --output data/results/qwen_lora_val_recall_prompt.json \
  --load-in-4bit \
  --max-image-side 448
```

Compare against the baseline no-team-context result:

- `data/results/qwen_lora_val_no_team_context_detailed.json`
- `data/results/qwen_lora_val_recall_prompt.json`

Use macro-F1, per-label recall, and the number of missing/extra divisions to
decide whether the recall prompt is actually better.

## 6. Balanced Prompt Control Model

If the recall-oriented prompt improves Atlantic recall but creates too many
Pacific false positives or over-predicts negative polarity, train the balanced
prompt model:

```bash
python3 scripts/modeling/finetune_qwen_lora.py \
  --split-dir data/processed \
  --output-dir models/qwen_lora_balanced_prompt \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-image-side 448 \
  --max-length 1536 \
  --prompt-style balanced
```

Evaluate on validation:

```bash
python3 scripts/modeling/evaluate_qwen_lora.py \
  --adapter models/qwen_lora_balanced_prompt \
  --split-dir data/processed \
  --split val \
  --output data/results/qwen_lora_val_balanced_prompt.json \
  --load-in-4bit \
  --max-image-side 448
```

Compare all three validation results:

- Baseline prompt: `data/results/qwen_lora_val_no_team_context_detailed.json`
- Recall prompt: `data/results/qwen_lora_val_recall_prompt.json`
- Balanced prompt: `data/results/qwen_lora_val_balanced_prompt.json`

Prefer the model with the best validation macro-F1 unless its error pattern is
clearly worse for the final report narrative.

## 7. Reporting

Use division per-label F1, division macro-F1, polarity accuracy, and polarity
macro-F1 in the final report. Also include qualitative error examples from the
human-verified test set, especially cases where the image carries information
that the Reddit text alone does not.

## 8. Final Train+Val Retraining

After choosing the final prompt and hyperparameters on validation, merge train
and validation for the final adapter. The held-out test split is not included.

```bash
python3 scripts/data_prep/build_train_val_split.py
```

Then retrain the final balanced model:

```bash
python3 scripts/modeling/finetune_qwen_lora.py \
  --split-dir data/processed \
  --train-file data/processed/train_val.csv \
  --output-dir models/qwen_lora_balanced_train_val_final \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-image-side 448 \
  --max-length 1536 \
  --prompt-style balanced
```

Evaluate once on the human-verified held-out test set:

```bash
python3 scripts/modeling/evaluate_qwen_lora.py \
  --adapter models/qwen_lora_balanced_train_val_final \
  --split-dir data/processed \
  --split test \
  --output data/results/qwen_lora_test_balanced_train_val_final.json \
  --load-in-4bit \
  --max-image-side 448
```
