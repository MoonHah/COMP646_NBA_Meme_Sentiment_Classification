# NBA Meme Multimodal Classification

This repository contains the code and processed data for our COMP 646 final
project on NBA meme classification. The final direction is intentionally scoped:
we fine-tune Qwen2.5-VL with LoRA and evaluate two outputs:

1. NBA division context: 6-label multi-label prediction.
2. Sentiment polarity: positive / negative / neutral.

Some older annotation files still contain `human_type__humor` and
`human_type__highlight`. We keep those columns for provenance, but they are not
part of the final model because their labels are too imbalanced for this stage
of the project.

## Current Source Of Truth

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `images_jpg/`
- `configs/label_schema.json`

Validate the active splits with:

```bash
python3 scripts/validation/validate_processed_splits.py
```

The validation summary is written to:

```text
data/processed/validated_split_summary.json
```

## Qwen LoRA Training

Colab instructions are in:

```text
docs/colab_qwen_lora_training.md
```

Training:

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

Evaluation:

```bash
python3 scripts/modeling/evaluate_qwen_lora.py \
  --adapter models/qwen_lora_division_polarity \
  --split-dir data/processed \
  --output data/results/qwen_lora_metrics.json \
  --load-in-4bit \
  --max-image-side 448
```

## Script Layout

The scripts are grouped by role so the final workflow is easier to follow:

- `scripts/common/`: shared CSV, JSON, and schema helpers.
- `scripts/data_prep/`: queue building, team-to-division conversion, split generation, and train+val merging.
- `scripts/labeling/`: Qwen/VLM weak-labeling utilities used as annotation aids.
- `scripts/modeling/`: Qwen2.5-VL LoRA training and evaluation.
- `scripts/validation/`: checks for processed splits and human annotation files.
- `scripts/reporting/`: LaTeX tables and optional report figures.
- `scripts/annotation_legacy/`: teammate-provided or earlier annotation-helper code kept for project history.

## Data Processing Utilities

These scripts are retained so the annotation and split pipeline can be rebuilt
or audited if needed:

- `scripts/data_prep/build_raw_annotation_queue.py`: create raw post CSV and a blank human annotation queue from `nbameme.json`.
- `scripts/data_prep/build_ai_labeling_queue.py`: create a queue for VLM weak labeling.
- `scripts/labeling/weak_label_with_qwen.py`: weak-label queue rows with Qwen2.5-VL in Colab.
- `scripts/data_prep/build_human_test_queue.py`: sample a roughly balanced human verification queue from weak labels.
- `scripts/data_prep/convert_team_annotations_to_divisions.py`: convert teammate-provided team labels into division columns.
- `scripts/data_prep/build_team_division_weak_queue.py`: build a team-grounded weak-label queue from existing team annotations.
- `scripts/data_prep/normalize_final_annotations.py`: normalize a final annotation CSV to the project schema.
- `scripts/data_prep/build_splits_from_annotations.py`: rebuild `train.csv`, `val.csv`, and `test.csv` from final annotations.
- `scripts/validation/validate_human_annotations.py`: check human annotation files before modeling.

The final report should emphasize that held-out evaluation labels are
human-verified. Weak Qwen labels are useful for speed and triage, but they are
not treated as final ground truth.

## Report Figures

Generate report-ready SVG figures from the processed splits:

```bash
python3 scripts/reporting/make_report_visualizations.py \
  --split-dir data/processed \
  --output-dir report_figures
```

After final evaluation, pass the metrics JSON:

```bash
python3 scripts/reporting/make_report_visualizations.py \
  --split-dir data/processed \
  --metrics data/results/qwen_lora_test_balanced_final.json \
  --output-dir report_figures
```

This creates split/label distribution plots, per-division F1 plots, polarity
confusion plots, missing/extra division error plots, and an HTML qualitative
example gallery.

Generate comparison tables from evaluation JSON files:

```bash
python3 scripts/reporting/make_experiment_tables.py \
  --metric baseline=data/results/qwen_lora_test_balanced_final.json \
  --metric train_val=data/results/qwen_lora_test_balanced_train_val.json \
  --format latex \
  --output-dir report_tables
```

The script can write Markdown, CSV, or LaTeX tables for overall metrics and
per-label F1 scores.

## Final Retraining

After selecting the final prompt on validation, merge train and validation while
leaving test held out:

```bash
python3 scripts/data_prep/build_train_val_split.py
```

Train with:

```bash
python3 scripts/modeling/finetune_qwen_lora.py \
  --split-dir data/processed \
  --train-file data/processed/train_val.csv \
  --output-dir models/qwen_lora_balanced_train_val_final \
  --prompt-style balanced \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-image-side 448 \
  --max-length 1536
```
