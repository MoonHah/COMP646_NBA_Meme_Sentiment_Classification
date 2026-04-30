# Colab Qwen Weak Labeling

Install dependencies:

```bash
pip install -U "transformers>=4.57.0" accelerate qwen-vl-utils pillow bitsandbytes tqdm
```

Example run:

```bash
python3 scripts/labeling/weak_label_with_qwen.py \
  --input data/weak/full_ai_labeling_queue.csv \
  --output data/weak/full_ai_labeling_queue_labeled.csv \
  --save-every 10 \
  --load-in-4bit \
  --max-image-side 448
```

For lower memory:

```bash
python3 scripts/labeling/weak_label_with_qwen.py \
  --input data/weak/full_ai_labeling_queue.csv \
  --output data/weak/full_ai_labeling_queue_labeled.csv \
  --save-every 10 \
  --load-in-4bit \
  --max-image-side 336 \
  --max-new-tokens 96
```

The script resumes from the output CSV if it already exists and skips rows with
`weak_label_status = labeled`.
