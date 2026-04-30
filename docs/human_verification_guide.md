# Human Verification Guide

Use AI labels only as suggestions. For the held-out evaluation set, humans
should verify or correct:

- `human_division__Atlantic Division`
- `human_division__Central Division`
- `human_division__Southeast Division`
- `human_division__Northwest Division`
- `human_division__Pacific Division`
- `human_division__Southwest Division`
- `human_polarity__positive`
- `human_polarity__negative`
- `human_polarity__neutral`

After editing an annotation CSV, run:

```bash
python3 scripts/validation/validate_human_annotations.py --input data/annotation/human_test_annotation_queue.csv
```

Resolve rows with invalid binary values, missing images, or more than one
polarity label before using them for final results.
