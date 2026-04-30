# Annotation Codebook

Active labels for the final model:

- Division context: mark every NBA division clearly referenced by the meme.
- Polarity: choose exactly one of `positive`, `negative`, or `neutral`.

Division is multi-label because a meme can compare or reference teams from more
than one division. Polarity is single-label because each sample should have one
dominant stance.

Older files may include `human_type__humor` and `human_type__highlight`; these
columns can remain in CSVs for provenance, but they are not used by the current
Qwen LoRA training/evaluation scripts.

Use `human_flag__needs_discussion = 1` when the label cannot be resolved quickly.
Use `human_flag__not_nba_relevant = 1` for samples that should be excluded from
training and final metrics.
