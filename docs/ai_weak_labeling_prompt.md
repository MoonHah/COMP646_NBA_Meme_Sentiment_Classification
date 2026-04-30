# AI Weak Labeling Prompt

Weak labeling should ask the VLM for compact JSON:

```json
{"divisions": ["Pacific Division"], "polarity": "negative"}
```

Allowed divisions:

- `Atlantic Division`
- `Central Division`
- `Southeast Division`
- `Northwest Division`
- `Pacific Division`
- `Southwest Division`

Allowed polarity values:

- `positive`
- `negative`
- `neutral`

Weak labels are not final ground truth. They are acceptable for bootstrapping
annotation and for training experiments only when final evaluation uses a
human-verified held-out set.
