# Web Dataset Acquisition and CPU Training V3

User authorized public dataset acquisition and actual model training. All downloads, source cards, license text, prepared data, checkpoints, runtime logs and caches remain on D:.

Sources: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k and https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 . Exact revisions and SHA256 in datasets-web/sources.json. Downloaded full JSON files: CodeAlpaca 8,057,896 bytes and Glaive 271,190,065 bytes. Viewer subset requests were also saved while full download was pending; final training uses full downloads.

Licenses: CodeAlpaca card declares CC BY 4.0, original DATA_LICENSE is Attribution-NonCommercial 4.0. Conservatively mark resulting research adapter noncommercial and preserve both records. Glaive source card declares Apache-2.0. No claim of unrestricted commercial rights. Public examples are synthetic and can be wrong; syntax/schema validation is not functional validation.

Preparation: language identification by explicit prompt keyword; Python syntax checks; tool schema and required-argument checks; safe literal parser for source JSON-like tool strings; no dataset code executed during preparation. Native chat-template tool calls and tool responses used, with loss only on final assistant target. Deduplicate messages by SHA256 and split at conversation groups; no group or exact message overlaps between train and validation. Cap at 512 tokenizer tokens without silent truncation. Seed1903. CodeAlpaca Express coverage sparse. Selected counts:

{
  "train_examples": 349,
  "validation_examples": 72,
  "train_categories": {
    "html": 24,
    "node": 24,
    "c": 24,
    "react": 24,
    "css": 24,
    "tool-call": 64,
    "java": 24,
    "tool-result": 64,
    "python": 24,
    "javascript": 24,
    "cpp": 24,
    "express": 5
  },
  "validation_categories": {
    "css": 5,
    "c": 5,
    "tool-call": 12,
    "python": 5,
    "java": 5,
    "tool-result": 12,
    "html": 5,
    "express": 3,
    "javascript": 5,
    "cpp": 5,
    "react": 5,
    "node": 5
  },
  "rejections": {
    "code_unsuitable": 12447,
    "python_syntax": 360,
    "over_512_tokens": 909,
    "tool_parse_or_schema": 4685,
    "duplicate": 5643
  }
}

Training V3: new rank4 alpha8 adapter on unchanged 494,032,768-parameter base, 270,336 trainable coefficients; CPU float32,6threads,one pass, AdamW learning rate0.00005, grad clipping1.0; checkpoints every64steps. Baseline and final validation report token-weighted target loss by category. This validation loss alone does not establish coding correctness. Previous regressions are preserved, not overwritten. Own inference servers paused for training. OpenCode provider remains configured but unavailable until server restart.

Status: training underway; not yet a quality improvement claim. Final results will be recorded in note89 after native Q8 packaging and held-out tool testing.
