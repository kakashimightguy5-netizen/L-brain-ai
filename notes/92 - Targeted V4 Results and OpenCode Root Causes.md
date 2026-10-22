# Targeted V4 Results and OpenCode Root Causes

Completed 2026-10-06 to 2026-10-07. All training, datasets, checkpoints, models, logs and notes were stored on D:. CPU-only training and inference; no GPU used. Installed runtimes on C: were read/run; this does not control Windows background writes.

## Results

| Check | Before | Latest | Scope |
| --- | --- | --- | --- |
| Exact read/glob/grep selection and arguments | 10/12 base | 12/12 selection | Synthetic held-out identifiers, shared task families |
| Code vs tool discrimination | 6/8 base | 8/8 selection | Synthetic bounded code tests |
| Original retention | 6/6 base | 6/6 selection | Small original suite |
| Independent Python | 4/4 base | 4/4 selection | Separate earlier suite |
| Independent JavaScript | 3/4 base | 3/4 selection | Zero-value firstOrNull weakness remains |
| Generic Glaive tool calls | 8/11 base | 5/11 selection | Broader tool schemas regress |
| Original patterned exact copying | 0/16 base | 16/16 copy and refined copy | Same first-line pattern in training and test |
| Varied exact copying | 0/12 original copy | 9/12 refined copy | New IDs across eight content styles; styles partly shared |
| Real OpenCode exact file copying | — | 1/2 specialized pipeline | Actual completed read tools; single-line passed, multiline returned no text |

No matched benchmark against Qwen 9B or 4B–48B competitors was performed. These results do not establish general coding or agent superiority.

## Training and lineage

Qwen2.5-Coder-0.5B-Instruct: 494,032,768 frozen base parameters. Two separate rank-4 LoRA skills, each 270,336 trainable coefficients. Selection trained 112 updates; original copying 128 updates on 64 examples; varied copying 48 updates on 40 new examples and eight replay examples. Total 288 updates and 216 distinct training examples. The refined copy warm-started the original copy adapter. V4 selection and original copy began fresh from the base, without the earlier V3 web-data checkpoint. Data were assistant-authored synthetic examples, not downloaded public training data. Base Apache-2.0 source records remain in the experiment folder.

Selection took 377.797 seconds, original copying 735.377 seconds, refinement 270.506 seconds. Refined validation loss fell from 0.2563 at warm start to 0.07992; loss is not task accuracy. The base safetensors and base Q8 hashes were verified unchanged (base-integrity.json). Small retention tests do not prove universal absence of forgetting. The old frontend selector is a separate component; this is a transformer adapter experiment, not an independently trained spiking LLM.

## Root causes and integration limits

The first copying curriculum was too narrow: every training/test first line used a MiXeD prefix. Its 16/16 score failed to transfer to varied content (0/12). Refining with varied case, punctuation, indentation, JSON, HTML, blank lines and Windows headers improved this to 9/12 while preserving 16/16 patterned cases. Three varied cases still fail.

OpenCode adds long system prompts, rich tool definitions, quoted user text and file metadata. Requests omitted temperature; the experimental gateway sets zero temperature, decodes quoted user strings, scopes read/glob/grep schemas, and compacts exact-copy context while retaining the actual read result. Controlled argument-string vs object and relative vs Windows header tests did not resolve the old copying failure, so neither is established as the sole cause.

Earlier experimental gateways and OpenCode retries failed, sometimes repeatedly issuing reads. The final copying phase blocks further generated tool calls after a read; it does not fabricate or repair file answers. This prevented loops but the actual multiline test still produced no text. Single-line success is pipeline-specific, not proof that the model works with unmodified OpenCode. Generic schemas fall back to the baseline. Earlier Bun missing-locale startup errors and our fixed missing Path import are infrastructure failures, separate from model scoring; raw logs are preserved.

## Current use and next proposal

The existing default remains the base model. The specialized gateway is an opt-in experiment, not a promoted replacement. All Q8 models run on CPU. Each merged Q8 variant is about 531 MB on disk; running multiple specialist backends duplicates the base weights in memory. Two adapters together contain 540,672 trained coefficients, not a standalone 540k-parameter LLM.

Next: focus training on multiline completion and remaining diverse-copy failures, then evaluate unseen content families and real OpenCode workflows. Improve the independent JavaScript zero-value failure and broad-schema routing before widening deployment. Keep a fixed untouched test set and compare CPU speed, memory and correctness under matching conditions. Do not infer success from lowering loss alone.

Artifacts: D:\AI\SpikingBrain-study\experiments\cpu-coder\targeted-v4. Reproducible datasets, adapters, scripts, raw failures and summaries are archived under SpikingBrain Assets/cpu-coder-v4 in this vault. Large model files are referenced by path and SHA-256 rather than duplicated.
