# Web-data CPU Training V3 — Results

## Outcome
Completed actual CPU LoRA training using downloaded CodeAlpaca and Glaive data. Selected 349 training examples,72 conversation-group-disjoint validation examples; strict generation validation excludes three repeated user prompts,leaving69. Training plus baseline/final validation took 900.1 seconds (~15 minutes). Every requested language/framework has selected examples; Express has only5 training examples. Full public downloads are retained on D:, but only the selected subset was trained in this run. Base494,032,768 parameters; trainable270,336 rank4 LoRA coefficients. Fresh adapter trained on unchanged frozen base with lower learning rate0.00005. The original 19,935 frontend selector parameters and old model checkpoints were not replaced.

Validation loss: base 0.832960, trained 0.630911, relative reduction 24.3%; every category improves teacher-forced target loss. These are loss scores, not accuracy percentages. The72-example loss set has three last-user prompts also present in otherwise different training conversations; strict generation tests remove them. Selected corpus is tiny versus full datasets and is synthetic/not comprehensively verified. Token count preflight initially counted the return object rather than input_ids; assertion stopped before optimizer updates. Corrected, regenerated subset, rejected909 overlength candidates, saved failure log. No silent truncation. See note88 and manifests.

## Generated-answer checks
| Check | Base Q8 | Full trained V3 Q8 | Experimental phase router |
|---|---:|---:|---:|
| Previous retention checks | 6/6 | 2/6 | 6/6 |
| New bounded Python functions | 4/4 | 4/4 | 4/4 |
| New bounded JavaScript functions | 3/4 | 3/4 | 3/4 |
| Strict held-out exact tool calls | 8/11 | 5/11 | 8/11 |
| Stop after tool result, nonempty answer | 0/10 | 10/10 | 10/10 |

The stopping row does not establish factual correctness or grounding: e.g. a generic answer can satisfy it. Three of11 tool names were absent from training. The new coding probes are independent prompts, not a standardized benchmark or proof their underlying concepts were unseen. Java,C,C++,React,HTML,CSS,Node and Express receive training/validation-loss coverage, not claimed executable functional benchmark accuracy here. Compiler availability was not changed. Both models fail JS firstOrNull on zero because they use xs[0] || null. Full V3 under the old XML-oriented prompt sometimes emits invented calls for code tasks, so it is not a safe universal replacement. Lower loss did not imply better task accuracy.

## OpenCode and inference
Merged V3 into FP16 GGUF then Q8_0 via pinned official llama.cpp Windows CPU b11429. V3 Q8 size531,067,872bytes (~506MiB). Observed model generation speeds across short held-out cases: 57.5–73.8 tokens/s, median 65.9. Separate prompt processing timings/raw output are saved. Measurements reflect short local CPU tests, with occasional concurrent agent requests; no matched speed comparison with larger models. GPU layers0, CPU-only binary. Measured resident working sets about676MiB base and670MiB V3, roughly1.3GiB combined at capture; dynamic usage can vary with context and workload.

OpenCode provider isolation correction: initial V3/router custom configs were overridden by parent project's opencode.json, so initial smoke logs do not count as trained-model evidence. Moved each experimental CLI test to separate sibling D: workspaces outside the base config's parent. Backend audit verified base8771 before calls and V3 8773 after tool results. All configs/cache/state/data/temp remain D:.

Actual isolated full V3 agent fails to produce an executable read call. Experimental phase router retains base for code/tool choice and uses V3 for post-tool replies. Original router reads then changes marker case; exact-answer fails. Adding an explicit file-copy instruction gets one new single-line mixed-case marker exactly right. A second multiline file fails with duplication/reader line numbers. An additional tool-result normalization prototype removes reader metadata before model inference but still fails capitalization/underscores; its failure is preserved, not repaired in scoring. Exact answer tests are1/2 for the verbatim router, not general agent success. No target contents are fabricated or supplied to the model except through actual tool results; prompt guidance and routing are orchestration, not new learned capability. The normalization prototype is not the selected runtime.

## Current runnable choices
Default stays unchanged base CPU Q8: start-cpu.ps1 then opencode-local.ps1. Experimental full trained GGUF at models/coder-trained-v3-q8.gguf, base remains models/coder-base-q8.gguf. For experimental phase router use start-research-v3.ps1 then opencode-router-verbatim.ps1. Read/glob/grep only; shell/edit actions denied in pilot config; step cap3. Base endpoint8771 and default adapter8770; experimental trained server8773 and verbatim router8777. Completed failing proxies8774,8776,8778 stopped. Both active models run without GPU. The phase router is a two-model system, not a single improved model. Optional runtime does not prove superiority to4B–48B models or robust autonomous programming.

Sources/attribution: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k ; https://github.com/sahil280114/codealpaca/blob/master/DATA_LICENSE ; https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 . CodeAlpaca card says CC BY4.0 but original data license says noncommercial; treat this adapter as noncommercial research and retain attribution. Glaive card declares Apache2.0. Exact pinned source revisions, original cards/license text, source hashes and selected data preserved. No C: task artifacts created; existing installed interpreters/libraries read normally.

## Next evidence needed
Use targeted coding-vs-tool discrimination examples and real OpenCode read/result trajectories, stronger answer-grounding checks, and tested language-specific examples. Separate adapters by skill or compare conservative checkpoints to avoid contaminating base code behavior. Increase balanced verified data once those quality gates work; more raw synthetic examples alone are not a demonstrated remedy. No superiority claim and no universal adapter promotion.
