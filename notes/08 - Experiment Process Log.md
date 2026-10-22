

2026-10-06: CPU coding pilot trained V1/V2, preserved failures, diagnosed dynamic INT8 regression, packaged native CPU Q8 baseline/trained weights, tested exact tool calls and OpenCode file reading. Original-prompt Q8 base 6/6; trained 4/6. OpenCode agent gates remain weak. Defaults baseline, detailed note86, assets cpu-coder.

2026-10-06: Downloaded full pinned CodeAlpaca/Glaive datasets on D:, retained source/license discrepancy, filtered349train72validation disjoint conversations; started fresh lower-LR CPU LoRA V3 with periodic checkpoints; note88.

2026-10-06: Downloaded full pinned CodeAlpaca/Glaive datasets on D:, retained source/license discrepancy, filtered349train72validation disjoint conversations; started fresh lower-LR CPU LoRA V3 with periodic checkpoints; note88.

2026-10-06: V3 web-data run completed349updates/~900sec; loss0.833→0.631; nativeCPUQ8 packaged. Full V3 original retention2/6, coding7/8,tool5/11,stopping10/10. Frozen-base phase router retention6/6,coding7/8,tool8/11,stopping10/10; exactcopy1/2, multiline failure preserved. Fixed OpenCode parent-config isolation; audited actual model endpoints. Default stays base; notes89–90 and CPU-coder-v3 assets saved.

2026-10-06: Began V4 two separate frozen-base skill adapters for tool/code discrimination and exact tool-result copying. Prepared112selection64copy examples, separate validation/test identifiers, CPU training started; note91.


## 2026-10-06–07: targeted CPU training V4

112 selection updates, 128 copying updates, 48 varied-copy refinement updates. Frozen base unchanged, confirmed SHA-256; no GPU. Tested exact tools, code discrimination, retention, independent coding/generic tools, original/varied copying, and actual OpenCode reads. Preserved failed gateway attempts, locale launch errors, controls, logs and datasets. Final gains: 12/12 scoped selection, 8/8 discrimination, 9/12 diverse copying; real OpenCode 1/2, multiline empty. Default stays base. See [[92 - Targeted V4 Results and OpenCode Root Causes]].

## 2026-10-07: V5 targeted follow-up started
Controlled actual-read probe: auto and disabled tool calling both repeated a read; removing schemas produced malformed contents. Prepared two CPU adapter continuations, 56 train / 8 validation / 16 fresh test rows each. Copying uses random mixtures of varied lines with actual OpenCode-style string arguments; selection combines unknown simple schemas and JavaScript zero/false/empty cases. Eight earlier training examples replayed per skill. Prior tests remain regression tests, excluded from training. Default backends paused temporarily for training memory; restore script prepared. Artifacts on D: targeted-v5.


V5 training and evaluation completed. Two 56-update continuations, base hashes unchanged: True. Actual OpenCode 0/2. See [[94 - V5 Targeted Follow-up Results]] for measured gains, regressions and limits. Default remains base.

V5 final correction: new selector failed actual OpenCode 0/2; retained V4 selector with V5 copier, which passed 3/3 actual exact reads. Broad tools V5 4/11 regress; baseline stays default. JavaScript independent 4/4 and original retention 6/6. See [[95 - Latest CPU Skill Training Status]].

Final service check: baseline 8771, V4 selector 8781, V5 copy 8795, hybrid gateway 8797 healthy. Stopped only verified obsolete experiment backends/gateways; saved cleanup evidence. Default 8770 proxy continues baseline. Separate V5 selector checkpoint retained on disk. Hybrid launcher checked; obsolete V4 copy and all-V5 gateway no longer occupy memory.



## 2026-10-07: stop and handoff to Claude chat
User explicitly stopped experiment work and requested Claude chat upload package. Created D:\AI\SpikingBrain-study\claude-handoff-2026-10-07 with project-upload.zip, separate runnable-models.zip, handoff, upload prompt and integrity manifests. No further training run. Local credentials/private OpenCode state/caches/dependencies excluded; large model files kept separately. Packages verified. No direct Claude chat connector is available, so uploads require the user; no transfer claimed.

Handoff verified: 2,786 project files, 121,269,307-byte project ZIP and 1,593,205,306-byte runnable-model ZIP; both ZIP CRC checks passed. Additional CLAUDE_REVIEW.md (about 41KB) and upload instructions prepared. CPU model services paused; previews left intact. User will upload to Claude chat; no direct remote transfer performed.

