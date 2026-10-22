# Targeted Skill Adapters V4 — Protocol

User approved targeted training after V3 failures. Base stays frozen and default. Task artifacts remain D:.

Two independent rank4 alpha8 adapters, 270,336 coefficients each, fresh initialization on the same unchanged494,032,768-parameter Qwen base. CPUfloat32,6threads,AdamW0.0001,clip1. Selection:112examples/onepass (72read/glob/grep requests,40code requests including zero/false edge cases and tool schemas present even for some code requests). Copy:64examples/twopasses, exact1–4line file copying from OpenCode-style read-result wrappers. Validation20selection/12copy; test20selection/16copy. Split seeds4101/4201/4301; content identifiers disjoint but task templates shared. These are assistant-authored targeted examples based on saved real failure formats, not a newly fetched web dataset or actual historical execution traces. Test comparisons must not claim unseen task-family generalization.

Source code and dataset files under D:\AI\SpikingBrain-study\experiments\cpu-coder\targeted-v4 and make_targeted_v4.py/train_targeted_v4.py. Runtime0GPU. Every64steps checkpoints, separate validation losses before/after. Planned tests: exact tool name/args; executable functions including0,false,empty arrays; exact case/punctuation/multiline copying; previous retention/newcoding and V3 held-out tools; isolated OpenCode directory avoids parent-config override. No default promotion based on loss alone. Baseline and old models remain available. Any router gains must be distinguished from native-model gains.

Status: training underway. Final results pending; no success claim yet.


Completed 2026-10-07. See [[92 - Targeted V4 Results and OpenCode Root Causes]]; varied-copy improvement is partial and OpenCode multiline still fails.
