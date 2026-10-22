---
tags: [decision, architecture, spiking]
created: 2026-10-07
status: decision recorded; plan waiting for approval
---
# 97 - Decision: drop the transformer base

Previous: [[96 - Claude Takeover Review of CPU Coder]]

**User decision (2026-10-07):** Qwen2.5-Coder is a transformer, so it will not be used as the base of our model.

## What this means
- The Qwen coder work (V1–V5 adapters, GGUFs, gateways, tests) stays **archived and untouched** on D:. Nothing is deleted.
- Our model must be **brain-inspired and non-transformer**: spiking/recurrent neurons, without attention.
- Inventory check of `experiments/` found GPT's existing brain modules:
  - LIF spiking experts (image tasks).
  - `growth_core.py`: append-only growing modules, cluster routing and self-healing (`growth/`).
  - A retrieval-based frontend selector.
- **No spiking language/coding model exists yet.** It has to be built from scratch.
- Our own Round 8 LM ([[25 - Round 8 Language Continual Learning (Phase 2)]]) also used attention blocks. Its principles carry over (columns, surprise routing, novelty growth); its attention trunk does not.

## Proposed plan (not started)
- **Step 0:** read GPT's spiking and growth code; write the spec of the spiking language model (BrainCoder).
- **Step 1:** a small spiking LM from scratch (LIF + recurrent state, no attention), trained on code text. It will be compared against a same-size transformer control.
- **Step 2:** connect GPT's growth modules (append-only skill columns, self-healing) and our surprise routing and novelty growth.
- **Step 3:** scale on the RTX 3050; add code-task and tool-use tests.

**Expectation:** it starts far below Qwen's coding ability. Tool calling and file copying come later.
