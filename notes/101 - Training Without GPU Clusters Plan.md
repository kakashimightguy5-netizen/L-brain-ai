---
tags: [L, training, no-gpu, decoupled-learning, plan]
created: 2026-10-07
status: plan recorded; first experiment (L-0b) queued after L-0
---
# 101 - Training L without GPU clusters: the plan

Previous: [[100 - LLM Problem Atlas]] · [[98 - Architecture L Charter]]

**User goal (2026-10-07):** break the belief that models need millions of GPUs. L must run AND train without GPUs, or at least without huge clusters. This needs a way to separate layers, even though each layer depends on the earlier ones.

## Honest starting point
- **Running small models on a CPU works.** We measured Qwen2.5-Coder-0.5B at about 66 tokens/s on the laptop CPU (notes 92–95).
- **Training large models without GPUs does not work today.** The cause is **backpropagation locking**:
  - forward locking: each layer waits for the previous layers on the way forward;
  - backward locking: each layer waits for the later layers on the way back;
  - update locking: no layer can update until the full forward and backward pass is done.

  As a result, all layers must be in memory and updated together, which requires big, tightly connected GPU clusters.

## Five routes, combined in L
| # | Route | Brain analogue | Evidence so far |
|---|---|---|---|
| 1 | **Width separation:** skill columns train independently on a frozen base, in parallel on separate CPUs or PCs | cortical columns | Ours, Round 8: no forgetting, matches joint training ([[25 - Round 8 Language Continual Learning (Phase 2)]]) |
| 2 | **Depth separation:** each layer has its own local goal (predict the next byte from its own state), with a stop-gradient between layers, plus a cheap, delayed signal from the layer above | local plasticity, predictive coding | Literature: local losses work for vision but lose accuracy on language; forward-forward and predictive coding remain small-scale |
| 3 | **Growth instead of retraining:** train small, consolidate, add layers or columns with the old parts frozen | development | Ours: developmental order principle (Round 2/3) |
| 4 | **Low-communication training on many ordinary machines:** sync every few hundred steps (DiLoCo-style) | — | INTELLECT-1: 10B model trained across the internet |
| 5 | **CPU-native arithmetic:** integer spike counts × ternary weights, so most multiplications become additions | sparse spikes | BitNet b1.58 runs on CPU; spike-count neurons are already in L-0 |

## Next experiment: L-0b, "decoupled layers"
- **Variants:**
  - L with normal end-to-end backprop;
  - L with **only local per-layer losses** (gradients stopped between layers);
  - L with local losses + a delayed hint from the layer above.
- **Setup:** same size, data and steps.
- **What it measures:** the accuracy cost of breaking the layer dependency, and whether layers can then train in separate processes.
- **Success criterion:** decoupled L stays within a small gap of end-to-end L. That would be the first evidence that L can train without a GPU cluster.
