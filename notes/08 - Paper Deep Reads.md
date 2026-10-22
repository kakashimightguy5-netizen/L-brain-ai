---
tags: [research, papers, self-healing]
created: 2026-10-06
---
# Paper Deep Reads

Back to [[00 - Concept and Roadmap]] · Results: [[07 - Experiment Results]]

*Read via full-text pages (methods, numbers, limitations), not abstracts. Some pages blocked access (noted).*

## Self-healing / self-correction
### 1. Huang et al., *LLMs Cannot Self-Correct Reasoning Yet* (ICLR 2024)
- **Intrinsic** self-correction = fixing answers with no outside feedback.
- GPT-4 on GSM8K: **95.5% → 89.0%** after self-correction; GPT-3.5: 75.9% → 74.7%. Models change right answers to wrong ones.
- Earlier "gains" came from **oracle labels** (being told which answers were wrong).
- Multi-agent debate (83.2%) lost to simple majority voting (85.3%) at equal cost.
- **Rule:** use external feedback (code execution, tools, verifiers).
- ✅ **Reproduced in our E2.**

### 2. Kumar et al., *SCoRe: Training LMs to Self-Correct via RL* (Google DeepMind, 2024)
- Training on correction examples fails because of **distribution shift** (fixes another model's errors, not its own) and **behaviour collapse** (learns to answer well first and make no real edits).
- Fix: 2-stage RL. Stage 1 keeps first attempts close to the base model; stage 2 adds a **progress bonus** α·(reward₂ − reward₁) that penalises breaking a right answer.
- MATH: base −11.2 pts after correction → **SCoRe +4.4 pts** (60.0% → 64.4%); HumanEval +12.2 pts.
- Limits: only 2 turns, two separate training runs.
- ➡️ For us: if we train a self-correcting stage, reward **improvement**, not just final correctness.

### 3. Wang et al., *WISE: Lifelong Model Editing* (NeurIPS 2024)
- "Impossible triangle": reliability, generalisation, locality.
- **Main memory (frozen) + side memory (copied FFN)** + an activation-based **router**; edits split into random-subspace **shards**, then **merged** (Ties-Merge).
- LLaMA-2-7B, 1000 sequential edits: **WISE 0.83** vs GRACE 0.67 vs ROME/MEMIT ~0.01.
- ➡️ This is our hippocampus + router design, proven at 7B scale. ✅ E2 matches.

### 4. Gupta et al., *Model Editing at Scale → Gradual and Catastrophic Forgetting* (ACL Findings 2024)
- ROME collapses after **100–1,000** sequential edits, MEMIT after 1,400–4,000; some single "disabling edits" break the model immediately.
- ✅ **Reproduced in E2** (direct edits: 97% → 52% old-skill accuracy).

### 5. Amazon, *Self-Healing through Error Detection, Attribution and Retraining* (voice assistant)
- Pipeline: detect user-perceived defects → attribute to the responsible model → add as training examples → retrain.
- +3.1% F1 in 6/9 domains; **3 domains got worse** where attribution was weak (F1 0.31–0.42).
- ➡️ Self-healing is only as good as the **error attribution**. Bad blame means bad fixes.

### 6. Chen et al., *Self-Healing Robust Neural Networks via Closed-Loop Control* (JMLR 2022)
- Treats a network as a control system: per-layer controllers detect and correct off-manifold activations caused by attacks or noise at inference time, without knowing the attack type.
- ➡️ Possible "immune system" layer for robustness. *(Only the abstract was readable.)*

### Brain analogue
- **Anterior cingulate cortex**: error detection (error-related negativity signal).
- **Cerebellum**: error-based correction of predictions.
- **Dopamine**: reward prediction error.
- ➡️ New blueprint module: **Error Monitor** (ACC + cerebellum).

## Core architecture papers (full reads)
### TRM, *Less is More: Recursive Reasoning with Tiny Networks* (Samsung, 2025)
- State: input x, answer y, latent z. Each cycle: update z n=6 times, then update y; T=3 cycles; **deep supervision** (gradient only on the last cycle, so ~42 effective layers).
- **Halting:** a single Q-head predicts "is the current answer correct?". This removes HRM's extra forward pass.
- 5–7M params, 2 layers: Sudoku-Extreme **87%** (HRM 55%), Maze-Hard 85%, ARC-AGI-1 **45%**, ARC-AGI-2 8%.
- A smaller network generalised *better* (less memorising).
- Limits: puzzles with fixed-size inputs; untested on language.
- ✅ Our E3 halting head follows this design.

### Genomic bottleneck (Koulakov/Zador, PNAS 2024)
- g-network takes binary IDs (Gray codes) of pre- and post-synaptic neurons and outputs the weight.
- MNIST **322× compression → 94%** innate (full 98%); 1038× → 79%; CIFAR-10 92× → 76%; Atari Beam Rider ~492×.
- Compression gives a **hot start but not faster learning**; transfer only works for complex tasks; one genome per layer.
- 🟡 Our E5 got only 3.8× at small scale.

### Sleep Replay Consolidation (Tadros et al., *Nature Communications* 2022)
- After training: convert to a spiking network, feed **noise** input, apply **Hebbian STDP** (strengthen co-active connections, weaken others).
- Incremental MNIST: 50% → ~60%; with replay buffer (iCaRL): 65.5% → **78.1%**.
- Mechanism: decorrelates class representations and re-activates old task neurons.
- Limits: weaker than modern replay alone; shallow networks; recent tasks slightly hurt.
- ➡️ We used replay-based sleep in E2/E6; a noise-driven STDP sleep is a Round 2 option.

### HiCL (Hippocampal-Inspired Continual Learning, 2025)
- Dentate gyrus = top-k **5% sparse** codes (pattern separation); CA3 = completion; routing by **cosine similarity to task prototypes** (no learned gate); one expert active per input.
- Split CIFAR-10: 92.4% task-incremental, 65.6% class-incremental; **38–152 MFLOPs vs 219–867** for baselines.
- ➡️ Candidate fix for our E6 routing problem.

### Memory Layers at Scale (Meta, 2024)
- Trainable key-value lookup adds parameters **without adding FLOPs**; up to 128B memory parameters, 1T tokens.
- Beats dense models with **2× the compute**; strongest on factual QA.

## Blocked pages (couldn't read)
- *Nature Communications* corticohippocampal hybrid network (rate-limited)
- PMC brain-inspired energy review (bot check)
- CVPR 2025 SEMA self-expanding adapters (403)
- PNAS genomic bottleneck page (403; read the bioRxiv full version instead)
- PNAS brain communication-energy paper (403; only the title-level finding is used)

## Sources
- https://arxiv.org/pdf/2310.01798 · https://arxiv.org/html/2409.12917 · https://arxiv.org/html/2405.14768v3 · https://arxiv.org/html/2401.07453v3
- https://cdn.amazon.science/35/ce/ffc1fa284de2815409a64c0f1dd3/self-healing-through-error-detection-attribution-and-retraining.pdf · https://arxiv.org/abs/2206.12963
- https://arxiv.org/html/2510.04871 · https://www.biorxiv.org/content/10.1101/2021.03.16.435261v2.full.pdf · https://www.nature.com/articles/s41467-022-34938-7 · https://arxiv.org/html/2508.16651v2 · https://arxiv.org/abs/2412.09764
