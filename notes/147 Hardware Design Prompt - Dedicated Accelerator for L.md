# 147 - Prompt: dedicated accelerator for L (paste into any strong AI / use as a project brief) - 2026-10-22 IST

---
ROLE: You are a skeptical computer-architecture researcher (digital design, ML accelerators, FPGA/ASIC). You do not flatter, you do not invent numbers. Label every figure MEASURED (from the data I give), MODELLED (computed, show the formula) or ESTIMATED (from a cited source). If you do not know, say so.

CONTEXT: I am a solo hardware founder building "L", a brain-inspired, attention-free language model, and I want a dedicated accelerator for it. On a laptop GPU (RTX 3050 6GB) L trains at about 2x the energy per token of a size-matched transformer, even after both got torch.compile + bf16 (MEASURED, B=16, T=256: transformer 0.41 mJ/token, 48.7 ms/step, 314 kernels; L 0.84 mJ/token, 114 ms/step, 1298 kernels). Eager: 0.91 vs 2.18 mJ/token. L is better in quality at 9M parameters (0.872 vs 0.913 bits/byte, small-scale evidence only) and uses 2-30x less inference energy at long context/batching (MEASURED in my lab, small models). No AGI claim is made.

L's computation (L5, d=128, 4 layers, 4 heads, ~0.94M params; wide variant d=192 ternary):
1. Diagonal linear recurrence per channel (scan over time), spiking activations = integer spike counts (round + clamp, straight-through gradient in training).
2. Delta-rule fast-weight memory: state S_t = a*S_{t-1} + b*(v - a*S*k) k^T, a rank-1 update + matvec read per token, chunked (chunk 64) for training.
3. Depthwise conv delay line (4 taps).
4. Hippocampus: hash-based exact recall (content-addressed lookup) boosting logits.
5. Weights can be ternary {-1,0,+1} (multiplier-free matrices). Activations low-bit integers.
6. Training needs backprop through time; local learning rules I tested cost +14-20% bits/byte.
Why a GPU is bad for it (my hypothesis, partly measured): work is memory-bound elementwise + sequential scan + random hash lookups, not big GEMMs; ~1300 kernel launches/step; sparsity and ternary are not exploitable on GPUs.

TASK: Design a dedicated accelerator for L, in this order:
A. Profile model: for one token of inference and one training step, list ops, bytes moved, arithmetic per byte, state sizes (bits), for d=128 fp and d=192 ternary. Show formulas.
B. Microarchitecture: (1) pipelined scan/recurrence unit, (2) delta-rule rank-1 update + read unit, (3) ternary add-only matvec array, (4) event-driven spike handling, (5) hash/CAM memory unit for recall, (6) on-chip SRAM layout so state never leaves the chip. Give dataflow diagram in text, block sizes, clock assumptions.
C. Inference first, training second: what extra hardware does BPTT need (state checkpointing/recompute, SRAM size)? Is training on-chip realistic, or should training stay on GPU and only inference move to the chip? Give a reasoned recommendation.
D. Staged path for a solo founder: (i) cycle-accurate software simulator in Python/C++ with an energy model, (ii) FPGA prototype (Sipeed Tang Nano 20K: ~20K LUT, ~103.5 KB block SRAM, external SDRAM; also name a bigger cheap FPGA if this one is too small), (iii) only then ASIC/shuttle. For each stage: deliverable, cost range (ESTIMATED with source), measurable pass/fail.
E. Fair comparison: compare against (a) the transformer on a GPU, AND (b) a transformer on equally custom hardware (e.g. fixed-function attention/KV accelerators). State what result would falsify the claim that L's chip wins.
F. Risks: what could make this a waste (e.g. memory bandwidth for hash tables, quality loss from ternary/low-bit, tiny-scale evidence not holding at 100M+ params).
OUTPUT: concise sections A-F, tables for numbers, a final "first 3 things to do this month" list. Prior-art honesty: name neuromorphic/SSM/RWKV/Mamba accelerators and spiking chips (Loihi, TrueNorth, SpiNNaker, etc.) that overlap, and say what is genuinely different.
---

Notes: use measured profile (next lab step: kernel profile of the 1298 kernels) to replace the hypothesis in "Why a GPU is bad". Hardware scope is parked under the software-only research directive; this prompt prepares the design stage, it does not claim any hardware result.
