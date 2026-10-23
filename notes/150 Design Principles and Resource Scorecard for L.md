# 150 - Why L exists, causal chain, design principles, resource scorecard (2026-10-22 IST)
Source of this note: the founder's stated motivation (heavy weights, heavy compute, power, heat, water, e-waste, RAM/GPU price pressure) turned into engineering targets. No claim here is a measured result unless labelled.

## Causal chain (what drives the cost)
1. Knowledge is stored in dense weights -> parameters x bits = memory (RAM/HBM size, price, number of chips).
2. Every token touches (nearly) all weights -> data movement + arithmetic per token. ESTIMATED (Horowitz, ISSCC 2014, 45 nm): DRAM read ~100x or more the energy of an fp32 multiply; int add ~30x cheaper than fp32 multiply. So DATA MOVEMENT is usually the larger cost, multiplies second. Removing multiplies helps; moving fewer bytes helps more.
3. Training cost ~ parameters x tokens x passes (+ optimizer state) -> big clusters, power, heat, cooling water, hardware turnover (e-waste).
4. Retraining from scratch to add knowledge repeats 3.

## Design principles for L (each tied to a cause)
P1 Move less data: small weights (ternary/low-bit), state kept on-chip, sparse access. (cause 2)
P2 Compute less per token: add/shift/compare instead of multiply; sparse activations; skip work when confident (never skip the state update). (causes 2,3)
P3 Knowledge outside dense weights: hash/associative memory and fast weights hold facts; a lookup costs one access, not a model pass. (causes 1,2)
P4 Learn without full retraining: sleep consolidation / continual learning, distillation, small student models. (cause 4)
P5 Run on modest, long-lived hardware (edge/FPGA/old GPUs): fewer chips, less e-waste. (cause 3)
P6 Measure what matters: joules per useful result, bytes moved, memory footprint, not parameter count alone.

## Honest limits
- Measured so far: L trains at ~2x the transformer's energy/token on a laptop GPU (compiled+bf16 both); GPU is the wrong target for P1/P2. Inference/long-context advantage is measured at small scale only.
- MatMul-free LM (arXiv 2406.02528) is slightly below Transformer++ on zero-shot averages at 370M-2.7B; efficiency claims at scale remain projections.
- Water/heat/e-waste follow from energy and chip count; we do not measure them directly. Report joules and device count instead.
- Evidence is toy/small scale (<10M params, bytes). No AGI claim.

## Resource scorecard (report for every candidate; label MEASURED / MODELLED / ESTIMATED)
params | weight bits | bytes moved per token | adds / multiplies per token | peak working memory | training joules to target bits/byte | inference joules/token (batch 1, long context) | needs GPU? | quality (bits/byte, 3 seeds) | continual-learning cost (joules to add a task)
