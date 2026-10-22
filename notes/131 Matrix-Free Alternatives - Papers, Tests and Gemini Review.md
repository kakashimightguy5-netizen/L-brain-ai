# 131 - Matrix-free alternatives: papers, tests, outside review (2026-10-11/12 IST)
Prompt: "throw away the maths and matrices and look for alternatives". Position: mathematics cannot be removed (any computer computes); the target is the dense floating-point matrix multiply, which is ~77% of L5's weights (720,896 of 936,357).

## Papers read (what they actually show)
- Differentiable logic-gate networks, convolutional (arXiv 2411.04732): fixed random wiring, each node learns one of 16 two-input gates. CIFAR-10 86.29% with 61M gates (~29x fewer gates than a binarised-weight net at 86.28%); 16M-gate version 80.17% at 24 ns/image on a large FPGA. Limits: images/MNIST only; expensive training; small models vary run to run.
- Recurrent logic-gate networks (arXiv 2508.06097): WMT'14 En-De BLEU 5.00 vs transformer 5.98, 40.8M params, 16-token sentences; no FPGA measurement in the paper.
- Differentiable weightless (LUT) networks (arXiv 2410.11112): MNIST 97.8% at 2.5 nJ vs FINN 5445 nJ (their numbers); CIFAR-10 only 57.4%. Tables in the paper have internal inconsistencies.
- NeuraLUT (arXiv 2403.00849): dense sub-networks hidden inside LUTs; MNIST 96% in 12 ns; small embedded tasks only.
- Tsetlin machine compression (ETHEREAL, arXiv 2502.05640): propositional-logic learning, tiny microcontroller tasks.
- MatMul-free LM (arXiv 2406.02528): ternary BitLinear + element-wise GRU; zero-shot average 40.3/46.2/49.9 vs Transformer++ 41.1/48.0/50.7 at 370M/1.3B/2.7B (slightly BELOW); no perplexity; crossover with transformer is a projection (~1e23 FLOPs); FPGA numbers are a single-core prototype with projections. Correction to an outside review that called this "matching perplexity at billion scale".
- Hebbian local-learning language paper (arXiv 2503.02057): abstract only, no benchmarks. Treat as unproven.

## My tests (MEASURED unless stated; CPU)
- Gate LM, 8 prior bytes, py_big val slice: soft 3.62 bits/byte (24.6 kbit modelled storage); hard (deployable) 5.67. Dense MLP references: H=32 -> 3.19 (92.7 kbit); H=64 -> 2.97 (199.7 kbit). Gate model fails as deployable for now.
- Hashed n-gram counts (order 5, 4 MB train): 3.65 bits/byte; hash-collision limited, storage figure is dense worst case. Not a fair verdict.
- Matmul-free spiking cell (`matmulfree_cell.py`): structural tests pass (binary outputs, integer fast weights, accumulate-only path equals dense emulation, streaming equals one-shot, state independent of T, 12/12 params get gradients). MODELLED ops: ~1.5k adds vs 7.4k dense MACs per step at ~29% spike rate.
- Delayed-copy training (8 random bits, 1000 steps): original recipe 51.4% (chance 50). With static variance-preserving scale folded into thresholds: 61.5%. Plus 5x LR on thresholds: 63.9%. At 3000 steps: fast weights on 63.1% vs off 59.3% (seed 1). Float weights (ternary ablation, delay 3): 57.2%. Delay 1: 63.1%. Delay 0 (identity), 600 steps: ternary 73.9%, float 79.7%.
- Reading: the recipe change helped but the cell is far from solving even identity copy. Ternarisation is NOT the main bottleneck (float is no better); the spiking/threshold training path is. Nothing here supports a language claim yet.

## Outside review (Gemini) and my verdicts
- Normalise before quantising and use higher LR on thresholds: tried, helped (51 -> 64%), not sufficient.
- STE/Gumbel annealing for the gate soft-to-hard gap: plausible, not yet tried. Published gate results show small hard-vs-soft loss, so my 3.62 -> 5.67 gap is likely a recipe flaw.
- Generative-replay consolidation of fast weights: plausible, unverified for spiking/quantised models (reviewer flagged this too).
- Latent-search/JEPA: only after the base cell learns reliably.

## Next
1. Find why the cell cannot copy identity: surrogate sharpness, optimiser/LR, hidden width, readout. Start from float weights + spikes, then add ternary.
2. Only then swap into L5's dense layers and run the 1M-parameter bits/byte benchmark under the 0.5% gate.
3. Retry gate LM with the hard choice in the forward pass (STE) as a bounded experiment.

## Update (2026-10-12 IST): isolation of the spiking-cell failure (MEASURED, CPU, delayed copy of 8 random bits, chance 50%)
- Bare float+spike layer, delay 0: 97-100% for surrogate sharpness 2/5/10 and LR 3e-3..3e-2 -> surrogate gradient and optimiser are NOT the problem.
- Delay 3, bare layer: 57-65% (linear or MLP readout, H=48 or 256, tanh control 69%). Learnable decays 64%. Stacking 2 or 3 leaky spiking layers 66%/66%. => leaky integrators (even stacked, 600 steps) cannot hold an exact lag.
- Shift-register input window (6 taps, causal) feeding a spiking layer: 100%; plus a leaky layer 98.9%. => an explicit delay line fixes the lag. Limit: only reaches `taps` steps back; longer range needs associative (fast-weight / hash) memory.
- Full MatMul-free cell, delay 0 (float weights): fast-weight path ON 79.7% vs OFF 90.1% -> the fast-weight path interferes with plain copying; needs gating/ordering work.
- Full cell with 6 taps, delay 3: float weights 99.9%; ternary weights 64% at lr 1e-2 (unstable, neurons die, hid_rate falls to 6%); ternary with H=128, lr 3e-3 and cosine decay: 90.8%/90.3% (two seeds), still short of 95%. => ternarisation does cost accuracy once the other bottlenecks are removed; the earlier 'ternary is not the blocker' reading was masked.
- Not tested: read-before-write ordering effect (Q3), hard vs soft reset (Q2), quantised readout alone.

## Update 2 (2026-10-12 IST): ternary spiking cell passes delay-3 copy (MEASURED, CPU)
Config: MatMulFreeSpikingCell(norm=True, taps=6), ternary weights, binary spikes, H=128, LR 3e-3 with cosine decay, threshold LR x1, fast weights OFF, 2400 steps, delay 3 of 8 random bits (chance 50%).
- seed 0: 96.7%; seed 1: 97.6%  (both beyond the 95% target)
- Same with threshold LR x5: 94.7% (seed 0); at 1200 steps: thr x1 92.7% vs thr x5 90.8%. => higher threshold LR was not helpful here; schedule/width were what mattered.
- Float weights reach 99.9% (H=48, 600 steps), so ternarisation still costs ~2-3 points at this budget.
- Not yet tested: long-range recall beyond the 6-tap window with fast weights ON (fast path hurt delay-0 copy: 79.7% vs 90.1% off), swap into L5, bits/byte under the 0.5% gate.

## Update 3 (2026-10-12 IST): lag beyond the shift register, fast weights on vs off (MEASURED, CPU, seed 0)
Delay 10 (taps=6, so the window cannot reach), ternary, H=128, 2400 steps, chance 50%: fast ON 61.8%, fast OFF 55.5%. Fast weights give a small gain past the window but nowhere near solving it. Reason (my reading, not tested): the delta-rule memory is content-addressed, a fixed positional lag needs position-like keys, so this is the wrong test for it. Seed 1: fast ON 60.7%, OFF 55.0%. Next: use a key->value recall task for the cell, then swap into L5.

## Limitation suite read-out (8000 steps, MEASURED, chance 10%)
- kv (3 seeds, N=4/8/16/32/64/128): L5 99/94/81/51/31/26%; transformer 39/19/15/10/9/13%.
- Transformer LR sweep (seed 0, kv): 5e-4 -> 34/31/22/19/16/16%; 1e-3 -> 34/30/22/16/14/14%; 4e-3 -> 33/22/9/14/14/14% (default 2e-3 seeds: 30-44% at N=4). No learning rate closes the gap, so the kv gap is not a tuning artefact (single seed each).
- hop2 (L5 s0,s1 / T s0,s1; chains 1/2/4/8/16/32): L5 100/59,53/42,42/34,33/23,13/16,17%; T 100/47,55/22,38/17,17/13,9/8,11%. L5 ahead at 4+ chains; seed 2 and kvupd/reverse still queued.

## Limitation suite final (8000 steps, 3 seeds mean, MEASURED, chance 10%; written 2026-10-13 IST)
- kv N=4/8/16/32/64/128: L5 99.0/93.8/81.2/50.5/30.7/26.0%; T 38.5/19.3/15.1/9.9/9.4/13.0%.
- hop2 chains 1/2/4/8/16/32: L5 100/55.2/38.5/31.2/18.8/13.0%; T 100/53.1/27.6/19.8/10.4/10.9%.
- kvupd (latest value of rewritten keys) N=4/8/16/32/64/128: L5 81.2/46.9/31.8/24.0/16.7/15.6%; T 69.8/45.8/16.1/10.4/8.9/8.3%.
- reverse (reversal curse), 1 seed each: forward 100%, reverse 0% for BOTH L5 and T. L5 does not fix the reversal curse.
