# 132 - Backprop-free training test (2026-10-12 IST)
Why: the user named training as the biggest problem; the brain has no backward pass. Question: how much quality does removing backprop cost on a matched task? (Dense float matrices are still used here; this tests the TRAINING rule only.)
Task: next byte from 8 prior bytes (bit inputs), MLP H=64, 6000 steps, 3 seeds, held-out py_big val, bits/byte (MEASURED, cloud CPU). Code: local_learn.py.
| method | what it is | bits/byte (3 seeds) | mean |
|---|---|---|---|
| bp | ordinary backprop | 3.117/3.113/3.121 | 3.117 |
| dfa | direct feedback alignment: fixed random feedback of the output error, no weight transport | 3.522/3.620/3.567 | 3.570 |
| local | each layer has its own fixed random readout + local loss, no gradient crosses layers | 3.624/3.524/3.524 | 3.557 |
| hebb_ro | hidden layers random and frozen, only readout trained | 4.191/4.204/4.223 | 4.206 |
Reading: backprop-free rules train, and far better than frozen random features, but cost about +0.45 bits/byte (+14%) here. That fails the 0.5% quality gate; no claim of a win. Not tested: scale, longer training, ternary weights, spiking cells, predictive-coding / three-factor rules, or whether the gap closes with width/steps (see 20000-step line below).
Honest limits: this is a tiny task; published DFA results degrade further on deep nets; energy savings from skipping the backward pass are MODELLED only (no measurement).

20000 steps, seed 0 (MEASURED): bp 2.945, dfa 3.549, local 3.523. The gap does NOT close with more steps (it widens to about +0.6 bits/byte, ~20%): the local rules plateau. Next candidates: predictive-coding with iterated inference, three-factor rule with a learned modulatory signal, and hybrid (backprop only inside a small block / fast-to-slow consolidation) rather than pure local learning.

## Update 2026-10-15 IST: predictive coding (PC), the most brain-like exact candidate (MEASURED, cloud CPU)
Implementation (local_learn.py, method pc): value nodes relax for PC_IT steps to minimise local prediction errors with a weakly nudged output (error / batch size); weights update by local error x pre-activity only.
- First version failed (5.0 bits/byte) from two bugs: wrong sign of the top-down term, and a strong output nudge. After fixes, at initialisation PC weight updates match backprop almost exactly (cosine 0.997-1.000 per layer).
- During training the alignment decays (cosine 0.83-0.99, sometimes negative) because the network does not settle fully: settling needs step size < 2/(1+|W|^2), and weights grow.
- Best setting (60 settling steps, step 0.05), 6000 steps: 3.515 / 3.572 bits/byte (seeds 0, 1) vs backprop 3.117 -> +13-15%, the same gap as feedback alignment and local losses, at ~40x the compute of backprop.
Reading: every local rule tested pays ~14-20% quality; PC can approach backprop only by settling to equilibrium, which costs tens of extra passes (more compute, not less). This failure is fundamental at current knowledge, not a tuning issue. Practical path (brain-like division of labour already in L5): fast local learning online (delta-rule fast weights, hippocampus), slow exact learning offline (backprop during 'sleep' consolidation, as in notes 133/137).

## Update 2026-10-15 ~13:00 IST: learned local readouts close most of the gap (MEASURED, cloud CPU, 3 seeds unless noted)
Brain idea: each cortical area predicts with its own output pathway; learning signals stay inside the area (or a small microcircuit) instead of travelling back through the whole hierarchy.
Two hidden layers (H=64, 6000 steps; backprop 3.117):
| method | bits/byte | gap |
|---|---|---|
| local, FIXED random readouts (old) | 3.557 | +14% |
| hybrid: bottom layer local with fixed random readout, top backprop | 3.324 | +6.6% |
| layer-local with LEARNED readouts (no gradient crosses a layer) | 3.185 | +2.2% |
At 20000 steps (seed 0): learned-readout local 3.033 vs backprop 2.945 -> +3.0% (old local was +20%).
Depth 4 (deep_local.py, H=64, 6000 steps): backprop 2.988; layer-local 3.118 (+4.4%); block-local (backprop only inside 2-layer microcircuits, detached between blocks) 3.069 (+2.7%).
Reading: the failure was mostly the FIXED random readouts. With learned local readouts the cost of never sending gradients across layers drops from ~14-20% to ~2-4%; small 2-layer blocks recover more. The gap still grows with depth (2.2% at 2 layers, 4.4% at 4), so it is reduced, not solved. Benefit (MODELLED): no long backward chain, layers can update in parallel and immediately, memory for activations is per-block only; extra cost: a readout per layer during training.
Next: block-local inside the real L5 (each L5 layer = a block with its own readout) on the language benchmark.

## Overlapping local windows and depth (MEASURED, 6000 steps, H=64, 3 seeds unless noted)
'overlap': each layer's local loss reaches back through 2 layers (gradient crosses at most one layer boundary; still no global backward pass).
- Depth 4 (plain MLP): backprop 2.988; overlap 3.041 (+1.8%); block-2 3.069 (+2.7%); layer-local 3.118 (+4.4%).
- Depth 8, plain MLP: backprop 3.128 (plain deep nets train badly); overlap 3.067 (1.9% BETTER than plain backprop).
- Depth 8 WITH residual connections (the fair modern baseline): backprop 2.863; overlap 3.050 (+6.5%); layer-local (seed 0) 3.124 (+9%); overlap reaching back 4 layers (seed 0) 3.008 (+4.7% vs that seed's 2.873).
Reading: local learning is now within ~2% at small depth, but the gap grows with depth against a strong residual backprop baseline (~5-7% at depth 8). Still a failure at depth; much better than the starting +14-20%. Longer reach-back windows trade locality for quality.
