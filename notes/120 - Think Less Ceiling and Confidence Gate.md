# 120 - Think-less ceiling and a better gate (L5E, d144, 4 layers, 1 checkpoint, CPU)

All costs MODELLED (full layer = 1, skimmed layer = 6/11). One checkpoint (L5E_post3k_s0), 10-12 val batches of 16x256. Single seed: directional only.

## 1. Ceiling (oracle gate that peeks at the true next byte, NOT realisable)
| tolerance vs full-depth loss | bits/byte | modelled cost |
|---|---|---|
| 0 | 1.168 | 0.898 |
| 0.05 bit/token | 1.177 | 0.812 |
| 0.25 | 1.220 | 0.770 |
| 1.0 | 1.420 | 0.716 |
Even a perfect gate saves only ~19% (at ~0 loss) to ~23% (at +0.05 bits). With 4 layers and skim cost 6/11 the floor of depth-skipping is 0.64 per token, so per-token depth skipping is a SMALL prize here. 42% of tokens are "easy" (exit at layer 1 with no loss), but skimming them still costs 64%.

## 2. The learned halt head is a poor gate
Halt-head thresholds: gate 0.95 -> cost 0.870 but bits 1.355 vs 1.317 at 0.999 (+3%); gate 0.9 -> +7%; far from the oracle.

## 3. Exit-head confidence (max softmax prob) beats the halt head
tau per layer = quantile on TRAIN windows, reported on VAL (never fit on eval). Cost / delta bits:
confidence: ~0.96 / -0.01%, 0.924 / +0.23%, 0.889 / +1.11%. halt head: 0.958 / +0.08%, 0.924 / +0.56%, 0.886 / +1.95%.
=> confidence gate is ~2x cheaper in quality at equal work. Saves ~8% at +0.2% bits, ~11% at +1.1%.

## Verdict
Honest: depth-gating cannot deliver a big energy win in this architecture; best realistic ~8-11% of model work. Bigger levers measured elsewhere: batching (5-10x J/byte vs transformer at batch 16-64), clocks 900-1200 MHz (-26 to -34% energy), ternary/low-rank (multiply-free, +20% bits though). Next idea: make skim cheaper (no memory write for tokens the delta rule already predicts), which would lower the 0.64 floor.
Code: analysis/think/oracle_exit.py, analysis/think/conf_gate.py. Results: results/think_oracle.json, think_confgate.json.
