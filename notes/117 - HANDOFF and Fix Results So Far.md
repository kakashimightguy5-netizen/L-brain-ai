---
tags: [L, handoff, fixes]
created: 2026-10-10
status: LIVING HANDOFF - read this first after any pause/limit reset
---
# 117 - Handoff: state, next commands, first fix result
## First fix result (MEASURED bits/byte on 6x1024 EVAL bytes, L-Edge, 1 seed; chip nJ = ESTIMATED, 45 nm Horowitz)
| variant | bits/byte | multiplies/byte | adds-only/byte | weight KB | chip nJ/byte |
|---|---|---|---|---|---|
| baseline LEdge_s0 | 0.9359 | 605,422 | 0 | 740.6 | 896 |
| ternary + low-rank 0.25 | 1.1218 (+19.9%) | 0 | 293,388 | 105.2 | 100.5 |
Reading: 7x less storage, zero multiplies, ~9x lower ESTIMATED chip energy, but a large quality cost (+20%) at this training budget (3000 steps, 1 seed). Combined fix is NOT free. Next: see ternary-only and low-rank-only to find which half costs the quality.
## Where things run
- Cloud (container may be reclaimed if idle): chain /tmp/claude-0/chain2.sh trains cE_dspk05, cE_dspk05_tern, cE_tern, cE_lr25 (3000 steps each, ~35-40 min, results/fix/). If the container is gone, re-run them: env in chain2.sh (DM=128 NL=4 HEADS=4 ... LOUT=results/fix), train_L1.py NAME L5 0 3000 0.03.
- Laptop runner (independent of me): D:\AI\SpikingBrain-study\L-architecture\runner\START_L_RUNNER.ps1 ; heartbeat results\runner_alive.json ; do NOT kill python jobs (a killed job keeps its .started marker and never restarts - delete the marker by hand). Queue: M3 s1/s2, eff_bench, power_sweep, serve, H7/H10, hwh traces, 1600-1655 quantisation/ternary/dense-spike 3 seeds, 1660-1662 BRACKET counter.
## M3 (seed 0 only so far): dense 1.4287 | dense_small 1.4764 | token 1.4415 | segment 1.4728 bits/byte; segment routing specialises by language (MI 0.55/0.38 in layers 2/3) and switches 4-7x/1000 bytes vs ~600 for token. Benefit = power-gating, not quality. Need seeds 1,2.
## TODO order
1 evaluate each fix variant (analysis/hw_hypotheses/fix_eval.py, THREADS=1) vs baseline 0.9359; 2 write Fixes section in results/HW_HYPOTHESES_REPORT.md; 3 M3 3-seed verdict; 4 re-run analysis/sustainability.py after eff_bench/serve results exist, write results/EFFICIENCY_STACK_REPORT.md; 5 BRACKET result vs baseline (1660-1662 vs 15o/15p/LEdge); 6 metacognition report; 7 L-HW QAT.
