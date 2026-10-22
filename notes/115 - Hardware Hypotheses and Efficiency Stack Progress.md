---
tags: [L, hardware, SPARK, hypotheses, energy, sustainability, efficiency-stack]
created: 2026-10-10
status: H1-H3, H5, H7-H10 measured on L-6 s0 + L-Edge s0; H4/H6 and the efficiency-stack benchmarks queued on the laptop
---
# 115 - Hardware hypotheses + software efficiency stack (progress)

Previous: [[114 - Research Weak Spots of L and Blind-Working LLMs]] · Report: `results/HW_HYPOTHESES_REPORT.md` · Plans: `docs/hw_hypotheses_plan.md`, `docs/efficiency_stack_plan.md`

## The sustainability point (user's concern)
Today's LLMs spend energy on big dense matrices for every token. L has the same problem: about 92% of its per-byte work is dense matrices (qkv, o, membrane in, f1). Only the spike-driven projections (out, f2) are sparse. Work per byte does not follow difficulty (Spearman(events, loss) 0.006 on L-6, 0.19 on L-Edge). This is MEASURED / MODELLED, not a hardware claim.

## Results so far (MEASURED activity, held-out Python; fitted on CAL, measured on EVAL)
| H | Verdict |
|---|---|
| H1 energy proportionality | FAIL as built (work flat across difficulty) |
| H2 prefetch | DROP (L-6 best 51% hit at 1.07× waste; L-Edge 70.1% / 1.03×, borderline) |
| H3 hot neurons | DROP (top 10% of neurons cover 21% / 17% of events) |
| H5 sorted queue | KEEP, small (0.81–0.99 of unsorted cycles) |
| H7 precision on demand | DROP (no better than random at equal 4-bit share); by-product: block-scaled 4-bit PTQ only +2.2% |
| H8 multi-stream | BUILD if on-chip state allows (0.27× weight traffic at 4 streams) |
| H9 clock gating | KEEP, fine-grained only |
| H10 analog noise | KEEP optional (5% weight variation +0.2%; 5% read noise +3.3%; noise-aware fine-tune no clear gain, 1 seed) |
| H4, H6 | PENDING 3-seed training (laptop jobs 1600–1626) |

## Added to the code (all behind flags, defaults unchanged)
- `QAT=ternary` (multiplier-free with integer spikes) and `LOWRANK=r` for the dense projections in `hwflags.py` / `train_L1.py`. Jobs 1627–1635 (3 seeds each) and stronger λ jobs 1637–1642. Evaluation: `bench/energy/edge_energy.py`.
- Energy-proportional runtime `serve/runtime.py` (instinct / deliberate / flag, effort knob, energy cap). Equivalence tests passed: flags off = reference stepper (diff 0.0); gated = full-window reference (1e-14).
- NVML energy kit `bench/energy/` (energy.py, eff_bench.py, power_sweep.py, wattmeter procedure). Power sweep restores the GPU state and logs it; if the laptop refuses `-pl` and `-lgc` it reports NOT PERMITTED.
- `configs/sustainability.yaml` + `analysis/sustainability.py`: PUE 1.56 (Uptime 2024), on-site WUE 0.55 L/kWh with range 0.3–1.8, off-site 3.14 (Li et al. 2023), India grid 0.727 kg CO2/kWh (CEA FY2023-24, CO2 only).

## Corrections made today
- `analysis/cost_model.py` default firing rates changed from 0.72 / 0.20 (untrained model, bench) to 0.55 / 0.052 (trained L-6 traces).
- Effort-gate skim cost 5/11 → 6/11 of a layer; earlier "compute used" numbers for skimmed tokens were about 4% optimistic.
- Post-hoc effort gate 3k (L-5 1M, s0): full depth 1.3032; gate 0.99 → 1.3118 at 92.1% compute; gate 0.95 → 1.3533 at 84.2%. Compute figures use the old 5/11 skim cost.

## Queued on the laptop (one GPU job at a time)
15m eff_bench → 15n power_sweep → 15o/p L5E s1/s2 → 15q–s serve bench → 15t H7 → 15u H10 → 15v/w traces L-6 s1/s2 → 1600–1626 (H4/H6) → 1627–1635 (ternary / low-rank) → 1636 edge_energy → 1637–1642 (λ).
