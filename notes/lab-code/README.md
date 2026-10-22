# Brain-Inspired Lab: Round 1 code

Toy experiments behind [[07 - Experiment Results]]. CPU-friendly; also runs on the RTX 3050.

Setup: `pip install torch scikit-learn snntorch matplotlib`

| Script | Experiment |
|---|---|
| e1_growing_columns.py | Auto-growing domain columns vs naive / EWC / replay |
| e2_self_healing.py, e2b_tau.py | Self-healing: direct edits vs hippocampus memory vs sleep consolidation vs intrinsic |
| e3_effort_gate.py | Effort gate on noisy digits (negative result) |
| e3b_effort_gate_reasoning.py | Generic looped net on pointer chasing (negative result) |
| e3c_effort_gate_graph.py | **Effort gate on pointer chasing (main result)**: `python e3c_effort_gate_graph.py 3 1500` |
| e4_spiking.py | LIF spiking network vs ANN, sparsity and energy estimate |
| e5_genome.py, e5b_genome_v2.py | Genome (DNA) weight compression |
| e6_integration.py, e6b_learned_router.py, e6c_soft_route.py | Integration of all modules vs replay baseline |

`results_*.json` contain the raw outputs.

## Round 2: root-cause fixes (see [[16 - Root Cause Analysis and Fixes]])
| Script | What it does |
|---|---|
| rc1_integration_diagnosis.py | Oracle-routing decomposition: router vs column error |
| rc1_fix_selfknowing_dreams.py | Self-knowing columns + dream sleep (`python rc1_fix_selfknowing_dreams.py 0.3`) |
| rc1_integration_v2.py, rc1_calibrated.py | Healing on split-digits; calibration hypothesis (rejected) |
| rc1_real_domains.py | **Final integration test** on the realistic 4-domain benchmark: `python rc1_real_domains.py protected` |
| rc2_genome_diagnosis.py | Why the genome failed (stability, ID information, low-rank comparison) |
| rc2_genome_fix.py | Learned gene-expression genome vs equal-size low-rank, widths 128/512/2048 |
| rc2_genomic_columns.py | Genomic columns: frozen genome + per-neuron codes per new domain |
| rc2_learn_big_sleep_small.py | Three-tier memory: distil a big column into genome codes during sleep |
