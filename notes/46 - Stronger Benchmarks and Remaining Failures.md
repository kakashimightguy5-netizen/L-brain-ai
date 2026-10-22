# Stronger benchmark results

All results below are five-run means. These are CPU experiments on handwritten digits, not full SpikingBrain language-model benchmarks. Training, validation and test examples are disjoint within each run. Repeated splits reuse the same underlying dataset; the runs are not five independent populations.

The deployment follow-up used previously inspected test splits and is exploratory. Perfect external error feedback is assumed in consolidation. No evidence of autonomous truth discovery is claimed.

## strong-vision

| Method | Input condition | Accuracy mean ± standard deviation |
|---|---|---|
| MLP_clean | clean | 96.00% ± 0.78 |
| MLP_clean | single35 | 64.83% ± 1.85 |
| MLP_clean | mixed | 64.72% ± 1.37 |
| MLP_clean | single50 | 44.50% ± 3.42 |
| MLP_robust | clean | 97.06% ± 1.16 |
| MLP_robust | single35 | 73.44% ± 1.85 |
| MLP_robust | mixed | 70.72% ± 0.82 |
| MLP_robust | single50 | 53.94% ± 2.18 |
| CNN_clean | clean | 95.67% ± 0.64 |
| CNN_clean | single35 | 60.83% ± 1.18 |
| CNN_clean | mixed | 62.00% ± 1.46 |
| CNN_clean | single50 | 41.50% ± 1.20 |
| CNN_robust | clean | 97.94% ± 0.72 |
| CNN_robust | single35 | 74.39% ± 3.09 |
| CNN_robust | mixed | 70.39% ± 1.34 |
| CNN_robust | single50 | 52.56% ± 2.25 |
| CNN_routed | clean | 97.28% ± 0.72 |
| CNN_routed | single35 | 74.39% ± 3.09 |
| CNN_routed | mixed | 70.39% ± 1.48 |
| CNN_routed | single50 | 52.56% ± 2.25 |
| likelihood_memory | clean | 97.94% ± 0.89 |
| CNN_memory_corrector | clean | 97.94% ± 0.89 |
| likelihood_memory | single35 | 83.39% ± 0.82 |
| CNN_memory_corrector | single35 | 83.56% ± 0.82 |
| likelihood_memory | mixed | 75.39% ± 1.79 |
| CNN_memory_corrector | mixed | 75.39% ± 1.79 |
| likelihood_memory | single50 | 58.67% ± 1.65 |
| CNN_memory_corrector | single50 | 59.44% ± 1.15 |
| CNN_independent_evidence | observations1 | 74.39% ± 3.09 |
| CNN_independent_evidence | observations3 | 91.00% ± 1.97 |
| CNN_independent_evidence | observations5 | 93.56% ± 1.34 |
| CNN_independent_evidence | observations9 | 95.61% ± 1.86 |
| CNN_evidence_gate | adaptive_observations | 95.44% ± 1.72 |
| MLP_stream_sleep | clean | 95.50% ± 0.87 |
| MLP_stream_sleep | single35 | 69.56% ± 1.42 |
| CNN_stream_sleep | clean | 96.06% ± 1.15 |
| CNN_stream_sleep | single35 | 71.11% ± 1.72 |

Seed 642: accepting 69.72% of queries gave 94.02% accuracy on accepted queries. Rejected queries are excluded from that accuracy; this is not whole-dataset accuracy.
Seed 643: accepting 70.28% of queries gave 96.84% accuracy on accepted queries. Rejected queries are excluded from that accuracy; this is not whole-dataset accuracy.
Seed 644: accepting 60.56% of queries gave 95.87% accuracy on accepted queries. Rejected queries are excluded from that accuracy; this is not whole-dataset accuracy.
Seed 645: accepting 77.78% of queries gave 90.71% accuracy on accepted queries. Rejected queries are excluded from that accuracy; this is not whole-dataset accuracy.
Seed 646: accepting 62.50% of queries gave 96.44% accuracy on accepted queries. Rejected queries are excluded from that accuracy; this is not whole-dataset accuracy.
## stable-gene

| Method | Input condition | Accuracy mean ± standard deviation |
|---|---|---|
| teacher | clean | 96.72% ± 0.69 |
| SVD8 | clean | 96.56% ± 0.93 |
| Gene8 | clean | 96.61% ± 0.82 |
| Gene8_int8 | clean | 96.61% ± 0.82 |
| SVD12 | clean | 97.06% ± 0.61 |
| Gene12 | clean | 96.94% ± 0.83 |
| Gene12_int8 | clean | 96.89% ± 0.77 |
| Gene_selected_int8 | clean | 96.67% ± 0.86 |

## strong-continual

| Method | Input condition | Accuracy mean ± standard deviation |
|---|---|---|
| naive | clean | 19.56% ± 1.97 |
| naive | single35 | 17.11% ± 1.42 |
| EWC10 | clean | 22.61% ± 5.36 |
| EWC10 | single35 | 19.17% ± 2.52 |
| EWC1000 | clean | 22.33% ± 3.44 |
| EWC1000 | single35 | 16.17% ± 4.04 |
| ordinary_replay | clean | 95.56% ± 1.00 |
| ordinary_replay | single35 | 67.61% ± 2.46 |
| balanced_replay | clean | 95.72% ± 0.93 |
| balanced_replay | single35 | 68.94% ± 2.22 |
| EWC_selected | clean | 23.00% ± 4.90 |
| EWC_selected | single35 | 16.00% ± 3.77 |
| shared_columns | clean | 94.56% ± 1.91 |
| shared_columns | single35 | 67.61% ± 2.49 |

## direct-gene

| Method | Input condition | Accuracy mean ± standard deviation |
|---|---|---|
| dense | clean | 96.72% ± 1.74 |
| Rank8 | clean | 94.61% ± 1.58 |
| Gene8 | clean | 94.56% ± 1.07 |
| Gene8_int8 | clean | 94.56% ± 1.19 |
| Rank12 | clean | 95.56% ± 2.19 |
| Gene12 | clean | 94.83% ± 2.06 |
| Gene12_int8 | clean | 95.00% ± 2.17 |
| Gene_selected_int8 | clean | 94.72% ± 1.93 |

## density-correction

| Method | Input condition | Accuracy mean ± standard deviation |
|---|---|---|
| fixed_memory | clean | 97.78% ± 0.52 |
| fixed_memory | single35 | 82.94% ± 1.45 |
| fixed_memory | single50 | 60.89% ± 3.70 |
| learned_density_corrector | clean | 97.89% ± 0.42 |
| learned_density_corrector | single35 | 82.72% ± 1.08 |
| learned_density_corrector | single50 | 60.44% ± 3.28 |

## strong-deployment

| Method | Input condition | Accuracy mean ± standard deviation |
|---|---|---|
| ordinary_replay_before | clean | 95.56% ± 1.00 |
| ordinary_replay_before | single35 | 67.61% ± 2.46 |
| ordinary_replay_before | observations3 | 85.00% ± 0.98 |
| ordinary_replay_before | observations5 | 87.94% ± 0.91 |
| ordinary_replay_before | observations9 | 90.61% ± 1.23 |
| ordinary_replay_consolidated | clean | 96.89% ± 0.66 |
| ordinary_replay_consolidated | single35 | 72.56% ± 2.53 |
| ordinary_replay_consolidated | observations3 | 89.50% ± 1.55 |
| ordinary_replay_consolidated | observations5 | 92.33% ± 1.22 |
| ordinary_replay_consolidated | observations9 | 93.83% ± 1.19 |
| balanced_replay_before | clean | 95.72% ± 0.93 |
| balanced_replay_before | single35 | 68.94% ± 2.22 |
| balanced_replay_before | observations3 | 85.94% ± 1.20 |
| balanced_replay_before | observations5 | 90.33% ± 0.91 |
| balanced_replay_before | observations9 | 92.83% ± 0.63 |
| balanced_replay_consolidated | clean | 95.83% ± 1.09 |
| balanced_replay_consolidated | single35 | 71.06% ± 2.97 |
| balanced_replay_consolidated | observations3 | 88.56% ± 1.61 |
| balanced_replay_consolidated | observations5 | 91.61% ± 1.15 |
| balanced_replay_consolidated | observations9 | 93.72% ± 1.53 |
| shared_columns_before | clean | 94.56% ± 1.91 |
| shared_columns_before | single35 | 67.61% ± 2.49 |
| shared_columns_before | observations3 | 85.17% ± 1.57 |
| shared_columns_before | observations5 | 89.28% ± 2.25 |
| shared_columns_before | observations9 | 91.56% ± 3.07 |
| shared_columns_consolidated | clean | 95.50% ± 0.97 |
| shared_columns_consolidated | single35 | 71.61% ± 2.06 |
| shared_columns_consolidated | observations3 | 87.39% ± 2.31 |
| shared_columns_consolidated | observations5 | 90.94% ± 1.49 |
| shared_columns_consolidated | observations9 | 92.94% ± 1.55 |

## Interpretation

Single-observation correction improved over the trained neural baseline but missed the 90% target. Independent observations reduce random corruption; nine observations are nine fresh measurements of the same underlying digit, not nine passes over one fixed input.

Pure EWC still fails under this class-incremental protocol. Replay retains old tasks much better. Shared columns do not consistently beat ordinary replay. Consolidation assumes an external source supplies correct labels for flagged errors.

Teacher-initialized gene codes compress well. Plain low-rank compression is equally competitive, so the nonlinear gene decoder has no established advantage. Direct gene training remains below the dense baseline. INT8 describes stored codes; computation uses decoded float32 weights.

The learned density corrector did not consistently improve on fixed likelihood memory. Validation-selected 95% confidence targets did not guarantee 95% accepted accuracy on every test run.

Proposed deployment router and sensor-bias inversion were not implemented in this round. They are not counted as tested results. Full 5B/7B native fine-tuning, intrinsic self-correction without feedback, and language benchmarks remain untested.

Saved prediction arrays reproduced: 430. All recorded finite-gradient/checkpoint checks passed.
