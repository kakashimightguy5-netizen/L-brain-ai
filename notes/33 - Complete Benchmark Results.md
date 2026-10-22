# Complete recorded benchmark results

Compiled from saved measurements, without retraining. Most model scores are means over five seeds42–46. Training horizons, noise, data exposure, model sizes and test sets differ between experiments: compare within each experiment, not across unrelated rows. Accuracy, accepted accuracy and rejection are distinct. Coverage is the fraction of supported requests answered; rejected supported requests are not counted as correctly answered.

No full pretrained SpikingBrain5B/7B inference, MMLU, GSM8K, HumanEval, perplexity, language-generation throughput or measured neuromorphic energy benchmark was run. Official encoder/QuantLinear tests and custom small models are distinct. Protected memory is an engineered two-event module, not a parameter-matched end-to-end learned SNN.

All aggregate condition tables below are retained. Individual-seed predictions/validation histories remain in the linked raw JSON artifacts. Earlier single-seed stress is exploratory; five-seed replication is the stronger reference.

## 07 - Experiment Results
[[07 - Experiment Results]]

| # | Idea tested | Verdict | Key number |
|---|---|---|---|
| E1 | Auto-growing domain columns (no forgetting) | ✅ **Works** | 94.9% vs 19.6% naive, 90.8% replay |
| E2 | Self-healing (memory → sleep) | ✅ **Works, with rules** | Direct weight edits destroy the model (97% → 52%); sleep consolidation fixes 86% of mistakes, +11 pts on new inputs, −3 pts old |
| E2 | Intrinsic self-correction (no feedback) | ❌ **Fails** | Breaks more answers than it fixes (6.8% vs 5.8%) |
| E3 | Effort gate (think when needed) | ✅ **Works** on reasoning; ❌ useless on perception | **Same 97.3% accuracy with 62.5% less compute**; steps track difficulty (r ≈ 0.79) |
| E4 | Spiking neurons | ✅ **Works** | 95.3% vs 97.2%, ~11% neurons firing, ~3.7× less energy (*estimate, neuromorphic only*) |
| E5 | Genome (DNA) compression | 🟡 **Weak / unstable** | 92.8% at 3.8× compression (best); other sizes fail |
| E6 | **Integration of all modules** | ❌ **Fails in v1** | Replay baseline 69.7% vs ours 62.5% on shifted data; the router is the weak link |

| Method | Avg accuracy | First domain at end | Params |
|---|---|---|---|
| Joint training (upper bound, sees all data) | 97.2% | 98.9% | 9.0k |
| Naive fine-tuning | 19.6% | **0%** (total forgetting) | 9.0k |
| EWC (classic anti-forgetting) | 19.8% | 0% | 9.0k |
| Replay (stores 20 images per class) | 90.8% | 87.0% | 9.0k |
| **Growing columns (ours)** | **94.9%** | **93.4%** | 24.1k total, 12.9k active |

| Healing method | Reliability (fixed stays fixed) | Generalisation (new noisy inputs) | Locality (old clean skill) |
|---|---|---|---|
| No healing | 0% | 54.4% | 97.3% |
| **Direct weight edits** (fix each mistake in the weights) | 38.6% | 37.2% ↓ | **52.3%** 💥 |
| **Hippocampus memory** (store fixes + router, threshold 1%) | **100%** | 53.6% | 96.2% |
| **Sleep consolidation** (memory + small replay → weights) | 86.0% | **65.2%** ↑ | 94.4% |
| Intrinsic self-correction (no feedback, switch low-confidence answers) | — | net **−0.98 pts** | — |

| Model | Accuracy | Steps |
|---|---|---|
| Fixed 1 step | 33.6% | 1 |
| Fixed 8 steps | 72.7% | 8 |
| Fixed 18 steps | 97.3% | 18 |
| **Effort gate (thr 0.95)** | **97.3%** | **6.7 avg (−62.5%)** |

| Sparsity penalty | Accuracy | Neurons firing per step | Est. energy |
|---|---|---|---|
| Normal ANN | 97.2% | 58% active | 40.6 nJ |
| SNN, none | 95.9% | 28% | 16.5 nJ |
| SNN, 0.02 | 95.7% | 19% | 13.3 nJ |
| **SNN, 0.1** | **95.3%** | **11%** | **10.9 nJ (3.7× less)** |
| SNN, 0.3 | 10.2% 💥 | 0% (network died) | — |

| Version | Compression | Accuracy (full net 97.5%) |
|---|---|---|
| v1 (tanh genome, 3 layers) | 2–25× | ~10–48% ❌ (vanishing weights; fixed with "homeostatic" scaling, still weak) |
| **v2 (Fourier IDs, 2 layers), genome 8** | **3.8×** | **92.8%** |
| v2, genome 16 | 1.8× | 58.7% (unstable) |
| v2, genome 32 | 0.8× | 84.4% |

| System | Clean after | Shifted after |
|---|---|---|
| Replay baseline, no healing | 91.6% | 58.0% |
| **Replay baseline + healing** | 92.9% | **69.7%** 🏆 |
| Columns + healing (autoencoder router) | 85.1% | 54.0% |
| Columns + healing + **learned, self-healing router** | **93.4%** | 62.5% |
| Same + soft routing | 93.1% | 62.9% |


## 11 - Two-Domain Prototype Results
[[11 - Two-Domain Prototype Results]]

| Metric | Mean | Sample SD |
|---|---:|---:|
| replay_A_after_B | 0.79688 | 0.13584 |
| replay_B_after_B | 0.97227 | 0.01151 |
| replay_accuracy | 0.88457 | 0.07293 |
| replay_forgetting_pp | 17.26562 | 14.05341 |
| memoryless_accuracy | 0.97988 | 0.00471 |
| shared_A_before_B | 0.96953 | 0.00764 |
| shared_A_after_B | 0.10859 | 0.07324 |
| shared_B_after_B | 0.96758 | 0.01315 |
| joint_accuracy | 0.97578 | 0.00327 |
| oracle_expert_accuracy | 0.97422 | 0.00406 |
| learned_route_task_accuracy | 0.97422 | 0.00406 |
| missing_marker_task_accuracy | 0.50137 | 0.00429 |
| flipped_marker_task_accuracy | 0.02695 | 0.00631 |
| snn_accuracy | 0.97930 | 0.00558 |
| adaptive_accuracy | 0.97910 | 0.00646 |
| shared_forgetting_pp | 86.09375 | 6.86573 |
| frozen_forgetting_pp | 0.00000 | 0.00000 |
| adaptive_mean_steps | 2.05801 | 0.02201 |
| fixed_snn_batch_ms | 1.15164 | 0.31759 |
| adaptive_snn_batch_ms | 4.05742 | 0.72763 |
| routed_ann_batch_ms | 0.32438 | 0.07297 |
| all_modules_ann_batch_ms | 0.26076 | 0.02123 |

[[SpikingBrain Assets/prototype-results.json]]
Static task: LIF97.93%, memoryless97.99%, adaptive97.91%; adaptive2.06/8steps but4.057ms vs fixed1.152ms. Frozen experts zero forgetting by unchanged weights; lost routing marker50.14%, flipped marker2.70%.

## 13 - Extended Prototype Results
[[13 - Extended Prototype Results]]

| Task | Model | Accuracy (mean +/- sample SD) | Parameters | Mean median batch ms |
|---|---|---|---:|---:|
| delayed_recall | memoryless | 50.00 +/- 0.00% | 146 | 0.047 |
| delayed_recall | analog | 100.00 +/- 0.00% | 146 | 0.447 |
| delayed_recall | RNN | 100.00 +/- 0.00% | 746 | 0.899 |
| delayed_recall | LIF | 99.84 +/- 0.35% | 722 | 1.598 |
| pulse_order | memoryless | 50.00 +/- 0.00% | 146 | 0.085 |
| pulse_order | analog | 100.00 +/- 0.00% | 146 | 0.424 |
| pulse_order | RNN | 100.00 +/- 0.00% | 746 | 0.880 |
| pulse_order | LIF | 100.00 +/- 0.00% | 722 | 1.898 |

| Routing metric | Mean +/- sample SD |
|---|---|
| test_accuracy_no_abstention | 100.00 +/- 0.00% |
| clean_coverage | 100.00 +/- 0.00% |
| clean_selective_accuracy | 100.00 +/- 0.00% |
| ambiguous_rejection | 96.84 +/- 1.50% |
| unknown_rejection | 100.00 +/- 0.00% |

| Metric (accuracy fractions; losses in percentage points) | Mean | Sample SD |
|---|---:|---:|
| shared_final_accuracy | 0.69401 | 0.01345 |
| replay_final_accuracy | 0.78516 | 0.01448 |
| growth_routed_accuracy | 0.98372 | 0.00122 |
| growth_oracle_accuracy | 0.98372 | 0.00122 |
| bounded_routed_accuracy | 0.91628 | 0.03727 |
| shared_A_retention_loss_pp | 42.34375 | 5.32097 |
| replay_A_retention_loss_pp | 23.28125 | 4.55836 |
| growth_A_retention_loss_pp | 0.00000 | 0.00000 |

| Metric | Mean +/- sample SD |
|---|---|
| without_memory_accuracy | 73.27 +/- 1.81% |
| with_memory_accuracy | 99.37 +/- 0.38% |
| stored_fact_accuracy | 100.00 +/- 0.00% |
| default_prediction_accuracy | 98.73 +/- 0.76% |
| with_memory_coverage | 100.00 +/- 0.00% |
| gated_memory_correct_fraction | 99.37 +/- 0.38% |
| gated_memory_coverage | 100.00 +/- 0.00% |

[[SpikingBrain Assets/extended-results.json]]

## 14 - Temporal Stress Results
[[14 - Temporal Stress Results]]

| Task | Length | Signal noise SD | Model | Accuracy |
|---|---:|---:|---|---:|
| delayed_recall | 12 | 0.0 | memoryless | 50.00% |
| delayed_recall | 12 | 0.0 | analog | 100.00% |
| delayed_recall | 12 | 0.0 | RNN | 100.00% |
| delayed_recall | 12 | 0.0 | LIF | 100.00% |
| delayed_recall | 12 | 0.2 | memoryless | 50.00% |
| delayed_recall | 12 | 0.2 | analog | 86.72% |
| delayed_recall | 12 | 0.2 | RNN | 91.21% |
| delayed_recall | 12 | 0.2 | LIF | 99.41% |
| delayed_recall | 24 | 0.0 | memoryless | 50.00% |
| delayed_recall | 24 | 0.0 | analog | 96.88% |
| delayed_recall | 24 | 0.0 | RNN | 100.00% |
| delayed_recall | 24 | 0.0 | LIF | 100.00% |
| delayed_recall | 24 | 0.2 | memoryless | 50.00% |
| delayed_recall | 24 | 0.2 | analog | 70.70% |
| delayed_recall | 24 | 0.2 | RNN | 86.13% |
| delayed_recall | 24 | 0.2 | LIF | 98.83% |
| delayed_recall | 48 | 0.0 | memoryless | 50.00% |
| delayed_recall | 48 | 0.0 | analog | 50.00% |
| delayed_recall | 48 | 0.0 | RNN | 44.92% |
| delayed_recall | 48 | 0.0 | LIF | 100.00% |
| delayed_recall | 48 | 0.2 | memoryless | 50.00% |
| delayed_recall | 48 | 0.2 | analog | 56.64% |
| delayed_recall | 48 | 0.2 | RNN | 51.56% |
| delayed_recall | 48 | 0.2 | LIF | 99.22% |
| delayed_recall | 96 | 0.0 | memoryless | 50.00% |
| delayed_recall | 96 | 0.0 | analog | 50.00% |
| delayed_recall | 96 | 0.0 | RNN | 0.78% |
| delayed_recall | 96 | 0.0 | LIF | 100.00% |
| delayed_recall | 96 | 0.2 | memoryless | 50.00% |
| delayed_recall | 96 | 0.2 | analog | 52.73% |
| delayed_recall | 96 | 0.2 | RNN | 35.16% |
| delayed_recall | 96 | 0.2 | LIF | 99.22% |
| pulse_order | 12 | 0.0 | memoryless | 50.00% |
| pulse_order | 12 | 0.0 | analog | 100.00% |
| pulse_order | 12 | 0.0 | RNN | 100.00% |
| pulse_order | 12 | 0.0 | LIF | 100.00% |
| pulse_order | 12 | 0.2 | memoryless | 50.00% |
| pulse_order | 12 | 0.2 | analog | 66.02% |
| pulse_order | 12 | 0.2 | RNN | 98.44% |
| pulse_order | 12 | 0.2 | LIF | 95.51% |
| pulse_order | 24 | 0.0 | memoryless | 50.00% |
| pulse_order | 24 | 0.0 | analog | 84.77% |
| pulse_order | 24 | 0.0 | RNN | 58.40% |
| pulse_order | 24 | 0.0 | LIF | 26.56% |
| pulse_order | 24 | 0.2 | memoryless | 50.00% |
| pulse_order | 24 | 0.2 | analog | 55.47% |
| pulse_order | 24 | 0.2 | RNN | 54.88% |
| pulse_order | 24 | 0.2 | LIF | 30.86% |
| pulse_order | 48 | 0.0 | memoryless | 50.00% |
| pulse_order | 48 | 0.0 | analog | 50.00% |
| pulse_order | 48 | 0.0 | RNN | 1.56% |
| pulse_order | 48 | 0.0 | LIF | 86.13% |
| pulse_order | 48 | 0.2 | memoryless | 50.00% |
| pulse_order | 48 | 0.2 | analog | 52.54% |
| pulse_order | 48 | 0.2 | RNN | 6.84% |
| pulse_order | 48 | 0.2 | LIF | 77.54% |
| pulse_order | 96 | 0.0 | memoryless | 50.00% |
| pulse_order | 96 | 0.0 | analog | 50.00% |
| pulse_order | 96 | 0.0 | RNN | 33.59% |
| pulse_order | 96 | 0.0 | LIF | 22.46% |
| pulse_order | 96 | 0.2 | memoryless | 50.00% |
| pulse_order | 96 | 0.2 | analog | 49.80% |
| pulse_order | 96 | 0.2 | RNN | 38.28% |
| pulse_order | 96 | 0.2 | LIF | 33.59% |

[[SpikingBrain Assets/temporal-stress-results.json]]

## 15 - Temporal Stress Replication
[[15 - Temporal Stress Replication]]

| Task | Length | Signal noise SD | Model | Accuracy mean +/- sample SD |
|---|---:|---:|---|---:|
| delayed_recall | 12 | 0.0 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 12 | 0.0 | analog | 100.00 +/- 0.00% |
| delayed_recall | 12 | 0.0 | RNN | 100.00 +/- 0.00% |
| delayed_recall | 12 | 0.0 | LIF | 100.00 +/- 0.00% |
| delayed_recall | 12 | 0.2 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 12 | 0.2 | analog | 84.38 +/- 3.36% |
| delayed_recall | 12 | 0.2 | RNN | 98.48 +/- 3.41% |
| delayed_recall | 12 | 0.2 | LIF | 94.61 +/- 4.52% |
| delayed_recall | 24 | 0.0 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 24 | 0.0 | analog | 68.55 +/- 19.96% |
| delayed_recall | 24 | 0.0 | RNN | 59.10 +/- 53.98% |
| delayed_recall | 24 | 0.0 | LIF | 78.71 +/- 25.41% |
| delayed_recall | 24 | 0.2 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 24 | 0.2 | analog | 66.80 +/- 2.06% |
| delayed_recall | 24 | 0.2 | RNN | 52.54 +/- 47.38% |
| delayed_recall | 24 | 0.2 | LIF | 74.80 +/- 24.37% |
| delayed_recall | 48 | 0.0 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 48 | 0.0 | analog | 50.16 +/- 0.35% |
| delayed_recall | 48 | 0.0 | RNN | 68.71 +/- 45.53% |
| delayed_recall | 48 | 0.0 | LIF | 69.80 +/- 27.57% |
| delayed_recall | 48 | 0.2 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 48 | 0.2 | analog | 55.20 +/- 1.86% |
| delayed_recall | 48 | 0.2 | RNN | 70.62 +/- 41.63% |
| delayed_recall | 48 | 0.2 | LIF | 71.80 +/- 25.01% |
| delayed_recall | 96 | 0.0 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 96 | 0.0 | analog | 50.00 +/- 0.00% |
| delayed_recall | 96 | 0.0 | RNN | 20.27 +/- 44.57% |
| delayed_recall | 96 | 0.0 | LIF | 69.77 +/- 27.60% |
| delayed_recall | 96 | 0.2 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 96 | 0.2 | analog | 50.74 +/- 4.08% |
| delayed_recall | 96 | 0.2 | RNN | 31.56 +/- 39.58% |
| delayed_recall | 96 | 0.2 | LIF | 70.31 +/- 26.69% |
| pulse_order | 12 | 0.0 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 12 | 0.0 | analog | 100.00 +/- 0.00% |
| pulse_order | 12 | 0.0 | RNN | 99.96 +/- 0.09% |
| pulse_order | 12 | 0.0 | LIF | 100.00 +/- 0.00% |
| pulse_order | 12 | 0.2 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 12 | 0.2 | analog | 66.72 +/- 1.53% |
| pulse_order | 12 | 0.2 | RNN | 98.16 +/- 0.86% |
| pulse_order | 12 | 0.2 | LIF | 96.60 +/- 0.74% |
| pulse_order | 24 | 0.0 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 24 | 0.0 | analog | 92.19 +/- 5.29% |
| pulse_order | 24 | 0.0 | RNN | 83.87 +/- 16.46% |
| pulse_order | 24 | 0.0 | LIF | 33.95 +/- 26.03% |
| pulse_order | 24 | 0.2 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 24 | 0.2 | analog | 57.85 +/- 2.63% |
| pulse_order | 24 | 0.2 | RNN | 78.63 +/- 16.20% |
| pulse_order | 24 | 0.2 | LIF | 36.33 +/- 18.68% |
| pulse_order | 48 | 0.0 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 48 | 0.0 | analog | 50.39 +/- 0.60% |
| pulse_order | 48 | 0.0 | RNN | 34.96 +/- 40.57% |
| pulse_order | 48 | 0.0 | LIF | 43.01 +/- 24.50% |
| pulse_order | 48 | 0.2 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 48 | 0.2 | analog | 51.95 +/- 2.88% |
| pulse_order | 48 | 0.2 | RNN | 39.02 +/- 35.19% |
| pulse_order | 48 | 0.2 | LIF | 45.51 +/- 22.05% |
| pulse_order | 96 | 0.0 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 96 | 0.0 | analog | 50.00 +/- 0.00% |
| pulse_order | 96 | 0.0 | RNN | 53.36 +/- 37.23% |
| pulse_order | 96 | 0.0 | LIF | 47.70 +/- 17.07% |
| pulse_order | 96 | 0.2 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 96 | 0.2 | analog | 50.12 +/- 3.38% |
| pulse_order | 96 | 0.2 | RNN | 54.49 +/- 28.29% |
| pulse_order | 96 | 0.2 | LIF | 50.08 +/- 13.50% |

[[SpikingBrain Assets/temporal-stress-five-seed.json]]

## 18 - Variable Delay Memory Results
[[18 - Variable Delay Memory Results]]

| Task | Model | Length | Signal noise SD | Accuracy | Batch ms |
|---|---|---:|---:|---|---:|
| delayed_recall | memoryless | 12 | 0.1 | 50.00 +/- 0.00% | 0.401 |
| delayed_recall | memoryless | 12 | 0.2 | 50.00 +/- 0.00% | 0.495 |
| delayed_recall | memoryless | 32 | 0.1 | 50.00 +/- 0.00% | 0.456 |
| delayed_recall | memoryless | 32 | 0.2 | 50.00 +/- 0.00% | 0.463 |
| delayed_recall | memoryless | 48 | 0.1 | 50.00 +/- 0.00% | 0.381 |
| delayed_recall | memoryless | 48 | 0.2 | 50.00 +/- 0.00% | 0.404 |
| delayed_recall | memoryless | 96 | 0.1 | 50.00 +/- 0.00% | 0.544 |
| delayed_recall | memoryless | 96 | 0.2 | 50.00 +/- 0.00% | 0.498 |
| delayed_recall | analog | 12 | 0.1 | 99.02 +/- 0.37% | 4.705 |
| delayed_recall | analog | 12 | 0.2 | 87.07 +/- 1.25% | 5.091 |
| delayed_recall | analog | 32 | 0.1 | 74.84 +/- 2.74% | 13.351 |
| delayed_recall | analog | 32 | 0.2 | 62.46 +/- 1.00% | 13.027 |
| delayed_recall | analog | 48 | 0.1 | 62.03 +/- 2.36% | 20.199 |
| delayed_recall | analog | 48 | 0.2 | 55.51 +/- 2.79% | 22.391 |
| delayed_recall | analog | 96 | 0.1 | 50.39 +/- 2.53% | 49.002 |
| delayed_recall | analog | 96 | 0.2 | 49.02 +/- 1.86% | 44.611 |
| delayed_recall | RNN | 12 | 0.1 | 80.00 +/- 44.72% | 3.168 |
| delayed_recall | RNN | 12 | 0.2 | 80.00 +/- 44.72% | 2.822 |
| delayed_recall | RNN | 32 | 0.1 | 100.00 +/- 0.00% | 6.422 |
| delayed_recall | RNN | 32 | 0.2 | 100.00 +/- 0.00% | 6.879 |
| delayed_recall | RNN | 48 | 0.1 | 100.00 +/- 0.00% | 10.489 |
| delayed_recall | RNN | 48 | 0.2 | 100.00 +/- 0.00% | 11.041 |
| delayed_recall | RNN | 96 | 0.1 | 100.00 +/- 0.00% | 21.060 |
| delayed_recall | RNN | 96 | 0.2 | 99.92 +/- 0.17% | 21.027 |
| delayed_recall | LIF | 12 | 0.1 | 100.00 +/- 0.00% | 8.980 |
| delayed_recall | LIF | 12 | 0.2 | 99.84 +/- 0.35% | 10.831 |
| delayed_recall | LIF | 32 | 0.1 | 89.41 +/- 23.67% | 23.694 |
| delayed_recall | LIF | 32 | 0.2 | 89.49 +/- 23.39% | 26.003 |
| delayed_recall | LIF | 48 | 0.1 | 89.14 +/- 24.28% | 39.920 |
| delayed_recall | LIF | 48 | 0.2 | 89.22 +/- 24.11% | 36.301 |
| delayed_recall | LIF | 96 | 0.1 | 90.43 +/- 21.40% | 76.964 |
| delayed_recall | LIF | 96 | 0.2 | 90.00 +/- 22.36% | 78.554 |
| delayed_recall | GRU | 12 | 0.1 | 100.00 +/- 0.00% | 7.628 |
| delayed_recall | GRU | 12 | 0.2 | 100.00 +/- 0.00% | 7.950 |
| delayed_recall | GRU | 32 | 0.1 | 100.00 +/- 0.00% | 18.959 |
| delayed_recall | GRU | 32 | 0.2 | 100.00 +/- 0.00% | 18.899 |
| delayed_recall | GRU | 48 | 0.1 | 100.00 +/- 0.00% | 28.107 |
| delayed_recall | GRU | 48 | 0.2 | 100.00 +/- 0.00% | 29.227 |
| delayed_recall | GRU | 96 | 0.1 | 100.00 +/- 0.00% | 56.730 |
| delayed_recall | GRU | 96 | 0.2 | 100.00 +/- 0.00% | 58.507 |
| delayed_recall | LSTM | 12 | 0.1 | 100.00 +/- 0.00% | 6.808 |
| delayed_recall | LSTM | 12 | 0.2 | 100.00 +/- 0.00% | 7.285 |
| delayed_recall | LSTM | 32 | 0.1 | 100.00 +/- 0.00% | 18.146 |
| delayed_recall | LSTM | 32 | 0.2 | 100.00 +/- 0.00% | 18.317 |
| delayed_recall | LSTM | 48 | 0.1 | 100.00 +/- 0.00% | 28.223 |
| delayed_recall | LSTM | 48 | 0.2 | 99.96 +/- 0.09% | 26.849 |
| delayed_recall | LSTM | 96 | 0.1 | 100.00 +/- 0.00% | 50.475 |
| delayed_recall | LSTM | 96 | 0.2 | 100.00 +/- 0.00% | 49.883 |
| pulse_order | memoryless | 12 | 0.1 | 50.00 +/- 0.00% | 0.418 |
| pulse_order | memoryless | 12 | 0.2 | 50.00 +/- 0.00% | 0.429 |
| pulse_order | memoryless | 32 | 0.1 | 50.00 +/- 0.00% | 0.479 |
| pulse_order | memoryless | 32 | 0.2 | 50.00 +/- 0.00% | 0.437 |
| pulse_order | memoryless | 48 | 0.1 | 50.00 +/- 0.00% | 0.461 |
| pulse_order | memoryless | 48 | 0.2 | 50.00 +/- 0.00% | 0.473 |
| pulse_order | memoryless | 96 | 0.1 | 50.00 +/- 0.00% | 0.449 |
| pulse_order | memoryless | 96 | 0.2 | 50.00 +/- 0.00% | 0.471 |
| pulse_order | analog | 12 | 0.1 | 83.44 +/- 1.11% | 4.604 |
| pulse_order | analog | 12 | 0.2 | 68.79 +/- 1.55% | 4.947 |
| pulse_order | analog | 32 | 0.1 | 88.71 +/- 1.87% | 13.493 |
| pulse_order | analog | 32 | 0.2 | 75.31 +/- 1.00% | 12.758 |
| pulse_order | analog | 48 | 0.1 | 87.93 +/- 1.92% | 20.498 |
| pulse_order | analog | 48 | 0.2 | 74.34 +/- 1.57% | 18.484 |
| pulse_order | analog | 96 | 0.1 | 79.06 +/- 2.93% | 47.286 |
| pulse_order | analog | 96 | 0.2 | 70.43 +/- 3.13% | 47.265 |
| pulse_order | RNN | 12 | 0.1 | 99.92 +/- 0.11% | 3.224 |
| pulse_order | RNN | 12 | 0.2 | 95.27 +/- 1.38% | 2.836 |
| pulse_order | RNN | 32 | 0.1 | 99.69 +/- 0.17% | 6.599 |
| pulse_order | RNN | 32 | 0.2 | 87.46 +/- 3.08% | 6.518 |
| pulse_order | RNN | 48 | 0.1 | 98.09 +/- 1.60% | 10.336 |
| pulse_order | RNN | 48 | 0.2 | 78.71 +/- 1.59% | 9.894 |
| pulse_order | RNN | 96 | 0.1 | 93.40 +/- 3.69% | 20.427 |
| pulse_order | RNN | 96 | 0.2 | 66.72 +/- 5.82% | 17.813 |
| pulse_order | LIF | 12 | 0.1 | 79.34 +/- 10.71% | 9.739 |
| pulse_order | LIF | 12 | 0.2 | 79.77 +/- 8.91% | 9.073 |
| pulse_order | LIF | 32 | 0.1 | 58.67 +/- 6.80% | 25.547 |
| pulse_order | LIF | 32 | 0.2 | 57.42 +/- 4.05% | 24.766 |
| pulse_order | LIF | 48 | 0.1 | 53.24 +/- 3.39% | 38.868 |
| pulse_order | LIF | 48 | 0.2 | 53.44 +/- 3.86% | 34.786 |
| pulse_order | LIF | 96 | 0.1 | 53.40 +/- 1.99% | 77.597 |
| pulse_order | LIF | 96 | 0.2 | 51.64 +/- 1.79% | 72.063 |
| pulse_order | GRU | 12 | 0.1 | 99.45 +/- 0.70% | 7.836 |
| pulse_order | GRU | 12 | 0.2 | 97.23 +/- 1.53% | 7.157 |
| pulse_order | GRU | 32 | 0.1 | 99.45 +/- 0.35% | 19.281 |
| pulse_order | GRU | 32 | 0.2 | 89.53 +/- 1.53% | 18.483 |
| pulse_order | GRU | 48 | 0.1 | 94.96 +/- 3.03% | 27.757 |
| pulse_order | GRU | 48 | 0.2 | 79.65 +/- 1.76% | 28.323 |
| pulse_order | GRU | 96 | 0.1 | 79.57 +/- 7.49% | 54.925 |
| pulse_order | GRU | 96 | 0.2 | 65.66 +/- 3.64% | 59.856 |
| pulse_order | LSTM | 12 | 0.1 | 99.84 +/- 0.21% | 7.338 |
| pulse_order | LSTM | 12 | 0.2 | 98.55 +/- 0.56% | 6.966 |
| pulse_order | LSTM | 32 | 0.1 | 98.48 +/- 0.52% | 19.185 |
| pulse_order | LSTM | 32 | 0.2 | 90.39 +/- 1.38% | 18.069 |
| pulse_order | LSTM | 48 | 0.1 | 85.00 +/- 4.08% | 26.581 |
| pulse_order | LSTM | 48 | 0.2 | 77.70 +/- 5.74% | 26.692 |
| pulse_order | LSTM | 96 | 0.1 | 57.50 +/- 11.16% | 50.953 |
| pulse_order | LSTM | 96 | 0.2 | 56.56 +/- 5.95% | 52.241 |

[[SpikingBrain Assets/variable-delay-results.json]]

## 19 - Real Query Routing Results
[[19 - Real Query Routing Results]]

| Evaluation | Metric | Mean +/- sample SD |
|---|---|---|
| standard | accuracy_without_rejection | 97.96 +/- 0.12% |
| standard | coverage | 84.37 +/- 1.07% |
| standard | accepted_accuracy | 99.98 +/- 0.04% |
| standard | unsupported_rejection | 90.95 +/- 0.29% |
| standard | correct_answer_fraction | 84.36 +/- 1.10% |
| heldout_intents | accuracy_without_rejection | 59.41 +/- 1.58% |
| heldout_intents | coverage | 16.15 +/- 1.47% |
| heldout_intents | accepted_accuracy | 89.01 +/- 4.15% |
| heldout_intents | unsupported_rejection | 90.54 +/- 0.52% |
| heldout_intents | correct_answer_fraction | 14.37 +/- 1.40% |

| Text | Expected | Decision | Outcome |
|---|---|---|---|
| I am not asking about my bank balance. I want to know my paid time off balance. | work | unknown | incorrect |
| Don't book a flight; check when my salary will be paid. | work | unknown | incorrect |
| This is not about my salary; I need the baggage allowance for my flight. | travel | unknown | incorrect |
| Not a flight question: what is the balance of my checking account? | banking | banking | correct |
| Please ignore my bank balance. How do I cook an omelette? | unknown | banking | incorrect |
| I don't want to schedule a meeting; tell me the weather. | unknown | work | incorrect |
| I don't need flight information. Tell me a joke. | unknown | travel | incorrect |
| I don't need a recipe, I need to check my account balance. | banking | banking | correct |

[[SpikingBrain Assets/real-query-routing-results.json]]

## 22 - Local Embedding Routing Results
[[22 - Local Embedding Routing Results]]

| Evaluation | Metric | Mean +/- sample SD |
|---|---|---|
| standard | accuracy_without_rejection | 98.76 +/- 0.16% |
| standard | coverage | 39.27 +/- 21.97% |
| standard | accepted_accuracy | 100.00 +/- 0.00% |
| standard | correct_answer_fraction | 39.27 +/- 21.97% |
| standard | unsupported_rejection | 93.58 +/- 3.59% |
| heldout_intents | accuracy_without_rejection | 62.59 +/- 0.91% |
| heldout_intents | coverage | 9.19 +/- 0.71% |
| heldout_intents | accepted_accuracy | 100.00 +/- 0.00% |
| heldout_intents | correct_answer_fraction | 9.19 +/- 0.71% |
| heldout_intents | unsupported_rejection | 91.67 +/- 0.58% |

| Text | Expected | Decision | Correct |
|---|---|---|---|
| I am not asking about my bank balance. I want to know my paid time off balance. | work | unknown | False |
| Don't book a flight; check when my salary will be paid. | work | unknown | False |
| This is not about my salary; I need the baggage allowance for my flight. | travel | unknown | False |
| Not a flight question: what is the balance of my checking account? | banking | unknown | False |
| Please ignore my bank balance. How do I cook an omelette? | unknown | unknown | True |
| I don't want to schedule a meeting; tell me the weather. | unknown | unknown | True |
| I don't need flight information. Tell me a joke. | unknown | unknown | True |
| I don't need a recipe, I need to check my account balance. | banking | unknown | False |

[[SpikingBrain Assets/embedding-routing-results.json]]
Standard seed43 selected reject-all fallback: zero supported coverage; accepted accuracy undefined and excluded from accepted-accuracy aggregate. Mean accepted accuracy100% therefore uses four nonempty runs; mean coverage39.27% includes allfive. Negation probe correct counts3/3/4/2/2 of8.

## 23 - Longer Order Memory Results
[[23 - Longer Order Memory Results]]

| Variant | Test length | Noise | Mean +/- sample SD | Seed range |
|---|---:|---:|---|---|
| RNN | 32 | 0.1 | 99.49 +/- 0.64% | 98.44–100.00% |
| RNN | 32 | 0.2 | 90.98 +/- 1.74% | 88.09–92.77% |
| RNN | 96 | 0.1 | 99.14 +/- 1.39% | 96.68–100.00% |
| RNN | 96 | 0.2 | 78.48 +/- 5.01% | 74.41–86.91% |
| RNN | 192 | 0.1 | 98.59 +/- 2.04% | 95.12–100.00% |
| RNN | 192 | 0.2 | 67.54 +/- 5.14% | 62.70–74.61% |
| GRU | 32 | 0.1 | 99.22 +/- 0.82% | 97.85–100.00% |
| GRU | 32 | 0.2 | 91.84 +/- 2.44% | 87.70–94.14% |
| GRU | 96 | 0.1 | 95.43 +/- 1.52% | 93.16–97.07% |
| GRU | 96 | 0.2 | 72.23 +/- 4.86% | 66.41–77.34% |
| GRU | 192 | 0.1 | 81.91 +/- 10.80% | 65.82–91.80% |
| GRU | 192 | 0.2 | 58.75 +/- 1.74% | 55.66–59.77% |
| LIF_spikes | 32 | 0.1 | 61.37 +/- 6.24% | 53.91–69.92% |
| LIF_spikes | 32 | 0.2 | 60.43 +/- 5.01% | 53.91–67.38% |
| LIF_spikes | 96 | 0.1 | 54.57 +/- 4.74% | 51.56–62.50% |
| LIF_spikes | 96 | 0.2 | 52.93 +/- 3.69% | 48.05–58.01% |
| LIF_spikes | 192 | 0.1 | 53.24 +/- 2.98% | 49.80–56.64% |
| LIF_spikes | 192 | 0.2 | 51.72 +/- 1.31% | 50.39–53.12% |
| LIF_membrane | 32 | 0.1 | 89.10 +/- 4.31% | 81.84–93.36% |
| LIF_membrane | 32 | 0.2 | 76.95 +/- 3.36% | 72.27–80.86% |
| LIF_membrane | 96 | 0.1 | 74.77 +/- 4.64% | 67.38–80.08% |
| LIF_membrane | 96 | 0.2 | 69.34 +/- 3.63% | 65.23–73.83% |
| LIF_membrane | 192 | 0.1 | 63.75 +/- 4.71% | 57.62–69.73% |
| LIF_membrane | 192 | 0.2 | 59.22 +/- 2.12% | 56.25–61.33% |

[[SpikingBrain Assets/long-order-results.json]]

## 26 - State Tracing and Neuron Ablation Results
[[26 - State Tracing and Neuron Ablation Results]]

| Frozen model | Representation | Step | Probe mean +/- SD |
|---|---|---:|---|
| RNN | hidden | 5 | 100.00 +/- 0.00% |
| RNN | hidden | 39 | 98.59 +/- 1.40% |
| RNN | hidden | 41 | 100.00 +/- 0.00% |
| RNN | hidden | 95 | 98.98 +/- 1.53% |
| LIF_spikes | membrane | 5 | 100.00 +/- 0.00% |
| LIF_spikes | membrane | 39 | 81.48 +/- 12.92% |
| LIF_spikes | membrane | 41 | 99.92 +/- 0.11% |
| LIF_spikes | membrane | 95 | 69.96 +/- 18.42% |
| LIF_spikes | instantaneous_spikes | 5 | 93.01 +/- 11.35% |
| LIF_spikes | instantaneous_spikes | 39 | 61.29 +/- 14.72% |
| LIF_spikes | instantaneous_spikes | 41 | 81.13 +/- 12.59% |
| LIF_spikes | instantaneous_spikes | 95 | 56.80 +/- 13.37% |
| LIF_membrane | membrane | 5 | 100.00 +/- 0.00% |
| LIF_membrane | membrane | 39 | 91.29 +/- 7.80% |
| LIF_membrane | membrane | 41 | 99.84 +/- 0.16% |
| LIF_membrane | membrane | 95 | 64.38 +/- 11.71% |
| LIF_membrane | instantaneous_spikes | 5 | 82.38 +/- 13.30% |
| LIF_membrane | instantaneous_spikes | 39 | 54.88 +/- 4.77% |
| LIF_membrane | instantaneous_spikes | 41 | 77.89 +/- 16.96% |
| LIF_membrane | instantaneous_spikes | 95 | 50.47 +/- 1.75% |

| Variant | Test length | Noise | Mean +/- sample SD | Seed range |
|---|---:|---:|---|---|
| RNN | 32 | 0.1 | 99.34 +/- 0.78% | 98.05-100.00% |
| RNN | 32 | 0.2 | 91.52 +/- 3.66% | 85.94-94.92% |
| RNN | 96 | 0.1 | 98.28 +/- 1.42% | 96.09-100.00% |
| RNN | 96 | 0.2 | 75.39 +/- 5.22% | 69.73-83.20% |
| RNN | 192 | 0.1 | 96.33 +/- 2.92% | 92.38-99.61% |
| RNN | 192 | 0.2 | 64.73 +/- 5.38% | 60.16-73.24% |
| spikes_base | 32 | 0.1 | 56.95 +/- 3.58% | 53.32-62.30% |
| spikes_base | 32 | 0.2 | 57.19 +/- 2.24% | 53.91-60.16% |
| spikes_base | 96 | 0.1 | 51.84 +/- 2.29% | 48.83-54.69% |
| spikes_base | 96 | 0.2 | 51.88 +/- 0.86% | 50.59-52.73% |
| spikes_base | 192 | 0.1 | 50.43 +/- 1.81% | 48.44-52.34% |
| spikes_base | 192 | 0.2 | 50.39 +/- 1.65% | 48.63-52.93% |
| membrane_base | 32 | 0.1 | 89.53 +/- 2.54% | 85.16-91.21% |
| membrane_base | 32 | 0.2 | 81.48 +/- 3.46% | 77.34-85.35% |
| membrane_base | 96 | 0.1 | 78.20 +/- 4.90% | 71.29-83.20% |
| membrane_base | 96 | 0.2 | 70.00 +/- 1.53% | 68.75-72.66% |
| membrane_base | 192 | 0.1 | 66.41 +/- 2.68% | 61.91-69.14% |
| membrane_base | 192 | 0.2 | 61.09 +/- 1.20% | 59.38-62.30% |
| membrane_decay99 | 32 | 0.1 | 75.20 +/- 17.54% | 52.54-94.53% |
| membrane_decay99 | 32 | 0.2 | 72.73 +/- 11.70% | 58.59-86.91% |
| membrane_decay99 | 96 | 0.1 | 66.25 +/- 14.34% | 50.78-81.25% |
| membrane_decay99 | 96 | 0.2 | 61.56 +/- 8.89% | 51.37-70.90% |
| membrane_decay99 | 192 | 0.1 | 59.34 +/- 11.35% | 50.00-74.61% |
| membrane_decay99 | 192 | 0.2 | 56.41 +/- 5.48% | 50.00-62.30% |
| membrane_hard_reset | 32 | 0.1 | 91.09 +/- 2.83% | 86.13-92.97% |
| membrane_hard_reset | 32 | 0.2 | 81.91 +/- 2.88% | 78.12-85.55% |
| membrane_hard_reset | 96 | 0.1 | 75.74 +/- 6.50% | 67.19-83.98% |
| membrane_hard_reset | 96 | 0.2 | 66.64 +/- 2.10% | 64.65-69.73% |
| membrane_hard_reset | 192 | 0.1 | 62.54 +/- 3.38% | 58.20-67.38% |
| membrane_hard_reset | 192 | 0.2 | 58.52 +/- 2.57% | 54.69-61.91% |

[[SpikingBrain Assets/state-trace-results.json]]
[[SpikingBrain Assets/neuron-ablation-results.json]]

## 27 - Independent Language Challenge Results
[[27 - Independent Language Challenge Results]]

| Router | Style | Correct decision incl. rejection | Supported coverage | Accepted accuracy | Unsupported rejection |
|---|---|---:|---:|---:|---:|
| lexical | all | 54.3% | 34.7% | 92.3% | 76.7% |
| lexical | neutral | 69.0% | 50.0% | 100.0% | 88.0% |
| lexical | reframed | 67.0% | 42.0% | 100.0% | 92.0% |
| lexical | negation | 27.0% | 12.0% | 33.3% | 50.0% |
| embedding | all | 56.3% | 30.7% | 100.0% | 82.0% |
| embedding | neutral | 64.0% | 46.0% | 100.0% | 82.0% |
| embedding | reframed | 64.0% | 42.0% | 100.0% | 86.0% |
| embedding | negation | 41.0% | 4.0% | 100.0% | 78.0% |

[[SpikingBrain Assets/language-challenge-results.json]]
[[SpikingBrain Assets/language-challenge.json]]

## 30 - Protected Event Memory Fix Results
[[30 - Protected Event Memory Fix Results]]

| Model | Length | Noise | Mean +/- sample SD |
|---|---:|---:|---|
| old_membrane | 32 | 0.1 | 90.94 +/- 0.98% |
| old_membrane | 32 | 0.2 | 79.10 +/- 3.30% |
| old_membrane | 32 | 0.3 | 70.70 +/- 4.24% |
| old_membrane | 96 | 0.1 | 78.32 +/- 3.67% |
| old_membrane | 96 | 0.2 | 69.06 +/- 2.72% |
| old_membrane | 96 | 0.3 | 63.32 +/- 2.02% |
| old_membrane | 192 | 0.1 | 67.70 +/- 2.82% |
| old_membrane | 192 | 0.2 | 60.47 +/- 2.40% |
| old_membrane | 192 | 0.3 | 57.15 +/- 3.31% |
| RNN | 32 | 0.1 | 99.65 +/- 0.32% |
| RNN | 32 | 0.2 | 89.96 +/- 4.15% |
| RNN | 32 | 0.3 | 75.98 +/- 2.47% |
| RNN | 96 | 0.1 | 98.36 +/- 1.16% |
| RNN | 96 | 0.2 | 74.30 +/- 6.28% |
| RNN | 96 | 0.3 | 59.77 +/- 3.71% |
| RNN | 192 | 0.1 | 96.37 +/- 2.66% |
| RNN | 192 | 0.2 | 63.95 +/- 4.60% |
| RNN | 192 | 0.3 | 55.35 +/- 1.77% |
| protected_slot | 32 | 0.1 | 100.00 +/- 0.00% |
| protected_slot | 32 | 0.2 | 99.96 +/- 0.09% |
| protected_slot | 32 | 0.3 | 97.11 +/- 1.53% |
| protected_slot | 96 | 0.1 | 100.00 +/- 0.00% |
| protected_slot | 96 | 0.2 | 99.61 +/- 0.68% |
| protected_slot | 96 | 0.3 | 91.33 +/- 5.98% |
| protected_slot | 192 | 0.1 | 100.00 +/- 0.00% |
| protected_slot | 192 | 0.2 | 99.41 +/- 1.10% |
| protected_slot | 192 | 0.3 | 86.05 +/- 7.40% |

[[SpikingBrain Assets/protected-slot-results.json]]
[[SpikingBrain Assets/protected-memory-v2-results.json]]

## 31 - Unknown Class and Request Scope Fix Results
[[31 - Unknown Class and Request Scope Fix Results]]

| Official evaluation | Metric | Mean +/- sample SD |
|---|---|---|
| standard | accuracy_without_rejection | 94.74 +/- 0.19% |
| standard | coverage | 95.13 +/- 0.19% |
| standard | accepted_accuracy | 99.60 +/- 0.07% |
| standard | correct_answer_fraction | 94.74 +/- 0.19% |
| standard | unsupported_rejection | 98.23 +/- 0.10% |
| heldout_intents | accuracy_without_rejection | 31.19 +/- 0.71% |
| heldout_intents | coverage | 37.78 +/- 0.94% |
| heldout_intents | accepted_accuracy | 82.57 +/- 1.98% |
| heldout_intents | correct_answer_fraction | 31.19 +/- 0.71% |
| heldout_intents | unsupported_rejection | 98.62 +/- 0.07% |

| Dataset | Router | Style | Correct decision | Supported coverage | Unknown rejection |
|---|---|---|---:|---:|---:|
| fresh_wrappers | old_lexical | all | 58.57% | 51.67% | 75.83% |
| fresh_wrappers | old_lexical | denial_after | 53.57% | 41.67% | 83.33% |
| fresh_wrappers | old_lexical | do_not_discuss | 51.19% | 41.67% | 75.00% |
| fresh_wrappers | old_lexical | neutral | 90.48% | 90.00% | 91.67% |
| fresh_wrappers | old_lexical | rather_than | 48.81% | 41.67% | 66.67% |
| fresh_wrappers | old_lexical | unsupported_form | 48.81% | 43.33% | 62.50% |
| fresh_wrappers | fixed_no_scope | all | 83.10% | 96.33% | 77.50% |
| fresh_wrappers | fixed_no_scope | denial_after | 78.57% | 98.33% | 62.50% |
| fresh_wrappers | fixed_no_scope | do_not_discuss | 76.19% | 98.33% | 58.33% |
| fresh_wrappers | fixed_no_scope | neutral | 92.86% | 90.00% | 100.00% |
| fresh_wrappers | fixed_no_scope | rather_than | 84.52% | 98.33% | 79.17% |
| fresh_wrappers | fixed_no_scope | unsupported_form | 83.33% | 96.67% | 87.50% |
| fresh_wrappers | fixed_scope | all | 88.10% | 93.00% | 90.00% |
| fresh_wrappers | fixed_scope | denial_after | 78.57% | 98.33% | 62.50% |
| fresh_wrappers | fixed_scope | do_not_discuss | 92.86% | 90.00% | 100.00% |
| fresh_wrappers | fixed_scope | neutral | 92.86% | 90.00% | 100.00% |
| fresh_wrappers | fixed_scope | rather_than | 92.86% | 90.00% | 100.00% |
| fresh_wrappers | fixed_scope | unsupported_form | 83.33% | 96.67% | 87.50% |
| fresh_wrappers | old_embedding | all | 46.19% | 29.33% | 88.33% |
| fresh_wrappers | old_embedding | denial_after | 44.05% | 28.33% | 83.33% |
| fresh_wrappers | old_embedding | do_not_discuss | 46.43% | 30.00% | 87.50% |
| fresh_wrappers | old_embedding | neutral | 47.62% | 30.00% | 91.67% |
| fresh_wrappers | old_embedding | rather_than | 46.43% | 30.00% | 87.50% |
| fresh_wrappers | old_embedding | unsupported_form | 46.43% | 28.33% | 91.67% |
| seen_regression | old_lexical | all | 54.33% | 34.67% | 76.67% |
| seen_regression | old_lexical | negation | 27.00% | 12.00% | 50.00% |
| seen_regression | old_lexical | neutral | 69.00% | 50.00% | 88.00% |
| seen_regression | old_lexical | reframed | 67.00% | 42.00% | 92.00% |
| seen_regression | fixed_no_scope | all | 67.67% | 82.67% | 75.33% |
| seen_regression | fixed_no_scope | negation | 34.00% | 100.00% | 36.00% |
| seen_regression | fixed_no_scope | neutral | 85.00% | 76.00% | 94.00% |
| seen_regression | fixed_no_scope | reframed | 84.00% | 72.00% | 96.00% |
| seen_regression | fixed_scope | all | 82.33% | 69.33% | 95.33% |
| seen_regression | fixed_scope | negation | 78.00% | 60.00% | 96.00% |
| seen_regression | fixed_scope | neutral | 85.00% | 76.00% | 94.00% |
| seen_regression | fixed_scope | reframed | 84.00% | 72.00% | 96.00% |

[[SpikingBrain Assets/fixed-routing-results.json]]

## Protected slot v1 stress: five-seed means
Ablation stress differs from primary random-noise test grid. Weak-signal abstention is failure to answer the intended label, not successful memory.
| Condition | Supported coverage | Correct-label fraction | Rejection |
|---|---:|---:|---:|
| independent_second | 100.00% | 99.84% | 0.00% |
| no_events | 2.19% | not defined | 97.81% |
| no_first | 100.00% | not defined | 0.00% |
| weak_signal | 2.19% | 1.25% | 97.81% |

## Protected memory v2: fresh confirmation
Exactly-two-detectable-event contract; weak/incomplete sequences abstain. Changed validation/test budget from recurrent baselines, not a general architecture ranking.
| Condition | Coverage | Correct-label fraction | Rejection |
|---|---:|---:|---:|
| normal | 97.30% | 97.30% | 2.70% |
| independent_second | 97.30% | 97.30% | 2.70% |
| weak_signal | 0.04% | 0.04% | 99.96% |
| missing_first | 2.66% | not defined | 97.34% |
| no_events | 0.04% | not defined | 99.96% |

## Scope v2: separate confirmation families
420 dependent wrapper texts over84base queries, disjoint from scope-development families; underlying official corpus was previously evaluated. No refit/threshold change. Compare these routers on this same subset.
| Router | Form | Correct routing/rejection | Supported coverage | Accepted accuracy | Unsupported rejection |
|---|---|---:|---:|---:|---:|
| old_lexical | all | 44.05% | 41.00% | 89.43% | 62.50% |
| old_lexical | denial_after | 38.10% | 25.00% | 93.33% | 75.00% |
| old_lexical | do_not_discuss | 35.71% | 31.67% | 78.95% | 62.50% |
| old_lexical | explicit_answer | 27.38% | 31.67% | 78.95% | 33.33% |
| old_lexical | neutral | 78.57% | 78.33% | 100.00% | 79.17% |
| old_lexical | rather_than | 40.48% | 38.33% | 82.61% | 62.50% |
| fixed_scope_v1 | all | 84.29% | 96.67% | 83.45% | 93.33% |
| fixed_scope_v1 | denial_after | 67.86% | 100.00% | 65.00% | 75.00% |
| fixed_scope_v1 | do_not_discuss | 95.24% | 95.00% | 98.25% | 100.00% |
| fixed_scope_v1 | explicit_answer | 67.86% | 98.33% | 59.32% | 91.67% |
| fixed_scope_v1 | neutral | 95.24% | 95.00% | 98.25% | 100.00% |
| fixed_scope_v1 | rather_than | 95.24% | 95.00% | 98.25% | 100.00% |
| fixed_scope_v2 | all | 95.24% | 95.00% | 98.25% | 100.00% |
| fixed_scope_v2 | denial_after | 95.24% | 95.00% | 98.25% | 100.00% |
| fixed_scope_v2 | do_not_discuss | 95.24% | 95.00% | 98.25% | 100.00% |
| fixed_scope_v2 | explicit_answer | 95.24% | 95.00% | 98.25% | 100.00% |
| fixed_scope_v2 | neutral | 95.24% | 95.00% | 98.25% | 100.00% |
| fixed_scope_v2 | rather_than | 95.24% | 95.00% | 98.25% | 100.00% |

## Verification, kept separate from performance
| Experiment | Implementation checks passed |
|---|---:|
| results.json | 15/15 |
| prototype-results.json | 33/33 |
| extended-results.json | 173/173 |
| variable-delay-results.json | 811/811 |
| real-query-routing-results.json | 37/37 |
| embedding-routing-results.json | 27/27 |
| long-order-results.json | 205/205 |
| neuron-ablation-results.json | 245/245 |
| language-challenge-results.json | 5/5 |
| protected-slot-results.json | 15/15 |
| protected-memory-v2-results.json | 15/15 |
| fixed-routing-results.json | 31/31 |
| scope-v2-confirmation-results.json | 5/5 |

Reproduction is independent of model accuracy: original short temporal scores40/40; variable-delay scores480/480; longer-order120/120; neuron ablation150/150; both original language routers reproduced300decisions each; final fix artifacts71/71 checks, comprising45slot scores,25full memory-condition arrays and420scope-v2 probabilities/decisions. Raw records define exact scope; do not sum repeated checks as independent evidence.

Current runnable bounded prototype and root-cause interpretation: [[32 - Root Cause Fix Findings and Runnable Prototype]].

## Later E-series repair results
The latest five-seed repairs and negative confirmations are recorded in [[40 - E Series Repair Benchmark Results]] and [[41 - Latest Repair Findings and Runnable Tests]]. Earlier results above remain historical records.
