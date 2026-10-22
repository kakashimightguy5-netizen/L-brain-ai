---
tags: [prototype, experiment, spiking, continual-learning]
created: 2026-10-05
---
# Two-domain prototype results

Back to [[10 - Two-Domain Prototype Protocol]] · [[08 - SpikingBrain Process Log]]

## Scope
A reproducible CPU toy experiment, not a language model and not the full SpikingBrain architecture. Seeds 42-46, independent train/validation/test splits, 384/256/512 examples per domain. Five seeds measure initialization/data variation; they do not establish statistical significance on real tasks.

## Main results (mean +/- sample standard deviation)
- Shared network A accuracy before learning B: 96.95 +/- 0.76%; after B: 10.86 +/- 7.32%.
- Shared forgetting: 86.09 percentage points. Frozen experts forgetting: 0.00 points.
- Rehearsal (64 retained A examples during B training): A after B 79.69 +/- 13.58%, B 97.23 +/- 1.15%, combined 88.46 +/- 7.29%.
- Joint shared-network accuracy: 97.58 +/- 0.33%; routed ANN experts: 97.42 +/- 0.41%.
- LIF experts: 97.93 +/- 0.56%; adaptive LIF: 97.91 +/- 0.65%; separately trained memoryless control: 97.99 +/- 0.47%.
- Adaptive effort: 2.06 of 8 steps on average. Fixed batch time 1.152 ms; adaptive 4.057 ms. Dynamic batching/indexing and repeated input projection are included; no claim of speed or energy benefit merely from fewer steps.
- All 33/33 structural/control checks passed if the table below contains no FAIL.

![[SpikingBrain Assets/prototype-comparison.png]]

## Model accounting
Shared network: 418 parameters. Two experts: 444 total; 222 in a routed expert, plus 4 router parameters. LIF experts: 444 parameters. Growth simulation increases allocated expert parameters from 222 to 444 on explicitly receiving domain B. The frozen A expert is identical before/after B training.

Growth and fixed expert allocation have the same trained result here. The only growth distinction is when B's parameters are allocated; semantic novelty detection, learning the router for unseen domains, pruning and capacity limits are not implemented.

## Controlled task and baselines
A predicts sign(x0+0.5*x1); B predicts its opposite on Gaussian features. Conflict intentionally stresses interference. Shared baseline has domain one-hot inputs. Experts receive 8 features and explicit or learned routing. Learned routing observes a noisy scalar domain marker; it does not discover semantic domains. Marker removal/flipping tests dependency on that cue.

Sequential shared model takes 80 full-batch updates on A then 80 on B. Joint shared model takes 160 updates on A+B and therefore processes more examples. Expert ANN and LIF use 80 updates per domain, Adam learning rate .02. The additional rehearsal baseline reuses 64 A training examples during each B update, increasing B-stage batch size to 448. It shares the exact A-trained starting state with the sequential baseline. This extra data exposure is a limitation of the comparison; no test examples enter replay. This is not a definitive ranking of continual-learning methods.

## Actual LIF model
Current=linear(input). Each step u=0.85*u+current; spike=1 if u>=1; u=u-spike. Spike rates over 8 steps feed a linear classifier. Training uses a sigmoid surrogate derivative (slope 5), with detached reset. State resets per input. An independently trained memoryless control uses identical initial weights and update counts with decay multiplier beta=0 (no membrane history). On this static task no temporal memory is intrinsically required. This implements genuine leak/threshold/reset dynamics in custom PyTorch; it is distinct from official SpikingBrain's integer count encoders and is not a snnTorch implementation.

## Effort gate
Threshold candidates .65,.8,.9,.95 and 1.1 are evaluated only on validation. Select fewest average steps within 1 percentage point of fixed-budget validation accuracy. At test time use the chosen threshold, minimum 2 steps, maximum 8. Test quality can still differ from validation; no guaranteed loss bound. The gate is confidence-based and can be confidently wrong.

## Memory controls
An isolated exact-match key/value memory holds 200 random-label facts. Known facts must be reconstructed exactly; missing, wrong-domain and unknown-domain queries abstain; update/delete behavior tested. This module is separate from the trained classifier in this experiment. Retrieval is not generalization and no integrated memory accuracy improvement is claimed.

## Raw aggregate metrics
Accuracy fields are fractions, forgetting is percentage points, times are milliseconds for a 1024-input batch.

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

## Control checks
| Check | Status |
|---|---|
| LIF hand-calculated firing sequence | PASS |
| Surrogate gradient finite and positive | PASS |
| Seed 42: old expert weights unchanged | PASS |
| Seed 42: routed vs all-module output | PASS |
| Seed 42: LIF inference resets state | PASS |
| Seed 42: effort bounds | PASS |
| Seed 42: adaptive full-budget equivalence | PASS |
| Seed 43: old expert weights unchanged | PASS |
| Seed 43: routed vs all-module output | PASS |
| Seed 43: LIF inference resets state | PASS |
| Seed 43: effort bounds | PASS |
| Seed 43: adaptive full-budget equivalence | PASS |
| Seed 44: old expert weights unchanged | PASS |
| Seed 44: routed vs all-module output | PASS |
| Seed 44: LIF inference resets state | PASS |
| Seed 44: effort bounds | PASS |
| Seed 44: adaptive full-budget equivalence | PASS |
| Seed 45: old expert weights unchanged | PASS |
| Seed 45: routed vs all-module output | PASS |
| Seed 45: LIF inference resets state | PASS |
| Seed 45: effort bounds | PASS |
| Seed 45: adaptive full-budget equivalence | PASS |
| Seed 46: old expert weights unchanged | PASS |
| Seed 46: routed vs all-module output | PASS |
| Seed 46: LIF inference resets state | PASS |
| Seed 46: effort bounds | PASS |
| Seed 46: adaptive full-budget equivalence | PASS |
| Memory retrieves 200 exact known facts | PASS |
| Memory absent key abstains | PASS |
| Memory unknown domain abstains | PASS |
| Memory wrong-domain key abstains | PASS |
| Memory fact update | PASS |
| Memory fact deletion | PASS |

## Reproduce
- Script: D:\AI\SpikingBrain-study\experiments\run_prototype.py
- Raw results: [[SpikingBrain Assets/prototype-results.json]]
- Script copied into vault: SpikingBrain Assets/run_prototype.py
- Reusable seed-42 state dictionaries: D:\AI\SpikingBrain-study\experiments\tiny-prototype-seed42.pt (our toy ANN/LIF experts and router, not SpikingBrain trained weights).
- PowerShell: `Set-Location 'D:\AI\SpikingBrain-study\experiments'; py -3.12 -B .\run_prototype.py`
- Python bytecode disabled; temp/cache on D:. Existing Python runtime read from C:. PyTorch 2.14.0+cpu, CPU threads 2.
- Run duration: 8.3 seconds. Script SHA256 included in raw results.

## Interpretation limits and next decision
Frozen expert retention is expected from untouched weights; it does not prove general catastrophic forgetting solved. Routing is only reliable when domain information is reliable. The memory and growth experiments do not establish an integrated learning architecture. Dense PyTorch simulation does not measure neuromorphic energy. LIF quality must be assessed against the ANN baseline rather than assumed superior.

Next: use three or more domains without an explicit domain marker, integrate retrieval with a real task and hold out novel keys, compare rehearsal and fixed-capacity baselines, then evaluate pruning and bounded growth. Avoid scaling until these controls succeed.


## Saved model verified and ready to try
- Reloaded saved ANN experts, LIF experts and learned router in an independent inference-only script; all 3 accuracies exactly match the seed-42 recorded results. Training controls: 33/33; reload checks: 3/3.
- Toy checkpoint, inference script and verification JSON copied into vault Assets.
- Try domain A or B: `py -3.12 -B 'D:\AI\SpikingBrain-study\experiments\try_prototype.py' --domain A --features 1 0 0 0 0 0 0 0`.
- This task is static. Memoryless control matched LIF accuracy; the experiment provides no demonstrated need for membrane history. Adaptive CPU inference was slower despite fewer steps. Routing reduced activated expert parameters but did not improve this small-batch CPU latency.

