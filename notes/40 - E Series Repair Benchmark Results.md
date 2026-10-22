# E-series repairs: complete results

[[37 - E Series Repair Protocol]] · [[38 - E Series Follow Up Confirmation Protocol]] · [[39 - Perception Repair and Memory Audit Protocol]] · [[41 - Latest Repair Findings and Runnable Tests]]

The named failures have all been investigated. Some mechanisms improved, some remained weak, and the simplest baseline was best in several comparisons. All results below are from native CPU PyTorch prototypes on 8×8 handwritten digits. They do not establish full SpikingBrain 5B/7B fine-tuning or language-model-scale performance.

Each experiment uses five seeds and stratified 60/20/20 training, validation and test partitions (1,078/359/360 images). Seeds 142–146 cover initial repairs, 242–246 follow-up confirmation, and 342–346 perception confirmation. Checkpoints and settings use validation only. All 15 partitions passed checks for disjoint image identities and exact pixel content. The underlying 1,797-image dataset is reused, so these are repeated experiments rather than independently collected datasets. Tables report mean ± sample standard deviation. Compare methods within each experiment, not across the seed groups.

## E1: native continual fine-tuning and EWC; E6 integration

| Method | Continual accuracy | First task retained | Clean | Noisy |
|---|---:|---:|---:|---:|
| E1: Naive native fine-tuning | 19.11 ± 0.46% | 0.00 ± 0.00% | — | — |
| E1: Original EWC estimator | 19.33 ± 0.25% | 0.00 ± 0.00% | — | — |
| E1: Replay (20 per class) | 92.06 ± 0.80% | 85.83 ± 3.32% | — | — |
| E1: Corrected EWC, validation selected | 20.33 ± 2.00% | 13.06 ± 20.52% | — | — |
| E1: EWC + replay, reused lambda | 64.11 ± 29.86% | 57.50 ± 32.73% | — | — |
| E6: Shared columns before sleep | — | — | 92.17 ± 1.12% | 69.28 ± 2.24% |
| E6: Shared columns after sleep | — | — | 92.50 ± 1.27% | 76.00 ± 1.22% |
| E6: Monolithic baseline before sleep | — | — | 91.94 ± 1.42% | 71.06 ± 2.59% |
| E6: Monolithic baseline after sleep | — | — | 93.72 ± 1.50% | 75.83 ± 1.92% |

The old EWC implementation squared an averaged gradient, allowing cancellation. The corrected version averages squared per-example gradients from 64 training examples per task. Fixing the calculation did not fix class-incremental forgetting: corrected EWC remains at 20.33%, versus replay at 92.06%. Native naive fine-tuning still forgets earlier classes. Replay supplies earlier examples; its improvement must not be attributed to EWC alone.

The first EWC + replay attempt reused a lambda calibrated without replay, overconstraining several runs. The follow-up below selects lambda for the actual replay setting and allows zero regularization.

E6 uses a shared feature trunk, five two-class heads and a domain router whose log probabilities are trained together. It has 9,295 parameters versus 9,382 in the monolithic baseline. Both receive the same 20 examples per class, augmentation budget, and union of externally corrected stream mistakes. Shared noisy accuracy rises from 69.28% to 76.00%; the matched baseline reaches 75.83%. This is a near tie, not evidence of clear superiority. Clean scores and saved hard-routing, oracle-routing and expert diagnostics remain in the JSON. Domain arrival is scheduled from labels: this is not autonomous novelty growth or an integration of every proposed module.

## E2: healing, direct edits, consolidation and intrinsic correction

| Method | Clean accuracy | Unseen noisy accuracy | Stored corrections retained |
|---|---:|---:|---:|
| No healing | 96.28 ± 0.91% | 55.22 ± 3.66% | 0.00 ± 0.00% |
| Aggressive direct edits | 47.67 ± 10.10% | 30.72 ± 4.35% | 35.20 ± 4.89% |
| Native error-only fine-tuning | 92.89 ± 2.09% | 61.83 ± 5.89% | 43.06 ± 29.20% |
| Protected direct edits | 95.50 ± 1.14% | 57.28 ± 3.79% | 8.74 ± 4.98% |
| Original consolidation | 93.83 ± 2.07% | 65.61 ± 2.70% | 68.73 ± 5.03% |
| First protected consolidation attempt | 95.94 ± 1.09% | 63.17 ± 6.44% | 30.17 ± 18.16% |
| Robust training before healing | 96.44 ± 1.20% | 73.06 ± 2.87% | 49.07 ± 4.08% |
| Original intrinsic second guess | — | 54.44 ± 2.99% | — |
| Intrinsic smoothing | — | 55.44 ± 3.38% | — |
| Exact correction memory | 96.28 ± 0.91% | 55.22 ± 3.66% | 100.00 ± 0.00% |

Aggressive direct edits collapse clean skill. Smaller clipped edits with clean replay and teacher anchors preserve clean accuracy at 95.50%, but correct only a small fraction of the deployment mistakes. Native error-only fine-tuning improves noisy accuracy while losing some clean accuracy. Exact memory restores every stored correction and leaves unseen accuracy unchanged; exact lookup is not generalization.

The original exact-memory audit used a tiny distance threshold and missed identical queries because of floating-point cancellation. Exact float32 byte keys fixed this implementation error. All five correction sets now recover 100%. The original results and script are retained beside the corrected versions.

The first protected sleep attempt had two avoidable problems: it distilled the old teacher on corrected errors, encouraging its old wrong answers, and corrupted already noisy errors again. The follow-up distills only clean anchors and augments only original clean examples.

The old second-guess self-correction breaks more answers than it repairs. Smoothing adds only 0.22 percentage points over no healing on average and harms some individual runs. It uses labelled offline validation to choose its strength but receives no labels at inference. This is not reliable autonomous truth detection. Robust training before deployment reaches 73.06% noisy accuracy with 96.44% clean accuracy, using existing training labels and synthetic corruption. The unchanged no-healing control remains at 55.22%.

## E3: effort gate and perception repair

The initial calibrated loop averages 3.92 of eight steps, but its 62.89% accuracy trails its own one-step readout (63.78%). Eight fixed steps reach 61.94%. All five actual active-batch executions matched simulated halting decisions.

Fresh confirmation replaces unhelpful extra loops with learned denoising and a matched robust native classifier:

| Method | Mixed-noise accuracy | Noise 0.35 accuracy | Cheap exits |
|---|---:|---:|---:|
| Clean teacher | 61.17 ± 2.68% | 56.28 ± 3.93% | — |
| Learned denoising | 71.50 ± 4.66% | 74.44 ± 1.00% | — |
| Robust native fine-tuning | 72.83 ± 3.38% | 75.72 ± 2.16% | — |
| Raw/denoised effort gate | 71.06 ± 4.80% | 73.83 ± 1.95% | 15.78 ± 23.42% |

Robust native fine-tuning improves fixed-noise accuracy from 56.28% to 75.72% and mixed-noise accuracy from 61.17% to 72.83%. It uses the same 8,970-parameter architecture as the teacher, with an additional 60 training epochs on corrupted training examples. Denoising and robust fine-tuning receive the same additional training budget. Denoising helps, but adds a 16,576-parameter module. The effort gate adds a preliminary classifier pass and rarely exits early.

Serialized CPU timings after all training jobs and demos completed, batch size 360, two threads, five warmups and 30 repetitions per model:

| Method | Mean of five seed median batch times |
|---|---:|
| One loop | 0.357 ms |
| Eight loops | 1.702 ms |
| Active loop gate | 2.753 ms |
| Robust native classifier | 0.152 ms |
| Denoising + classifier | 0.343 ms |
| Raw/denoised gate | 0.752 ms |

Windows background activity remains uncontrolled; these are CPU batch timings, not energy measurements. Step savings do not translate into runtime savings here. Use the one-pass robust classifier for this perception task. Reasoning tasks may benefit from extra effort, but that is a separate benchmark.

## E5: genome compression

| Method | Accuracy | Parameters | Checkpoint bytes |
|---|---:|---:|---:|
| Uncompressed teacher | 96.72 ± 0.72% | 9610 | 40971 |
| fixed8_CE | 92.83 ± 1.23% | 2558 | 110393 |
| fixed8_distill | 91.83 ± 2.81% | 2558 | 110587 |
| code4_CE | 85.50 ± 4.36% | 1683 | 11269 |
| code4_distill | 83.94 ± 5.32% | 1683 | 11413 |
| lowrank5_distill | 93.06 ± 1.26% | 1788 | 10437 |

The fixed-ID genome has fewer trainable parameters, but its explicit Fourier feature buffers make its saved checkpoint larger than the teacher. These buffers could instead be regenerated from a recipe; the reported files include them for standalone exact reproduction. Parameter counts alone are not storage compression. Pure learned-code genomes still underfit and vary by seed; teacher distillation alone did not resolve this.

The fresh follow-up initializes low-rank factors from the teacher's learned weights, then distills using training data only. A separate hybrid adds a learned nonlinear genome residual:

| Method | Accuracy | First task retained | Clean | Noisy | Parameters | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|
| EWC_replay: EWC_replay_lambda0 | 90.11 ± 2.03% | 81.39 ± 2.52% | — | — | — | — |
| EWC_replay: EWC_replay_lambda10 | 90.67 ± 1.65% | 81.67 ± 4.33% | — | — | — | — |
| EWC_replay: EWC_replay_lambda1000 | 87.06 ± 2.27% | 78.89 ± 6.76% | — | — | — | — |
| EWC_replay: EWC_replay_lambda100000 | 43.28 ± 12.16% | 58.06 ± 30.51% | — | — | — | — |
| EWC_replay: EWC_replay_selected | 90.83 ± 1.46% | 81.94 ± 3.40% | — | — | — | — |
| sleep: No healing | — | — | 96.78 ± 0.80% | 55.78 ± 5.02% | — | — |
| sleep: Original consolidation | — | — | 93.00 ± 0.99% | 66.83 ± 1.99% | — | — |
| sleep: Repaired consolidation | — | — | 94.67 ± 1.10% | 70.33 ± 0.91% | — | — |
| compression: Uncompressed teacher | 96.78 ± 0.80% | — | — | — | 9610 | 41115 |
| compression: SVD initialized rank 5 | 94.67 ± 1.23% | — | — | — | 1788 | 10449 |
| compression: SVD initialized rank 8 | 96.72 ± 0.93% | — | — | — | 2778 | 14417 |
| compression: Rank 5 + genomic residual | 95.50 ± 0.97% | — | — | — | 2652 | 17734 |

Rank 8 retains 96.72% accuracy versus 96.78% for the teacher, with 2,778 parameters versus 9,610 and 14,417 bytes versus 41,115 (3.46× fewer parameters; 2.85× smaller checkpoint). The hybrid genome reaches 95.50% versus 94.67% for rank 5, but loses to rank 8 in both accuracy and checkpoint bytes. It is a genome-plus-factorization hybrid, not proof that the pure nonlinear genome is now solved.

## Follow-up consolidation and replay calibration

Repaired consolidation reaches 70.33% noisy accuracy versus 55.78% with no healing and 66.83% with original consolidation. Clean accuracy is 94.67%, versus 96.78% before healing. The two-point clean guard was enforced on validation; it did not guarantee the same bound on unseen clean test data. Keep correction memory alongside consolidation when instant exact-query reliability matters.

Replay-calibrated EWC reaches 90.83% versus 90.11% for matched replay. Zero EWC is selected in two of five seeds and lambda 10 in three. This small gain is within seed variation; replay is the dependable component, and EWC adds importance-buffer overhead.

## Verification and reproducibility

310 saved prediction arrays reproduced exactly after disk reload, plus all five exact-memory correction sets. All 25 genome finite-value checks, five active-halting checks and 15 image-separation checks passed. Every candidate, validation choice, seed result, correction failure and checkpoint remains in the archived JSON/state files. SHA-256 copies are verified in SpikingBrain Assets.

Execution issues were repaired and recorded: a missing timing import stopped the first E2/E3 run before saving a seed, so the corrected suite was rerun; the exact-memory identity bug was corrected separately; the summary writer was corrected to handle structured timing fields. Earlier experiment evidence remains intact. All new files, temporary data, caches and vault records are on D:.
