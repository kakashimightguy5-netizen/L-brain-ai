---
tags: [experiment, temporal-stress, robustness]
created: 2026-10-05
---
# Temporal stress results

Back to [[13 - Extended Prototype Results]] · [[12 - Extended Prototype Protocol]]

The original short temporal tasks were easy for stateful baselines, so we tested the saved seed-42 checkpoint at longer delays and additional signal-channel noise. **One training seed only; exploratory distribution shift, not model-selection or five-seed evidence.** No retraining. 512 fresh balanced examples/condition. Original checkpoint inference matches all 8/8 primary seed-42 temporal scores. All stress logits finite: True.

![[SpikingBrain Assets/temporal-stress.png]]

## Complete condition results
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

## Interpretation
Models were trained at length 12 with no distractors in the signal channel. Longer lengths and noise change the input distribution. A model can succeed at the training horizon and fail beyond it. RNN and LIF can retain information through recurrent state; analog decay can attenuate earlier evidence. This test is descriptive and does not show general SNN superiority or infer power savings.

## Reproduce
- D:\AI\SpikingBrain-study\experiments\run_temporal_stress.py
- [[SpikingBrain Assets/temporal-stress-results.json]]
- `py -3.12 -B 'D:\AI\SpikingBrain-study\experiments\run_temporal_stress.py'`
