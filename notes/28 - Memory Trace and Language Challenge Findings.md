# Memory trace and language challenge findings

[[25 - State Tracing and Language Challenge Protocol]] · [[26 - State Tracing and Neuron Ablation Results]] · [[27 - Independent Language Challenge Results]] · [[08 - SpikingBrain Process Log]]

## What the trace says
Before the second signal, independently trained diagnostic probes recover the first signal from frozen states as follows (five-seed means). Probe labels and scaling are separate diagnostic training, not learned original readouts.

| Frozen model | State examined | First-sign probe accuracy |
|---|---|---:|
| RNN | hidden | 98.59% |
| LIF_spikes | membrane | 81.48% |
| LIF_spikes | instantaneous_spikes | 61.29% |
| LIF_membrane | membrane | 91.29% |
| LIF_membrane | instantaneous_spikes | 54.88% |

The membrane can hold information that instantaneous spikes expose poorly. This supports testing readout/state choices; it does not prove one unique cause. Fixed-position paired pulses differ from the original random-position task. After the second pulse, a probe can infer the first sign from the opposite second sign, so post-second probe success does not prove original memory survives. Original traced final logits match model forward; all raw scores/distances and trace plots in note26.

## Fresh five-seed neuron comparison
All models trained on16/32/64, same120updates and validation-selected checkpoints; fresh independent inputs. Below test96 with noise .1; complete .2noise and192 results/ranges in note26. All LIF configurations start from identical weights perseed;722params versus RNN746. Membrane output is hybrid analog.

| Model/configuration | Event-order mean | Seed range |
|---|---:|---|
| RNN | 98.28% | 96.09-100.00% |
| spikes_base | 51.84% | 48.83-54.69% |
| membrane_base | 78.20% | 71.29-83.20% |
| membrane_decay99 | 66.25% | 50.78-81.25% |
| membrane_hard_reset | 75.74% | 67.19-83.98% |

Same-seed comparison against membrane_base at96/noise .1: [{'variant': 'membrane_decay99', 'improved_seeds': 1, 'equal_seeds': 0, 'worse_seeds': 4, 'percentage_point_deltas': [-7.2265625, -24.8046875, 9.9609375, -27.734375, -9.9609375]}, {'variant': 'membrane_hard_reset', 'improved_seeds': 3, 'equal_seeds': 0, 'worse_seeds': 2, 'percentage_point_deltas': [-15.0390625, -3.3203125, 4.296875, 0.9765625, 0.78125]}]. Changes are fixed pre-registered interventions, not hyperparameters tuned on test. Any recommendation from this exploratory comparison needs separate fresh confirmation before treating the candidate as generally reliable. Preserve all states/results, including failures. These are tiny custom synthetic networks, not full pretrained SpikingBrain.

## Language challenge
100 authored task families,300 texts; neutral/reframed/negated distractor variants. Labels/texts frozen before evaluation, no exact source/prior-probe overlap. Both frozen seed42 routers keep saved thresholds; no retraining, tuning or rules. Correct decision includes correct rejection of unsupported tasks; rejecting supported tasks is unanswered, not correct.

| Router | Style | Correct routing/rejection | Supported acceptance |
|---|---|---:|---:|
| lexical | neutral | 69.0% | 50.0% |
| lexical | negation | 27.0% | 12.0% |
| embedding | neutral | 64.0% | 46.0% |
| embedding | negation | 41.0% | 4.0% |

![[SpikingBrain Assets/language-challenge-comparison.png]]

Templates within a family are dependent; this is a larger controlled diagnostic, not300 independent real-user samples. Reframing uses fixed wrappers, not diverse paraphrases. Unsupported credit-card labels reflect our support boundary policy. The embedding router's100% accepted accuracy on negation comes from only2 accepted supported examples, out of50; coverage4%. It therefore does not establish reliable negation understanding. Every question/decision/failure is archived.

## Verification and saved records
Neuron implementation controls 245/245; all150 saved memory test scores reproduced after disk reload. Language structural/probability checks 5/5; both saved classifiers and cached features independently reproduce all300 decisions each. Archive hashes verified; [[SpikingBrain Assets/trace-challenge-manifest.json]]. All new scripts, states, cache, figures and results on D:. Existing CPU Python/local Ollama reused. No installations, new model downloads, power claims or generated-answer benchmark.

## Next decision
Use the RNN control and full seed/noise/length results to judge improvements, not a single favorable run. Keep domain-column growth on hold until temporal reliability improves. Language routing still needs training/evaluation that distinguishes negated distractors and out-of-domain requests; a new larger model alone is not an established fix. Any new rule/model must use a separate development set, with this challenge retained as an already-seen diagnostic.

## Current choice
Keep the existing decay0.95/subtractive-reset membrane readout as the spiking baseline. Slower decay improved only1of5 runs, while hard reset improved3of5 but reduced mean accuracy. Neither satisfies consistent improvement. The ordinary RNN remains the stronger temporal baseline. The membrane readout beat the spike-only readout in allfive fresh runs at96/noise0.1. These findings apply to this task and budget.


Completed root-cause follow-up: [[32 - Root Cause Fix Findings and Runnable Prototype]].
