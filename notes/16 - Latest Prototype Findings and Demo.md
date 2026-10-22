---
tags: [research-summary, prototype, next-steps]
created: 2026-10-05
---
# Latest prototype findings and runnable demo

Back to [[00 - Concept and Roadmap]] · [[13 - Extended Prototype Results]] · [[15 - Temporal Stress Replication]] · [[08 - SpikingBrain Process Log]]

## Current status
Working Windows CPU experiments and an inference-only routing/expert/memory demo. Not a trained language model; no pretrained SpikingBrain weights or full-model generation used. Routing/growth/memory experts are ordinary MLPs; recurrent LIF temporal experiments are separate.

| Experiment | Five-seed result | Interpretation |
|---|---|---|
| Short delayed recall | LIF 99.84%; analog/RNN 100%; memoryless 50% | State matters, spiking superiority not demonstrated |
| Short pulse ordering | Stateful models 100%; memoryless 50% | Easy at training horizon |
| Three-domain shared sequential model | 69.40% final accuracy | Sequential interference |
| Shared + replay | 78.52% | Earlier examples improve retention |
| Three frozen routed experts | 98.37% | Strong controlled-domain separation |
| Two-slot capacity limit | 91.63% | Explicit slot reuse + replay reduces accuracy |
| Default + supplied-fact workload | 73.27% without memory; 99.37% with memory | External lookup supplies information absent from model weights |
| Topic-word routing | 100% clean routing; 96.84% ambiguous rejection | Lexical cues, not semantic language understanding |

Shared parameters 552; three experts + router 543; two experts + router 381. External record storage is additional and not included in parameter counts. Replay/data exposure differ across controls; no definitive benchmark ranking.

## Longer-delay stress
All five original temporal seeds reproduced with unchanged settings. At 96 steps:
- LIF recall: 69.77% +/- 27.60 percentage-point SD without noise; 70.31% +/- 26.69 with signal-noise SD .2.
- LIF pulse ordering: 47.70% +/- 17.07 without noise; 50.08% +/- 13.50 with noise, approximately chance.
- Analog and simple RNN also fail or vary. One favourable LIF checkpoint does not establish reliable memory.
- Out-of-distribution stress designed after primary results; exploratory. No stress-based tuning. Tables show SD; final stress plot shows full observed seed ranges.

## Checks
173/173 primary controls; 40/40 recreated original temporal scores; all 320 stress outputs finite; 9/9 integrated inference-demo checks passed. Finite logits do not mean accurate predictions. Raw data contains every seed and condition.

## Runnable demo
D:\AI\SpikingBrain-study\experiments\try_extended.py loads our saved toy expert/router state and editable demo_memory.json. Existing CPU PyTorch; no installation needed. All new files/temp/cache on D:, bytecode disabled.

Synthetic numerical classification:
```powershell
py -3.12 -B 'D:\AI\SpikingBrain-study\experiments\try_extended.py' --prompt 'inspect button border' --features 1 0 0 0
```
Externally stored label:
```powershell
py -3.12 -B 'D:\AI\SpikingBrain-study\experiments\try_extended.py' --prompt 'inspect rows table' --mode stored --key override-label
```
Unknown-topic example: `--prompt 'orchestra violin sonata'`. Ambiguous example: `--prompt 'style query'`. Both abstain in the demo. Missing stored keys abstain; updates change answers without training. Outputs are synthetic labels, not natural-language explanations. Verification changes memory in-process only; saved demo_memory.json is unchanged.

## Assessment
Prioritize contextual routing, explicit memory and controlled module reuse. Spiking remains a research branch: temporal behavior works but long-horizon retention is variable, and CPU speed/energy advantages have not been established. Frozen-module retention is expected because old weights are untouched. No novelty or general forgetting solution proved.

## Next proposal, not executed in this round
1. Train with varying delays and compare matched-budget GRU/LSTM models before interpreting long-horizon SNN advantage.
2. Evaluate held-out topic synonyms, negation, real labelled queries and unknown-domain abstention.
3. Compare slot selection/reuse under fixed parameter and replay budgets.
4. Replace exact toy facts with versioned documents and evidence retrieval; test stale/conflicting information.
5. Measure actual time and memory overhead before scaling.

## Records
[[12 - Extended Prototype Protocol]] · [[13 - Extended Prototype Results]] · [[14 - Temporal Stress Results]] · [[15 - Temporal Stress Replication]]. Scripts, results, model states, plots and provenance manifest are copied into SpikingBrain Assets. Toy states are not the SpikingBrain 5B/7B checkpoints.

Latest follow-up: [[20 - Latest Memory and Routing Findings]] (2026-10-06).
