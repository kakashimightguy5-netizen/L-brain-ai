# 134 - Brain principle: sleep reverse replay breaks the reversal curse (2026-10-14 IST)
Limitation (T54, reversal curse): a model that learns 'AB>CD' cannot answer 'CD>' -> 'AB'. Note 131: L5 and the transformer both 100% forward, 0% reverse (3 seeds, 8000 steps).
Brain principle: during sleep/rest the hippocampus spontaneously reactivates recent memories, often in REVERSE order (Foster & Wilson 2006; Diba & Buzsaki 2007), and the cortex consolidates from that replay. Recall is cue-driven and gated by a familiarity signal (perirhinal 'I know this').
Implementation (rev_sleep.py, no external data in sleep): load the trained model; every 200 steps it 'dreams': all 676 two-letter cues 'xy>' are completed greedily, and only completions it is FAMILIAR with (joint prob > 0.99) are kept as memories. It then trains on those memories replayed forward and reversed (50/50), 1500 steps, LR 1e-3. The fact table is used only to evaluate and to count true vs false memories, never for training.
First attempt (free sampling without cues) produced 0 memories: the model only learned to complete cued facts, like a brain needs a cue to recall.
## Result (MEASURED, CPU, 3 seeds, all 200 facts)
| model | forward before -> after | reverse before -> after | peak reverse during sleep |
|---|---|---|---|
| L5 s0 | 100 -> 98.5% | 0.0 -> 90.5% | 94.0% |
| L5 s1 | 100 -> 98.5% | 0.0 -> 91.0% | 93.0% |
| L5 s2 | 100 -> 99.0% | 0.5 -> 91.0% | 95.0% |
| T s0 | 100 -> 94.5% | 0.0 -> 75.5% | 79.5% |
| T s1 | 100 -> 93.5% | 0.0 -> 71.0% | 81.0% |
| T s2 | 100 -> 93.5% | 0.0 -> 75.0% | 86.0% |
Means: L5 reverse 90.8%, forward 98.7%; transformer reverse 73.8%, forward 93.8%.
## Reading
- Sleep reverse replay removes most of the reversal curse in BOTH models with zero new data; it is a learning procedure, not an L5-only architecture feature.
- L5 benefits more (+17 points reverse, +5 forward vs the transformer). Initial dream pools: L5 254 memories (200 true, 54 false) vs T 304 (200 true, 104 false): L5's familiarity is more selective, so it replays fewer false memories.
- Failure mode seen: memory drift. Replaying false memories slowly corrupts true ones (T pool: 200 true -> ~150-165 by the end; reverse peaks then declines). This is like reconsolidation-induced false memory in people.
- Not shown: natural-language facts, larger fact sets, other relation types, cost at scale. Sleep cost (MODELLED): 676 cue completions per dream + 1500 small training steps.
## Next
Stop the drift: protect true memories during sleep (noise-distillation anchor from note 133, or replay only memories that stay familiar across dreams); multi-hop over reversed facts; mix sleep with continued wake learning.
