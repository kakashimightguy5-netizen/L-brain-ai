---
tags: [protocol, temporal-learning, routing, memory, bounded-growth]
created: 2026-10-05
---
# Extended prototype experiment protocol

Back to [[08 - SpikingBrain Process Log]] · [[11 - Two-Domain Prototype Results]]

Registered before execution. CPU-only, two threads, seeds 42-46, all new files/temp/cache on D:. These are small synthetic architecture experiments, not SpikingBrain language-model inference.

## Temporal tasks
12 timesteps, 3 input channels. Task 1: recall the first signed pulse after distractors. Task 2: identify which sign occurred first in a positive/negative pulse pair. Last timestep contains a query cue, not the label. Input final timestep does not directly encode earlier pulses. Independent train/validation/test datasets (256/256/512).
Compare memoryless last-input MLP, analog leaky integrator, tanh recurrent network and recurrent LIF with decay .95, threshold 1, subtractive reset, surrogate derivative. Hidden width 24. Report unequal parameter counts. Train up to 100 full-batch Adam updates, lr .015; retain best validation checkpoint (evaluated each 10 epochs), then evaluate test once. No test-based architecture selection. LIF readout uses mean spikes over last 3 steps, not all historical spikes.

## Three-domain contextual routing
Supervised lexical topic classifier over web styling, database querying and networking prompts, without a domain-ID input. Two relevant words per clean prompt and an occasional unrelated topical word. Test prompt wording differs from training. This tests lexical contextual cues, not semantic language understanding. Mixed-topic ambiguous queries and out-of-vocabulary prompts are separately evaluated.
Select abstention threshold on validation only from .55,.65,.75,.85,.90,.95,.99,1.1, maximizing clean coverage subject to <=10% acceptance of ambiguous queries and >=95% correctness among accepted clean queries. A no-topical-word guard abstains on unknown vocabulary; this is an explicit heuristic.

## Sequential tasks and capacity
Each domain has 4 Gaussian numerical features and a different label rule: A sign(x0), B opposite sign(x0), C sign(x1). Router probabilities provide context. Experts input numerical features plus 3 router probabilities. Compare shared sequential network, shared rehearsal, unbounded independently trained experts and bounded two-slot experts. Shared network consumes original lexical topic counts, not a supplied domain ID. Actual parameter counts reported.
Independent train/validation/test numerical features (256/256/512/domain). Sequential updates: 80/domain, lr .02. Shared rehearsal retains 32 training examples per prior domain. Capacity limit: 2 experts; preserve A, reuse B slot for C with 64 B training examples during C learning. Report the explicit slot policy and replay-size differences; this is not automatic novelty detection or optimal pruning. Router is trained with all 3 topic labels before task learning, so taxonomy is known in advance.

## Memory integrated into predictions
Mixed query workload: 100 default-rule predictions plus 100 stored-fact queries per domain. Facts have fresh randomly generated labels and domain-qualified keys, stored externally after classifier training. Compare same expert pipeline with memory enabled vs disabled. For stored-fact mode, missing keys abstain; default mode uses expert prediction. Evaluate absent keys, new stored keys, wrong domains, irrelevant records, fact updates and deletion. Stored-fact lookup correctness demonstrates integration, not generalization to unstored random labels.

## Measurements and records
Per-seed accuracy, retention after each domain, routing coverage/selective accuracy/rejection, memory workload accuracy/coverage, allocated parameters and temporal inference batch times. Report mean and sample SD, not statistical significance. Structural controls validate state reset, gradients, split separation, bounded allocation, frozen state, memory behavior and checkpoint reload. Save scripts, raw JSON, small trained checkpoints and figures in vault Assets and D:\AI\SpikingBrain-study\experiments. No claims of power savings, model novelty or solved forgetting.

## Exploratory temporal stress protocol (after primary results)
- Primary short temporal tasks were near-perfect for all stateful models. Stress the saved seed-42 models without retraining or changing thresholds.
- Both tasks at 12,24,48,96 timesteps, with signal-channel distractor noise SD 0 or 0.2. 512 balanced examples/condition from fresh seeds.
- These are out-of-distribution delay/noise checks on one checkpoint, not five-seed evidence or model selection. Report all conditions including failures.
- Prepare an inference-only demo and verify serialized models against original seed-42 primary results. No retraining needed for delivery.


## Replication of temporal stress (registered after seed-42 exploration)
- Seed-42 LIF retained delayed-sign recall at 96 steps but lost long-delay pulse-order accuracy. Replicate the full fixed delay/noise grid across all five original training seeds.
- Recreate identical original train/validation-selected temporal checkpoints without changing architecture or hyperparameters. Verify original short-task test scores match for all 40 model/task/seed combinations.
- Evaluate 512 fresh examples for each seed, task, length and noise level. No condition or seed excluded; report means and sample SD. This remains exploratory, not independently designed confirmation.

