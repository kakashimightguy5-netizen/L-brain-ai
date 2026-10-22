---
tags: [experiment, temporal-learning, contextual-routing, memory, bounded-growth]
created: 2026-10-05
---
# Extended prototype results

Back to [[12 - Extended Prototype Protocol]] · [[08 - SpikingBrain Process Log]]

## Scope and verification
Five seeds (42-46), CPU-only, 173/173 control checks passed. These are synthetic classifier and retrieval experiments, not pretrained SpikingBrain inference or a language model. Protocol was saved before execution. Independent train/validation/test examples; temporal model selection and routing threshold selection use validation only.

![[SpikingBrain Assets/extended-comparison.png]]

## Temporal learning
12 steps, 3 input channels; final query input identical for all examples. Recall task remembers initial sign; order task distinguishes which sign appeared first. Independent 256 train, 256 validation, 512 test examples per seed/task. Max 100 Adam updates, lr .015; best validation snapshot evaluated on test. Counts differ across model families and are reported below.

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

Memoryless model sees final input only, so 50% is expected. Analog model integrates signed currents with decay .95 and no spikes. RNN uses tanh recurrence. LIF uses recurrent spikes, decay .95, threshold 1, subtractive detached reset, sigmoid surrogate gradient and mean spikes from the last 3 steps. State resets per input. This limited LIF setup may underperform because of optimization or architecture choices; do not generalize failure to all SNNs or superiority to all tasks.

## Contextual domain routing
Prompts contain topic vocabulary for styling, querying or networking. No domain ID is supplied; this is supervised bag-of-words topical routing, not semantic language understanding. Test phrasing differs; topical word combinations are sampled from shared known vocabulary. Mixed-topic prompts are labelled ambiguous (desired action: abstain), and unseen-topic prompts have no known topical words.

| Routing metric | Mean +/- sample SD |
|---|---|
| test_accuracy_no_abstention | 100.00 +/- 0.00% |
| clean_coverage | 100.00 +/- 0.00% |
| clean_selective_accuracy | 100.00 +/- 0.00% |
| ambiguous_rejection | 96.84 +/- 1.50% |
| unknown_rejection | 100.00 +/- 0.00% |

Threshold chosen on validation to maximize clean coverage with >=95% selective accuracy and <=10% acceptance of ambiguous inputs. Unknown-topic guard explicitly checks topical-vector sum=0. Selective accuracy excludes rejected inputs; rejection and coverage must be read together. The numerical classifier can be evaluated before or after this gate; gated correct fraction penalizes abstentions rather than hiding them.

## Bounded growth and forgetting
Domain A labels sign(x0), B its opposite, C sign(x1). Numerical features Gaussian. Router sees all three labelled topics before numerical task learning; taxonomy is known, no semantic novelty detection. Sequential train: 256 examples/domain, 80 full-batch updates/domain, independent 512 test/domain. Shared input combines features and lexical counts; expert input combines features and router probabilities.

Shared parameters: 552. Router: 57. Expert: 162. Unbounded 3-expert total with router: 543; two-slot total: 381. Expert allocation stages: [162, 324, 486].

Shared rehearsal retains 32 training examples/prior domain; bounded policy preserves A and reuses B slot for C, retaining 64 B examples. These policies have different example exposure and capacities. No automatic slot selection, pruning or universal forgetting solution is claimed.

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

Per-domain, per-stage retention and bounded-domain accuracies are in [[SpikingBrain Assets/extended-results.json]]. Unbounded experts preserve old parameters by construction. Growth does not establish transfer between domains.

## External memory integrated into prediction
600 queries/seed: 100 default predictions + 100 stored-fact queries per domain. Facts use random fresh labels and domain-qualified keys, supplied to external memory after training. Default queries use routed learned experts. Stored mode uses exact lookup and abstains when a key is absent. Memory-disabled pipeline predicts default classifier labels for every query. These are externally supplied facts, not held-out model knowledge; improvement is a retrieval effect.

| Metric | Mean +/- sample SD |
|---|---|
| without_memory_accuracy | 73.27 +/- 1.81% |
| with_memory_accuracy | 99.37 +/- 0.38% |
| stored_fact_accuracy | 100.00 +/- 0.00% |
| default_prediction_accuracy | 98.73 +/- 0.76% |
| with_memory_coverage | 100.00 +/- 0.00% |
| gated_memory_correct_fraction | 99.37 +/- 0.38% |
| gated_memory_coverage | 100.00 +/- 0.00% |

Missing novel facts abstain; new stored facts immediately work; updates alter answers without retraining; deleted facts abstain; wrong-domain keys and irrelevant records are controlled. No generalization to unstored random labels is claimed. Gated correct fraction counts rejected known/default queries as incorrect; gated coverage describes the cost of rejection.

## Controls
| Check | Status |
|---|---|
| 42 delayed_recall: identical final query | PASS |
| 42 delayed_recall: independent split content | PASS |
| 42 delayed_recall memoryless: finite logits | PASS |
| 42 delayed_recall memoryless: state reset | PASS |
| 42 delayed_recall: memoryless chance | PASS |
| delayed_recall memoryless: checkpoint reload | PASS |
| 42 delayed_recall analog: finite logits | PASS |
| 42 delayed_recall analog: state reset | PASS |
| delayed_recall analog: checkpoint reload | PASS |
| 42 delayed_recall RNN: finite logits | PASS |
| 42 delayed_recall RNN: state reset | PASS |
| delayed_recall RNN: checkpoint reload | PASS |
| 42 delayed_recall LIF: finite logits | PASS |
| 42 delayed_recall LIF: state reset | PASS |
| delayed_recall LIF: checkpoint reload | PASS |
| 42 pulse_order: identical final query | PASS |
| 42 pulse_order: independent split content | PASS |
| 42 pulse_order memoryless: finite logits | PASS |
| 42 pulse_order memoryless: state reset | PASS |
| 42 pulse_order: memoryless chance | PASS |
| pulse_order memoryless: checkpoint reload | PASS |
| 42 pulse_order analog: finite logits | PASS |
| 42 pulse_order analog: state reset | PASS |
| pulse_order analog: checkpoint reload | PASS |
| 42 pulse_order RNN: finite logits | PASS |
| 42 pulse_order RNN: state reset | PASS |
| pulse_order RNN: checkpoint reload | PASS |
| 42 pulse_order LIF: finite logits | PASS |
| 42 pulse_order LIF: state reset | PASS |
| pulse_order LIF: checkpoint reload | PASS |
| 43 delayed_recall: identical final query | PASS |
| 43 delayed_recall: independent split content | PASS |
| 43 delayed_recall memoryless: finite logits | PASS |
| 43 delayed_recall memoryless: state reset | PASS |
| 43 delayed_recall: memoryless chance | PASS |
| 43 delayed_recall analog: finite logits | PASS |
| 43 delayed_recall analog: state reset | PASS |
| 43 delayed_recall RNN: finite logits | PASS |
| 43 delayed_recall RNN: state reset | PASS |
| 43 delayed_recall LIF: finite logits | PASS |
| 43 delayed_recall LIF: state reset | PASS |
| 43 pulse_order: identical final query | PASS |
| 43 pulse_order: independent split content | PASS |
| 43 pulse_order memoryless: finite logits | PASS |
| 43 pulse_order memoryless: state reset | PASS |
| 43 pulse_order: memoryless chance | PASS |
| 43 pulse_order analog: finite logits | PASS |
| 43 pulse_order analog: state reset | PASS |
| 43 pulse_order RNN: finite logits | PASS |
| 43 pulse_order RNN: state reset | PASS |
| 43 pulse_order LIF: finite logits | PASS |
| 43 pulse_order LIF: state reset | PASS |
| 44 delayed_recall: identical final query | PASS |
| 44 delayed_recall: independent split content | PASS |
| 44 delayed_recall memoryless: finite logits | PASS |
| 44 delayed_recall memoryless: state reset | PASS |
| 44 delayed_recall: memoryless chance | PASS |
| 44 delayed_recall analog: finite logits | PASS |
| 44 delayed_recall analog: state reset | PASS |
| 44 delayed_recall RNN: finite logits | PASS |
| 44 delayed_recall RNN: state reset | PASS |
| 44 delayed_recall LIF: finite logits | PASS |
| 44 delayed_recall LIF: state reset | PASS |
| 44 pulse_order: identical final query | PASS |
| 44 pulse_order: independent split content | PASS |
| 44 pulse_order memoryless: finite logits | PASS |
| 44 pulse_order memoryless: state reset | PASS |
| 44 pulse_order: memoryless chance | PASS |
| 44 pulse_order analog: finite logits | PASS |
| 44 pulse_order analog: state reset | PASS |
| 44 pulse_order RNN: finite logits | PASS |
| 44 pulse_order RNN: state reset | PASS |
| 44 pulse_order LIF: finite logits | PASS |
| 44 pulse_order LIF: state reset | PASS |
| 45 delayed_recall: identical final query | PASS |
| 45 delayed_recall: independent split content | PASS |
| 45 delayed_recall memoryless: finite logits | PASS |
| 45 delayed_recall memoryless: state reset | PASS |
| 45 delayed_recall: memoryless chance | PASS |
| 45 delayed_recall analog: finite logits | PASS |
| 45 delayed_recall analog: state reset | PASS |
| 45 delayed_recall RNN: finite logits | PASS |
| 45 delayed_recall RNN: state reset | PASS |
| 45 delayed_recall LIF: finite logits | PASS |
| 45 delayed_recall LIF: state reset | PASS |
| 45 pulse_order: identical final query | PASS |
| 45 pulse_order: independent split content | PASS |
| 45 pulse_order memoryless: finite logits | PASS |
| 45 pulse_order memoryless: state reset | PASS |
| 45 pulse_order: memoryless chance | PASS |
| 45 pulse_order analog: finite logits | PASS |
| 45 pulse_order analog: state reset | PASS |
| 45 pulse_order RNN: finite logits | PASS |
| 45 pulse_order RNN: state reset | PASS |
| 45 pulse_order LIF: finite logits | PASS |
| 45 pulse_order LIF: state reset | PASS |
| 46 delayed_recall: identical final query | PASS |
| 46 delayed_recall: independent split content | PASS |
| 46 delayed_recall memoryless: finite logits | PASS |
| 46 delayed_recall memoryless: state reset | PASS |
| 46 delayed_recall: memoryless chance | PASS |
| 46 delayed_recall analog: finite logits | PASS |
| 46 delayed_recall analog: state reset | PASS |
| 46 delayed_recall RNN: finite logits | PASS |
| 46 delayed_recall RNN: state reset | PASS |
| 46 delayed_recall LIF: finite logits | PASS |
| 46 delayed_recall LIF: state reset | PASS |
| 46 pulse_order: identical final query | PASS |
| 46 pulse_order: independent split content | PASS |
| 46 pulse_order memoryless: finite logits | PASS |
| 46 pulse_order memoryless: state reset | PASS |
| 46 pulse_order: memoryless chance | PASS |
| 46 pulse_order analog: finite logits | PASS |
| 46 pulse_order analog: state reset | PASS |
| 46 pulse_order RNN: finite logits | PASS |
| 46 pulse_order RNN: state reset | PASS |
| 46 pulse_order LIF: finite logits | PASS |
| 46 pulse_order LIF: state reset | PASS |
| 42 routing: unknown topic guard | PASS |
| 42 growth: previous experts untouched | PASS |
| 42 capacity: at most two slots | PASS |
| 42 capacity: A remains frozen | PASS |
| 42 memory: novel unstored facts abstain | PASS |
| 42 memory: wrong-domain key abstains | PASS |
| 42 memory: irrelevant record has no effect | PASS |
| 42 memory: pipeline fact update | PASS |
| 42 memory: pipeline deletion abstains | PASS |
| 42 memory: newly stored fact immediately available | PASS |
| Context router checkpoint reload | PASS |
| Domain 0 expert checkpoint reload | PASS |
| Domain 1 expert checkpoint reload | PASS |
| Domain 2 expert checkpoint reload | PASS |
| 43 routing: unknown topic guard | PASS |
| 43 growth: previous experts untouched | PASS |
| 43 capacity: at most two slots | PASS |
| 43 capacity: A remains frozen | PASS |
| 43 memory: novel unstored facts abstain | PASS |
| 43 memory: wrong-domain key abstains | PASS |
| 43 memory: irrelevant record has no effect | PASS |
| 43 memory: pipeline fact update | PASS |
| 43 memory: pipeline deletion abstains | PASS |
| 43 memory: newly stored fact immediately available | PASS |
| 44 routing: unknown topic guard | PASS |
| 44 growth: previous experts untouched | PASS |
| 44 capacity: at most two slots | PASS |
| 44 capacity: A remains frozen | PASS |
| 44 memory: novel unstored facts abstain | PASS |
| 44 memory: wrong-domain key abstains | PASS |
| 44 memory: irrelevant record has no effect | PASS |
| 44 memory: pipeline fact update | PASS |
| 44 memory: pipeline deletion abstains | PASS |
| 44 memory: newly stored fact immediately available | PASS |
| 45 routing: unknown topic guard | PASS |
| 45 growth: previous experts untouched | PASS |
| 45 capacity: at most two slots | PASS |
| 45 capacity: A remains frozen | PASS |
| 45 memory: novel unstored facts abstain | PASS |
| 45 memory: wrong-domain key abstains | PASS |
| 45 memory: irrelevant record has no effect | PASS |
| 45 memory: pipeline fact update | PASS |
| 45 memory: pipeline deletion abstains | PASS |
| 45 memory: newly stored fact immediately available | PASS |
| 46 routing: unknown topic guard | PASS |
| 46 growth: previous experts untouched | PASS |
| 46 capacity: at most two slots | PASS |
| 46 capacity: A remains frozen | PASS |
| 46 memory: novel unstored facts abstain | PASS |
| 46 memory: wrong-domain key abstains | PASS |
| 46 memory: irrelevant record has no effect | PASS |
| 46 memory: pipeline fact update | PASS |
| 46 memory: pipeline deletion abstains | PASS |
| 46 memory: newly stored fact immediately available | PASS |
| Saved checkpoint disk round trip | PASS |

## Files and rerun
- Script: D:\AI\SpikingBrain-study\experiments\run_extended.py
- Results: [[SpikingBrain Assets/extended-results.json]]
- Checkpoint: D:\AI\SpikingBrain-study\experiments\extended-prototype-seed42.pt (our toy trained models, not pretrained SpikingBrain)
- Rerun: `Set-Location 'D:\AI\SpikingBrain-study\experiments'; py -3.12 -B .\run_extended.py`
- PyTorch 2.14.0+cpu, CPU threads 2. New temp/cache on D:, bytecode writes disabled. Existing Python libraries read from C:, no installation there.
- Recorded run duration 16.2 s. Script SHA256 recorded in JSON.

## Interpretation constraints
Read accuracy with parameter count, example exposure, routing coverage and replay policy. Five seeds are descriptive evidence, not a formal significance claim. Synthetic lexical prompts are not semantic routing. No power measurements, GPU benchmark, autonomous growth, real language generation or full SpikingBrain inference occurred.
