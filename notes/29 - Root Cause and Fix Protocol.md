# Root cause and bounded fixes protocol

[[28 - Memory Trace and Language Challenge Findings]] · [[08 - SpikingBrain Process Log]]

Before fitting/testing: all new files/cache/temp/results on D:, existing libraries/models reused.

## Hypotheses supported by prior measurements
1. Final spikes expose much less retained first-signal information than membrane states. Ordinary recurrent membrane dynamics have no dedicated protected event slot; later input/noise/reset can change the state.
2. Routers force every text into three supported classes, then try to infer unknownness from a single confidence threshold. Known-topic distractors influence whole-text features. A frozen embedding alone did not repair scope or support boundaries.
These are bounded hypotheses, not proof of a unique biological mechanism.

## Memory intervention
Build an engineered first-event spike memory: positive/negative input neurons emit at a validation-selected absolute threshold; a write-once synaptic slot stores the first detected sign until reset/query. Event times and target labels are never supplied to the inference module. A small trained logistic readout maps stored spikes to task labels. This is a task-specific event-memory design, not an approximately parameter-matched end-to-end trained recurrent SNN or proof of general cognition. Preserve original LIF membrane and RNN checkpoints as unchanged controls.
Five seeds42–46; fit readout on fresh train length16/32/64,256each, noise .1. Select threshold .5/.65/.8/.9 using independent validation at32/64 and noise .1/.2; maximize mean decision accuracy, ties prefer smaller threshold. Test fresh512examples/condition at32/96/192,noise .1/.2/.3. Primary opposite-sign pulse order. Additional independent-second-sign test (same-sign and opposite-sign events), amplitude .6 shift, absent first signal and absent all signals. A write-once slot should resist second-signal overwrites; weak first events and false noise spikes remain possible. Report abstentions, rejection and accuracy together. Compare unchanged baselines on primary conditions; no test-driven threshold retuning. Save/reload every candidate artifact and decision.

## Language intervention
Train explicit fourth unknown class using available official CLINC other-domain/OOS training texts; fit word1–2gram TF-IDF max12000 plus char3–5gram TF-IDF max12000, both train-only. Five seeds42–46 sample80/examples perintent; OOS train retained. C .1/1/10 selected by four-class balanced validation accuracy. Unknown predictions reject directly; supported predictions additionally use an empirical confidence threshold selected on validation only, maximize supported coverage with accepted accuracy>=95% and unknown rejection>=90%. All empirical candidate thresholds considered, preventing coarse .99-to-reject-all gap. This differs from prior three-class training by added unknown supervision and larger feature budget; report those costs.

Add bounded request-scope extraction: remove only explicit standalone topic-denial clauses or explicit leading 'rather than discuss...' clauses when another positive clause remains; preserve intrinsic negation such as 'do not cancel my flight'. If the entire message only denies a topic, abstain. No claim of general semantic parsing. Keep exact original and selected text in results. Parser definitions frozen before new challenge evaluation.

Evaluate ordinary filtered official CLINC supported/unsupported test splits and held-out supported intents, plus newly composed contrast forms over official test queries. Contrast families hold source query fixed; dependent wrapper variants are not independent human observations. Underlying official test corpus was evaluated previously. Compare old saved lexical/embedding routers and new classifier with/without scope extraction on identical fresh wrapper texts. No new challenge labels tune classifier/parser/threshold. Also report old300-question challenge as already-seen regression, not a fresh benchmark. Save failures and runnable demo.
