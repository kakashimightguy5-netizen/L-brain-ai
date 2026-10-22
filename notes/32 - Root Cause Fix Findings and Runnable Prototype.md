# Root-cause fixes and runnable prototype

[[29 - Root Cause and Fix Protocol]] · [[30 - Protected Event Memory Fix Results]] · [[31 - Unknown Class and Request Scope Fix Results]] · [[08 - SpikingBrain Process Log]]

## What changed and why
Memory: prior tests showed first-signal information in membrane states that current spikes did not expose well. Changing decay/reset alone failed to improve consistently. Added a protected first-event synaptic slot: positive/negative threshold neurons write once, later events cannot overwrite it. This is an engineered task-specific fix, not an end-to-end learned recurrent SNN or full pretrained SpikingBrain. No target sign or true event time is given at inference. A small logistic readout is trained; event threshold is validation-selected.

Language: three-way classifiers had no directly trained unknown class and processed negated distractors together with desired requests. Added explicit unsupported training examples, word/character features, validation-only empirical calibration, and bounded request-scope extraction. Version1 failed suffix denials without punctuation and explicit answer markers; version2 fixes those forms. Intrinsic negation such as 'Do not cancel my flight' remains. These rules do not parse unrestricted language.

![[SpikingBrain Assets/root-fix-comparison.png]]

## Memory evidence and limits
At96steps/noise .2, unchanged old membrane averaged69.06%, RNN74.30%, first-event slot v199.61%. Slot v1 also handles independent second signs, preventing the opposite-sign shortcut. This is a specialized model/data/validation-budget comparison, not proof of spiking superiority. Weak amplitude .6 events fail detection. High noise can create false events.

Version2 adds an exactly-two-event completeness contract and was confirmed on fresh generated inputs without threshold retuning. Mean correct answer fraction97.30%, coverage97.30%, accepted accuracy100% for normal and independent-second cases. Missing-first rejection97.34%, no-event rejection99.96%; weak-signal rejection99.96% means almost all weak inputs remain unanswered, not successfully classified. Do not count those abstentions as memory success. Cannot use the two-event contract for arbitrary event counts. Full perseed/noise/length/stress results in note30. Keep both original recurrent controls for broader future tasks.

## Language evidence
Official familiar-intent evaluation across five seeds:
- Supported coverage: 84.37% old lexical to 95.13% fixed four-class.
- Correctly answered fraction: 84.36% to 94.74%.
- Accepted accuracy: 99.60% fixed.
- Unsupported rejection: 90.95% to 98.23%.
Four-class raw accuracy counts unknown predictions as errors on supported examples, unlike earlier forced-three-class routing; compare answered fraction/coverage rather than equating those raw metrics.

On420 newly authored confirmation wrappers over84source families, scope v2 correct routing/rejection 95.24% vs old lexical 44.05%. Scope v1 scored84.29%. Version2 classifier weights/threshold unchanged; only scope logic changed. Confirmation base queries are disjoint from scope-development families. Underlying official test corpus was used previously, and wrappers within family are dependent; not420 independent real-user samples. Confirmation distractors avoid the desired supported domain to prevent contradictory framing. Accepted accuracy98.25%, supported coverage95.00%, unsupported rejection100.00% on this chosen subset, not a universal guarantee.

Held-out intents remain weak: fixed correct-answer fraction31.19%, coverage37.78%, accepted accuracy82.57%. Raw four-class accuracy31.19%; this includes unknown outputs as failures. The scope correction does not solve unfamiliar topic generalization. Extra unknown supervision/feature count differ from prior classifiers. Plain embedding replacement was not the successful fix.

## Runnable bounded prototype
`py -3.12 -B "D:\AI\SpikingBrain-study\experiments\try_fixed_prototype.py" --query "I am not asking about my bank balance. Instead, I want to know my paid time off balance." --events "0,1,0,-1,0"`

Uses local persisted classifier and protected-event module. Does not require a new model download or Ollama for its own inference. Predicts a topic/abstains and remembers the first distinguishable signal under its two-event contract; does not generate answers.

## Verification and records
71/71 fresh-process artifact checks reproduced all45 slot primary scores, all25version2 condition decision arrays, and all420scope-v2 probabilities/decisions. Archive hashes verified, manifest [[SpikingBrain Assets/root-fix-manifest.json]]. All new files/data/cache/temp on D:, existing libraries reused. Records include failures and version1development outcomes. Behavioral improvements are separate from implementation checks. Do not overwrite previous baselines or interpret this as full SpikingBrain inference.
