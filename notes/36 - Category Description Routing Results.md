# Category-description routing results

[[34 - Weak Benchmark Improvement Protocol]] · [[31 - Unknown Class and Request Scope Fix Results]]

Root cause: lexical training has no positive examples for nine held-out intents, and confidence rejection alone cannot create category knowledge. Supply all150official CLINC intent names and domain descriptions as explicit ontology knowledge. The frozen local Nomic encoder compares requests with these descriptors. Rescue only classifier-unknown predictions whose nearest descriptor belongs to banking/travel/work and meets validation-selected similarity and margin. No held-intent query examples fit or calibrate either classifier or rescue. This is transfer to explicitly named categories, not discovery of unknown unnamed concepts; encoder and taxonomy are extra prior information.

Five original transfer baselines reproduced exactly on270held-intent and4148unsupported test queries each. Per-seed C and rescue settings chosen on known-intent and unsupported validation only. The registered supported-nearest rule selects cosine .65 and nonnegative known-minus-unknown margin; all saved strict held and unsupported decisions reproduced after disk reload (22090decisions).

| Metric | Original explicit-unknown router | Strict descriptor rescue |
|---|---:|---:|
| coverage | 37.78% | 87.85% |
| accepted_accuracy | 82.57% | 86.09% |
| correct_fraction | 31.19% | 75.63% |
| unknown_rejection | 98.62% | 94.91% |

Correct fraction is correct accepted answers divided by all supported questions; rejected questions receive no credit. Accepted accuracy remains below95% on unfamiliar test intents despite a95% validation requirement, showing calibration transfer failure. Unsupported false acceptance increases from 1.38% to 5.09%. This is a useful experimental coverage option, not a safe universal replacement for the conservative existing router.

An initial implementation permitted a supported candidate even when an unsupported descriptor ranked slightly higher (margin -.05), contradicting the registered supported-nearest wording. Kept as exploratory evidence:77.41%correct fraction,90.74%unknown rejection. Corrected the rule without retuning on test; strict validation-selected results above are the primary result. Verification initially failed due to a script variable-name collision; corrected and reran all saved decisions. Frozen descriptors, original and strict grids and decisions remain archived for review.

[[SpikingBrain Assets/ontology-routing-results.json]] · [[SpikingBrain Assets/ontology-strict-results.json]] · [[SpikingBrain Assets/ontology-prototypes.json]]
