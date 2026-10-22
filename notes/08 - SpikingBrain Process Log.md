---
tags: [spikingbrain, experiment, process-log]
created: 2026-10-05
---
# SpikingBrain process log

## Scope and storage
User authorized official neuron demonstrations, architecture comparison, tests, and vault records. All new scripts, outputs, caches and temporary files are on D:. Existing Python libraries on C: are read, not installed or modified.

## Completed preparation
1. Found active vault through Obsidian configuration.
2. Read Concept and Roadmap and Local Model Setup; found stale deletion status in setup note.
3. Downloaded and extracted official SpikingBrain-7B and SpikingBrain2.0 repositories to D:\AI\SpikingBrain-study. These are source releases, not trained weights.
4. Inspected neuron.py, quant_linear.py and the 5B model assembly. Found spike-count encoders, tensor-level fake quantization and hybrid attention/recurrent-state layers.
5. Checked Python 3.12: torch, numpy and matplotlib already available. No dependency installation needed.
6. Prepared deterministic CPU tests with seed 42. Full-model generation, training and hardware energy measurement are outside this experiment.

## Validation plan
- Exhaustive integer round trips for binary, ternary, signed bitwise and two's complement.
- Invalid-input checks, zero inputs and insufficient timesteps.
- Observe repeated-use behavior and zero-point API limitations without changing official code.
- Test official QuantLinear against an independent quantized reference and the original float linear layer.
- Compare simulated encoding steps, nonzero events and CPU encoding time; avoid interpreting them as power savings.
- Produce readable plots, machine-readable results, source hashes and a rerun command.

## Executed CPU run
- Executed official encoder/linear-layer demo with seed 42; 15/15 checks passed.
- Ran 30 measured calls per encoding after 5 warmups.
- Recorded truncation, repeated-instance, two's-complement and zero-point behavior.
- Saved two figures and raw JSON in the vault; saved runnable script and results on D:.
- Recorded SHA256 hashes of inspected official files.
- Results: [[07 - SpikingBrain Neuron Experiments]].

## Final review
- Visually inspected both generated figures: readable labels and no clipping.
- Compared official 7B model blocks and 5B SFT configuration; saved architecture note.
- Rechecked Ollama list and corrected the stale pending-removal note; preserved its previous contents in Assets.
- Copied runnable script into vault Assets for a self-contained experiment record.
- Existing PyTorch is CPU-only (2.14.0+cpu); no GPU test claimed.

- Final integrity check: all experiment-note links resolve; copied script SHA256 matches original; all 15 checks remain passing in saved results.

## Two-domain experiment preparation
- Read prior decisions and checked available RAM (~2.3 GiB).
- Registered split sizes, seeds, baselines, LIF equations, routing stress tests, memory controls and validation-only effort selection in [[10 - Two-Domain Prototype Protocol]].
- Chose small CPU models to fit current memory; no installations or full-model downloads.


## Two-domain CPU run
- Ran five seeds with sequential shared, joint shared, ANN experts, learned routing and LIF experts.
- Tested removed/flipped routing markers, bounded adaptive steps and exact-memory controls.
- Saved per-seed metrics, all 33/33 control results and chart; no trained SpikingBrain weights used.
- Result note: [[11 - Two-Domain Prototype Results]].
- Duration: 7.1 seconds.

## Follow-up review and changes
- Initial run completed with 33/33 controls; preserved initial results in Assets/prototype-initial-results.json.
- Found timing comparison included accuracy computation only in routed path; corrected both paths before interpreting timing.
- Added rehearsal and no-leak controls and genuine delayed B-module allocation; recorded follow-up protocol.
- Rerunning because implementation and controls changed.


## Two-domain CPU run
- Ran five seeds with sequential shared, joint shared, ANN experts, learned routing and LIF experts.
- Tested removed/flipped routing markers, bounded adaptive steps and exact-memory controls.
- Saved per-seed metrics, all 33/33 control results and chart; no trained SpikingBrain weights used.
- Result note: [[11 - Two-Domain Prototype Results]].
- Duration: 8.3 seconds.

## Terminology correction
- The beta=0 control removes membrane history (complete decay between steps); calling it no-leak was inaccurate. Renamed it memoryless in the protocol, results and script. Numerical computations unchanged. Initial preserved results retain the old metric label.


## Final prototype delivery verification
- Reloaded seed-42 checkpoint without training; ANN/LIF/router held-out accuracies match saved results exactly (3/3 checks).
- Executed example domain-B inference successfully; all three prediction paths return correct class 0, adaptive exits at step 2.
- Copied training script, inference script, toy checkpoint and reload results into vault Assets.
- Visually inspected final chart; labelled means and sample standard deviations are readable.


## Extended experiment preparation
- Read last results and checked available RAM (~2.4 GiB).
- Registered temporal tasks, lexical routing with validation-calibrated abstention, integrated memory and bounded two-slot growth in [[12 - Extended Prototype Protocol]].
- Defined conventional controls and disclosed replay-size, model-capacity and known-taxonomy limitations before execution.


## Extended CPU experiment execution
- Ran both temporal tasks with memoryless/analog/RNN/LIF controls, five seeds each.
- Ran lexical routing, validation-calibrated abstention, three-domain sequential/rehearsal/growth/capacity experiments and integrated external memory.
- Saved all raw per-seed data, model states and comparison chart; 173/173 controls passed.
- Temporal and topic/expert checkpoints checked for exact reload; disk checkpoint round trip checked.
- Runtime 16.2 s. Result note [[13 - Extended Prototype Results]].

## Exploratory stress and replication
- First tested seed-42 at longer delays/noise, saved [[14 - Temporal Stress Results]].
- Registered and executed the complete five-seed stress grid using original checkpoint settings.
- Reproduced 40/40 primary temporal scores; 320 stress outputs finite.
- Saved every condition and variability in [[15 - Temporal Stress Replication]].

## Final analysis and plot review
- Created inference-only context/expert/memory demo and verified all 9 checks.
- Reviewed charts; stress SD bars crossed plot bounds. Rendered full observed seed ranges instead and labelled them accurately; numeric results unchanged.
- The first summary-save command lacked its vault-path variable, so summary writing failed. Corrected the path and saved [[16 - Latest Prototype Findings and Demo]].
- No experiments were rerun for that file-save correction.

- Final integrity verification: all new vault links resolve, source/model copies match SHA256 hashes, provenance manifest saved.

## 06 Oct: next-round preparation
- Inspected current RAM (~2.4 GiB free) and existing sklearn/scipy libraries; no installation needed.
- Downloaded official CLINC150 data_full.json, domains.json, README, CC BY 3.0 license and pinned revision/checksum manifest to D:.
- Registered matched-count variable-delay models and public-query routing/calibration/transfer/probe methods in [[17 - Variable Delay and Real Query Protocol]].


## Public-query routing run
- Audited duplicate texts and preserved filtered official splits; fit vocabulary on train only.
- Evaluated three known domains and held-out supported intents across five training subsets.
- Calibration uses validation only; rejection, coverage and negation failures recorded.
- 37/37 controls passed. Saved router and [[19 - Real Query Routing Results]].
- Runtime 10.2 s.

## Variable-delay CPU run
- Ran all 60 model/task/seed fits, equal 120-update budgets and approximate parameter matching.
- Saved all 480 held-out condition results, validation histories and five checkpoints.
- 811/811 controls passed; results [[18 - Variable Delay Memory Results]].
- Runtime 469.7 s.

## Saved-artifact verification and archive, 2026-10-06
- Independently loaded five saved checkpoints and reproduced all 480 recorded held-out memory accuracies exactly. Verified empty-query rejection.
- Archived scripts, results, checkpoints, router and original CLINC data with attribution/license. Verified archive SHA256 hashes; manifest in SpikingBrain Assets.
- Added runnable query demo and [[20 - Latest Memory and Routing Findings]], including failures and next proposals.
- All new artifacts/cache/temp on D:; no new full language-model downloads.

## Follow-up preparation, 2026-10-06
- Confirmed installed nomic embedding model and unloaded GPU before tests; checked official Ollama embedding API and Nomic classification prefix instructions.
- Registered [[21 - Embedding and Longer Order Protocol]] before fitting. Reuse local models/libraries; new artifacts on D:.


## Longer event-order run
- Completed20 fits across five seeds, longer train lengths16/32/64, shared budgets and validation selection.
- Compared spike-only final readout with hybrid membrane readout and RNN/GRU controls.
- 205/205 controls;120 saved scores reproduced exactly. [[23 - Longer Order Memory Results]].

## Local embedding routing run
- Encoded 13103 independent texts with frozen installed Nomic model; cached on D:.
- Evaluated all ten seed/transfer fits with validation-only calibration. 27/27 controls passed.
- Saved classifier and [[22 - Local Embedding Routing Results]], including manual failures.

## Follow-up verification and archive
- Reproduced all8 routing probe probabilities/decisions from saved classifier and cached features. All120 memory test scores reproduced after disk reload.
- Archived new scripts, results, local classifier, five memory states and full feature cache; SHA256 verified.
- Recorded comparative findings and limitations in [[24 - Embedding and Order Findings]].

## Final follow-up checks
- Reviewed both comparison plots visually; verified all links in notes21–24 resolve.
- Ran saved embedding-router demo against local Ollama: account-balance query mapped to banking, confidence0.9990, threshold0.99. No retraining.
- Recorded standard seed43 reject-all failure and accepted-accuracy exclusion explicitly in notes22/24.
- 27 routing and205 memory implementation checks passed;120 memory scores and8 routing probe predictions reproduced from saved artifacts.


## State tracing follow-up preparation
- Registered [[25 - State Tracing and Language Challenge Protocol]] before fitting new ablations or evaluating the independent authored language challenge.
- Reuse saved checkpoints and frozen local routers; all new artifacts on D:.


## Independent language challenge
- Froze 100 labelled task families/300 authored texts before evaluation; checked unique texts and no exact dataset/prior-probe overlap.
- Evaluated frozen lexical/Nomic routers with saved thresholds; recorded every failure and coverage. [[27 - Independent Language Challenge Results]].

## State tracing and neuron comparisons
- Traced three frozen models across five seeds; fitted separate diagnostic probes on independent train labels, with before/after-second-signal caveat.
- Completed25 fresh five-seed fits comparing readout, decay and reset against RNN.
- 245/245 controls;150 saved scores reproduced exactly. [[26 - State Tracing and Neuron Ablation Results]].

## Trace/challenge verification and archive
- Independently reproduced all300 language decisions for both saved routers using archived frozen features.
- Verified archive hashes for scripts, states, challenge data/results/features and reports; manifest retained.
- Consolidated mechanisms, neuron comparison, language failures and limitations in [[28 - Memory Trace and Language Challenge Findings]].

## Final visual and note checks
- Reviewed comparison charts; redrew state trace with explicit sampled stages to separate adjacent step39/41 labels.
- Verified every link in notes25–28 and all archived hashes.
- Retained existing membrane readout as spiking baseline because decay/reset alternatives did not improve consistently across five runs. RNN remains stronger.
- Completed25 ablation fits, frozen-state diagnostics and300-text authored language challenge; recorded limitations and failures.


## Root-cause fix preparation
- Registered [[29 - Root Cause and Fix Protocol]] before fitting fixes or evaluating new wrappers.
- Memory fix adds an explicit protected event slot; language fix adds unknown-class supervision, bounded request-scope extraction and empirical validation calibration.
- These are bounded engineering interventions, with model/feature budget differences and generalization limits recorded. No new downloads.


## Routing root-cause fixes
- Trained explicit unknown class and word/character features on official training splits only, five seeds plus held-intent transfer.
- Used validation-only empirical calibration and frozen bounded clause extraction.
- Evaluated identical new wrapper families with old baselines and fixed classifier with/without extraction; preserved already-seen challenge as regression. [[31 - Unknown Class and Request Scope Fix Results]].

## Root-cause fix completion
- Implemented first-event protected slot, trained/calibrated on independent inputs, then added two-event completeness checking and fresh confirmation after stress failures.
- Trained explicit unknown-class router across five seeds, added bounded request extraction, corrected punctuation/answer-marker scope failures and confirmed on disjoint base families.
- Preserved weak-signal/unfamiliar-intent failures; no general SNN or unrestricted language claims.
- Reproduced45 memory scores,25memory condition arrays and420routing probabilities/decisions; SHA256-verified archive.
- Added runnable bounded prototype and [[32 - Root Cause Fix Findings and Runnable Prototype]].

## Final root-fix checks
- Ran saved combined prototype: negated banking distractor/paid-leave request routed to work; first event retained as positive.
- Added pinned scope-v1 reproduction entrypoint, checked script syntax and archive hashes; reviewed comparison plot.
- All new experiment artifacts remain on D:. Records include weak/missing-signal and unfamiliar-intent failures.


## Complete benchmark inventory
- Compiled every recorded aggregate condition table, latest confirmation metrics and verification counts into [[33 - Complete Benchmark Results]].
- Distinguished different training/noise grids, accepted accuracy vs coverage, authored dependent templates, specialized protected slot and unrun full-model benchmarks. No experiments rerun.

## Weak-benchmark follow-up preparation
- Registered [[34 - Weak Benchmark Improvement Protocol]] before fitting new memory/routing variants.
- New memory tests add noise at the actual pulses and independent second signs, fixing two simplifying assumptions in earlier synthetic generators.
- Routing rescue uses explicit intent descriptions as extra taxonomy knowledge; held-out query examples remain excluded from fitting/calibration.


## Weak-benchmark improvement completed
- Registered34before fits; learned gated hybrid memory, five seeds,15fits,360condition rows and30incomplete-input diagnostics. All270 learned scores reproduced;45structural checks passed. [[35 - Learned Weak Signal Memory Results]].
- Recreated allfivelexicaltransfer baselines; category-description rescue improved named held-intent coverage with explicit unknown-acceptance tradeoff. Fixed nearest-descriptor rule discrepancy; initial verification variable collision corrected; all22090strictdecisions reproduced. [[36 - Category Description Routing Results]].
- All new scripts, descriptors, model states and reports onD;SHA-verifiedcopies in vaultassets. Existing prototypes retained.

## E-series repair scope registered
User expanded work toE2/E3/E5/E6,native fine-tuning,EWC,no-healing,directedits,sleep,intrinsiccorrection. Audited lab-code and prior root-cause runs. Registered new five-seed protocol37before newfits; retained all oldresults and no-healingnegativecontrol.

## E-series repair runs completed
- CoveredallnamedE1/EWC/nativenaive,E2/nohealing/edits/memory/sleep/intrinsic,E3loops/denoising/gates,E5genome/compression,E6integration. Five-seedcomparisonswithregisteredprotocols37–39.
- OriginalEWCgradientestimatorfixedbutEWCalonefailed;replayregularizationcalibratedindependently. Sleepcorrectedstale-teacheranddouble-corruptionerrors. SVDinitialization/hybridablationtested.
- 310predictionarraysreproduced,15rawimagesplitchecks,5haltingand5exactmemorychecks. Firsttimingimportfailureandexactmemorynumericbugfixed;oldrecordsretained.
- [[40 - E Series Repair Benchmark Results]];91newartifactsSHA-verifiedinvaultassets;allD:.

## Repair results finalized and demonstrated
- Five runnable modes executed: sleep, perception, compression, integration and EWC. Demo examples explicitly selected for display, never used to compute benchmark scores.
- Serialized post-training latency verified: robust perception~0.152ms/batch360, gated~0.752;8loop~1.702msvsactivegate~2.753. Fewer steps did not save runtime here.
- Polished complete report40and findings/demo41; comparison plot saved in vaultassets. Summary aggregation of structured timing corrected; final scripts parsed and archive hashes rechecked. All D:.

## Stronger-results round registered
User judged earlier gains too weak. Registered42before fits: matched spatialrepresentation, inference correction from training memory, explicit independent-observation effort, balanced continualreplay, stable weight-generating genes/int8storage. Old models and failed claims retained. All D:.


## EWC root-cause and weak-benchmark repairs — 2026-10-06

Registered protocols48–53 before each relevant experiment. Completed nine experiment groups, including failed large-penalty sweeps, incomplete bias correction and unsuccessful RBF alternative. Recorded task-wise loss/gradient/Fisher diagnostics; confirmed single-input training and confidence-controlled EWC on five fresh splits. Saved248reproduced prediction arrays and69checksum-verified artifacts. See notes54 and55. All writes and caches remained on D:. No full-language-model result is claimed.

Report-generation correction: a Windows default-encoding write made the report script unreadable to Python. Re-saved it as UTF-8, reran successfully, and visually inspected the final comparison chart. All three runnable repair modes were exercised; the default saved-example demos include incorrect predictions and were not selected to appear successful.



## Frozen clean-retention confirmation — 2026-10-06

Downloaded small public UCI archive to D:, checked prior-image overlap (zero), froze partition and protocol56. Trained20sequential models with identical1200update budgets, including five ANN/LIF pairs after validation eligibility. First final test evaluation followed all fitting. Passed140recorded checks plus60deployment-array checks. Separated deployment storage from test-containing artifacts and recalculated combined-method first-task retention from actual predictions. Verified71archive checksums and visually inspected result chart. Notes57–58 report successes, memory-caused harms, missed noisy target and spiking parity. All files/temp/caches on D:.


## Larger ten-task benchmark — 2026-10-06

Registered notes59and60before official-test evaluation. Downloaded eight public benchmark archives to D:, validated dimensions/counts and recorded SHA256sources. Selected training/validation30000/6000, tested all20000official examples after choices frozen. Exact selected split overlaps were zero. Completed eleven neural fits including three LIF pilot settings and frozen selected-LIF confirmation; measured firing/gradient/forgetting. Original raw references proved unnecessarily float32: added lossless packing proof, same-content control and greater inference memory below the same1.5MiB cap. Decoded runtime RAM differs from storage. Passed238original and126packed checks; archived74checksum-verified files. Stopped only the two verified owned evaluation workers after all29deployments were frozen; resumed identical calculations from checkpoints using shared memory scores. Verified full neural predictions and balanced100-example cached/deployed agreement. Standardized cached CPU timings to two BLAS/Torch threads. Notes61-62report all successes and failures. Existing Python only, no environment installations, all writes/cache/temp on D:.


## Larger ten-task benchmark — 2026-10-06

Registered notes59and60before official-test evaluation. Downloaded eight public benchmark archives to D:, validated dimensions/counts and recorded SHA256sources. Selected training/validation30000/6000, tested all20000official examples after choices frozen. Exact selected split overlaps were zero. Completed eleven neural fits including three LIF pilot settings and frozen selected-LIF confirmation; measured firing/gradient/forgetting. Original raw references proved unnecessarily float32: added lossless packing proof, same-content control and greater inference memory below the same1.5MiB cap. Decoded runtime RAM differs from storage. Passed238original and126packed checks; archived75checksum-verified files. Stopped only the two verified owned evaluation workers after all29deployments were frozen; resumed identical calculations from checkpoints using shared memory scores. Verified full neural predictions and balanced100-example cached/deployed agreement. Standardized cached CPU timings to two BLAS/Torch threads. Notes61-62report all successes and failures. Existing Python only, no environment installations, all writes/cache/temp on D:.

Final inspection: CNN and memory-only demos recognized the unchanged default first test digit; spiking demo misclassified the first clothing example. No examples were cherry-picked. Existing saved predictions revealed newest-pair accuracy ANN88.75%, CNN92.30%, LIF9.78%; report updated to distinguish non-acquisition from forgetting. All18cached full-test probability files passed normalization/finite/exact reload checks. Added the original corruption helper to the verified archive. Chart visually inspected; runnable models and dependencies remain on D:.


## 2026-10-06 — Training root-cause repair
Registered note63. Tested larger actual training replay, half-current batches and acquisition-aware checkpoint selection with matched update counts. Selected method on validation; completed two fresh confirmation initializations. Evaluated unused training-source holdout, audited exact image overlap and reported a stricter filtered result. Verified finite parameters, checkpoint byte caps and exact reload predictions. Notes64–65 contain results and limits. All files and processing stayed on D:.

Final audit: no exact-image overlap in the10,000-image holdout. All checkpoint reload predictions identical. CNN and ANN runnable demos correctly classified index0; LIF demo at index9000 ran but misclassified clothing class18 as19. Figure visually checked. Chosen-method confirmation-only clean means: ANN76.145%, CNN80.175%, LIF71.995%; three-run means include the pilot. Archived reports, code, models and checksum manifest in SpikingBrain Assets/training-repair.


## 2026-10-06 — Append-only growth and integrity self-healing
Registered note66. Trained six growing libraries (60 independent task experts), with no updates to old modules. Verified270 old-module parameter/prediction checks. Tested explicit-context and automatic routing on fresh6000-image holdout and previously inspected10,000-image confirmation holdout. Audited zero exact-image overlap. Tested frontend node addition, persistent reload, isolated weight/knowledge corruption recovery and invalid-backup refusal. Semantic checksum limitation explicitly tested. Persistent deployments and notes67–68 contain actual scores, routing changes, storage growth and remaining weaknesses. All writes on D:.


## 2026-10-06 — Router root cause follow-up
First automatic growth result was only64%, despite99% oracle-context experts. Registered note69; compared six class-cluster router configurations using validation only. Froze unit normalization/32 centers per class, evaluated new4000-image holdout and audited zero overlap. Appended versioned routing nodes without changing old experts or old router entries; verified both old and new policy outputs. Tested integrated category-node healing and refusal after both copies damaged. Notes70–71 record actual improvements, storage costs and remaining automatic-routing changes. Every file on D:.

Additional fault test: a valid-format but corrupted category mapping demonstrably changed predictions; automatic integrity recovery restored exact healthy predictions. Full details in growth-meaningful-healing.json and note70.

Completion audit: all60 source expert files match their training hashes; all126 deployed expert/router nodes and their snapshots match trusted hashes. Both automatic cluster models and explicit-category LIF demos ran successfully. Final comparison plot visually checked. Exact recovery probe scores98.75%→50%→98.75% recorded; preliminary100% wording in chat was corrected. Archived current code, reports, scores, frozen holds and routing nodes with verified SHA256 manifest. No writes on C:.


## 2026-10-06 — Frontend growth and feedback correction
Registered note72. Installed React19.3.0/ReactDOM19.3.0/jsdom30.1.2 locally with D: cache/temp. Reviewed official React/MDN lesson sources. Created fixed-feature routing, novelty/clarification, six HTML/CSS lessons then four React lessons, immutable version history and verified feedback correction. Initial missing helper import stopped the harness before evaluation; preserved failure and reran in isolated session. Development-only selection and candidate verification preceded held-out evaluation. Six old nodes and24 old-query decisions preserved;36 old execution tests passed. Final18/20 retrieval,60/60 direct execution,54/60 routed execution;unknown10/10;ambiguity3/4. Injected counter0/6→6/6. Independent list keys6/6. Eleven live/snapshot hashes verified. Notes73–74 disclose failures, template/lexical limits and external-feedback assumptions. All new files on D:.

Completion check: all60 previous spiking/CNN expert source files unchanged;11 frontend active nodes/snapshots valid. Actual CLI latest counter, unsupported effect request and pinned injected original verified. Archive checksums include trusted per-store manifests; all143 archived files verified. Records retain held-out clarification/ambiguity failures.


## 2026-10-06 — College page comparison
Checked existing Qwen installation; froze shared local HTML brief; recorded baseline clarification; authored a clearly labeled lesson in a separate immutable branch; preserved all 11 original nodes; ran Qwen with 5000-token cap and recorded truncation; ran one continuation; tested both versions and taught template in real Chrome; corrected shared test handling of whitespace/validation/delays; reviewed screenshots; saved comparison preview and detailed notes 75–77. Qwen completed 18/22, taught template 22/22. No generative claim for prototype. Everything saved on D:. See [[76 - College Landing Page Comparison Results]].


## 2026-10-06 — Hospital component experiment
Extracted eight generic blocks from existing college lesson, supplied hospital text as data, built deterministic composer in a separate D: directory, stored eight new immutable generic nodes while preserving 12 records, ran real-browser 22/22 checks, visually corrected stale text/layout, audited alias routing and integrity, observed raw router incorrectly selects college lesson. Clearly recorded assistant assistance and no autonomous generation. See [[78 - Hospital Component Composition Results]].


## 2026-10-06 — Automatic component assembly
Wrote bounded rule planner and reusable renderer; froze source hash before test fixtures; assembled library, bakery, clinic, college and no-form library with variable section order and card counts; 10/10 planning cases, 92/92 browser checks. Verified 12 original records and all 26 branch records/live backups. Kept model router unchanged and explicitly labeled rules as assistant-authored, not learned neural planning. Recorded design limitations and input grammar. See [[79 - Automatic Component Assembly Results]].



## 2026-10-06 — Learned language domain selector
Trained and froze statistical v1 classifier on 50 examples; core 20/20, harder stress 5/8; preserved failures. Trained separate v2 on 59 examples including development counterexamples; fresh v1/v2 comparison 6/10 to 10/10; v2 core replay 20/20. Browser checks v1 228/228 and v2 76/76. Verified 12 original nodes, all 26 branch live/backups and both frozen models. Slots and rendering remain rules, no spiking or LLM learning claimed. Saved D: artifacts and notes 80–81.


## 2026-10-06 — React/CSS/JavaScript builds and training
Trained technology selector on 40 then 48 examples, retained v1 and failed Vue request, fresh v1/v2 6/8 to 8/8. Compiled real JSX via isolated D: esbuild using existing React. React apps 24/24 each; vanilla initially 20/23, fixed CSS/hidden conflict in DOM wiring then 23/23. JS 10/10, CSS 9/9. Appended four verified source skills, preserved 26 earlier records, audited 30 current records/live backups. Sources assistant-authored; selection learned, no generative model fine-tune. All artifacts and caches on D:. See notes 82–83.


## Parameter count audit
Read saved v2 domain and technology classifier coefficient/intercept arrays; counted TF-IDF statistics separately; no model training or modification. Saved [[84 - Frontend Parameter Count]].


## CPU coding training pilot — underway
Audited hardware and runtimes; downloaded pinned public Qwen coder 0.49B to D:; prepared requested language/agent/tool dataset; fixed tokenizer API setup; trained real CPU LoRA V1 (270,336 coefficients); preserved tool regression and strict vs semantic tests; started separate V2 with natural tool requests. Installed D: native OpenCode because C: launcher invalid; provider registration works. Downloaded CPU-only llama runtime and matching converter. See [[85 - CPU Coding Model Training Protocol]]. Final results pending actual tests.
