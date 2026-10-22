# Learned ordinary-language domain selector — 2026-10-06

Latest gallery: http://127.0.0.1:8768/ . Model-assisted page domain selection from natural-language cues, not general code generation.

| Evaluation | Result |
|---|---|
| Original bounded rules on same 20 core requests | 8/20 (all supported natural-language requests require clarification) |
| Learned v1, new core cases | 20/20 |
| Learned v1, harder nearby-domain cases | 5/8 |
| Fresh comparison, v1 | 6/10 |
| Fresh comparison, corrected v2 | 10/10 |
| V2 replay regression of core cases | 20/20 |
| V1 rendered-page checks, twelve pages | 228/228 |
| V2 fresh rendered-page checks, four pages | 76/76 |

Browser scores are functional assertions, not language understanding accuracy or design quality. No direct new Qwen benchmark. Both versions and all failures retained.

## What learned, what remains rules
A new TF-IDF word/character feature model and logistic regression classifier trained on 50 assistant-authored examples for college, hospital, library, bakery and other. This is statistical supervised learning, not a spiking-neuron planner or an LLM. It is a separate learned selector alongside the immutable knowledge store. It has not replaced or trained FrontBrain's raw retrieval route.

Brand/item extraction still uses rules: quoted brand plus include/highlight/showcase/offer/cover/featuring and a comma/and-separated list. Default section order is fixed; illustrations and layout are reused. It can infer a domain without explicit college/hospital/library/bakery words using cues such as degrees, patients, borrowing and pastries. It cannot freely reason about layout, follow arbitrary wording or generate new code/components. Confidence values are uncalibrated classifier outputs, not reliability guarantees.

## Root cause and correction
V1 confused veterinary practice with human hospital care, retail book sales with lending libraries, and online coding courses with degree-granting colleges. Shared vocabulary and insufficient nearby negative examples caused these errors. V1 development also predicted college for an automotive example, but low confidence/margin causes clarification.

V2 cloned the same estimator configuration and trained on 59 examples: original 50 plus three failed stress prompts and six nearby negative counterexamples. This is full retraining of a separate selector, not EWC or intrinsic self-correction. The original model file remains unchanged. The failed stress prompts became development/training data and are not claimed as held-out wins. V2 was frozen before ten newly written comparison prompts; it improved 6/10 to 10/10 while retaining the earlier 20-case replay results.

## Test limits
All training, development and tests were written by the same assistant. No independently authored, blinded test set exists here. New text is disjoint from training strings, but shares concepts and vocabulary; ten fresh cases do not establish generalization broadly. Unknown-domain labels reflect the declared four-domain scope. Negative medical/retail/course cases are unsupported under this scope. Explicit domain-conflict handling is a rule, not learned contradiction reasoning. Automated invalid-form checks test browser validity and confirmation; they do not certify complete accessibility/security.

## Preservation and execution
Twelve original knowledge records preserved; all 26 component branch live and backup entries verified against their manifests. Original renderer and both frozen selector files/source hashes verified. No old node, model or routing configuration overwritten. V1 and v2 serialized files, freeze manifests, training data, counterexamples, requests, probabilities and browser screenshots archived. Chrome profiles/cache and every new artifact on D:, existing C: runtimes read only. New gallery remains local; no real submissions or public hosting.

## Process
Trained v1 and inspected development output; froze model/source; ran new core cases; compared original bounded-rule baseline; rendered twelve pages; ran additional stress tests and preserved failures; trained v2 with counterexamples; replayed core regression; froze v2; compared both models on fresh prompts; rendered and checked four v2 pages; verified immutable records; saved this note and archive SHA manifest. predict.py exposes the v2 selector for future local requests.

## Next proposal
Collect independently written ordinary-language requests, including negation and mixed-domain scenarios. Measure acceptance coverage, wrong-domain rate and clarification rate separately. Add a learned section planner only after domain selection holds up. Current result improves domain selection in a bounded task; it does not demonstrate an independently learning designer.

![Latest library preview](SpikingBrain Assets/language-planner/latest-preview.png)
