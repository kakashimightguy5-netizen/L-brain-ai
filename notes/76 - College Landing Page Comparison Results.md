# College landing-page comparison — 2026-10-06

Built and rendered a fictional Northbridge College landing page in self-contained HTML/CSS/JavaScript. All experiment files, browser profiles, screenshots and records are on D:. Local preview: http://127.0.0.1:8765/ . No public deployment or real submissions.

| Condition | Result | What it demonstrates |
|---|---|---|
| Original prototype | Clarify; no complete page | Its ten taught topics cannot compose a complete page |
| Qwen first attempt | 11/22 browser checks; incomplete | Reached 5,000-token output cap |
| Qwen with one continuation | 18/22 | Independent code generation with remaining defects |
| Separate taught prototype branch | 22/22 | Reliable replay of a human-authored complete-page template |

## Comparison limits
The template was written by the assistant, taught as one new immutable node and indexed against the exact brief. It is not an independently generated response, neural fine-tuning or unseen-task success. Timing excludes authoring and therefore cannot fairly be compared with Qwen inference. No SpikingBrain 5B/7B checkpoint was used. These are requirements checks on one page, not a general intelligence or certified accessibility benchmark. Visual design judgment is subjective: Qwen gives a conventional centered institutional layout; the supplied template uses editorial typography, a split hero and illustrated campus.

## Qwen configuration and results
Installed qwen3.5:9b, 9.7B parameters, Q4_K_M; context 8192, temperature 0.2, seed 1902, think false. Initial call: 569.8 seconds, 5000 generated tokens, stopped for length. Continuation: 246.6 seconds, 1916 generated tokens. Total calls: 816.4 seconds. Output preserved without manual code repairs; HTML wrapper extraction only. Runtime snapshot in assets records automatic partial GPU residency, approximately 3.35 GiB at observation. Existing model executed locally; no model download.

Four remaining failed checks: semantic regions (missing main), all internal anchors valid (placeholder links), explicit local-only form confirmation (promises receipt/contact despite no transmission), runtime errors (mobile menu tries to style absent .hamburger-line elements). Menu aria-expanded changes successfully despite its script error. Invalid input is blocked with error messages; both pages fit 390px and 1440px and make no external requests in the browser test.

## Prototype branch and preservation
Original route score 0.19497, action clarify. Added html_college_page_v1 in a separate taught-library. Existing html.label skill supplies the labeled fields. All 11 original knowledge records verified byte-identical; originals untouched. Exact-brief retrieval score approximately 1.0 is expected because the prompt itself is an alias, not generalization evidence.

## Verification and process record
Headless real Chrome via Playwright, local HTTP server, desktop 1440x1000 and mobile 390x844. Screenshots reviewed. Checks cover title, viewport, h1 count, semantic regions, required sections/programs, anchor targets, SVG/favicon, required labeled inputs, email type, overflow, desktop navigation, menu state, live status, invalid/valid form behavior, keyboard focus, runtime errors and external requests. See browser-results.json for each result. Check corrections applied equally: normalize program-name whitespace, treat visible validation errors as blocked inputs, allow delayed form messages, and include keyboard focus. The form test validates the requested no-transmission message, not just a thank-you.

Initial teaching helper failed before adding data because a wildcard import replaced its BASE variable; explicit imports corrected it. Failed log retained. One permission review timed out on a diagnostic write; a narrower retry succeeded, no work remains blocked. Browser output cap failure preserved; one continuation separately recorded. No public hosting, paid APIs or new C: files requested by this experiment. Existing executables and Python/Node libraries read from C:.

## Next experiment
Evaluate an unseen hospital brief without teaching its complete answer. Introduce a component composer with verified HTML/CSS building blocks and a separate held-out test set. Keep learned records immutable, test old behavior after every new node, and treat repair as a versioned correction. This separates actual composition from exact-template replay. Qwen's four defects can be used as independently verified repair exercises without changing the preserved benchmark.

Artifacts: D:\AI\SpikingBrain-study\experiments\frontend-learning\landing-comparison
Vault evidence: SpikingBrain Assets/landing-comparison
