# React, CSS and JavaScript learning/build experiment — 2026-10-06

Gallery: http://127.0.0.1:8769/ . Real React hospital and college apps; separate plain HTML/CSS/JavaScript clinic. No public deployment or real submissions.

## Results
| Check | Result |
|---|---|
| Initial technology selector, new requests | 19/20 |
| Corrected selector, replay of initial requests | 20/20 |
| Fresh comparison, initial selector | 6/8 |
| Fresh comparison, corrected selector | 8/8 |
| React hospital real-browser interactions | 24/24 |
| React college real-browser interactions | 24/24 |
| Plain HTML/CSS/JS interactions, after fix | 23/23 |
| Shared JS unit cases | 10/10 |
| CSS desktop/mobile grids and visible focus | 9/9 |

Scores describe small assistant-authored tests, not independent general code-generation ability or direct Qwen comparison.

## What was trained
A separate TF-IDF plus logistic-regression technology selector learns five labels: react_app, vanilla_app, css_skill, js_skill, unsupported. V1 trained on 40 examples. V2 trained on 48 examples with unsupported-framework counterexamples. Frozen v1 retained. V1 incorrectly selected React for a Vue request; that failed prompt became development/training data for v2, and fresh Vue/Angular/Svelte wording tested afterward. Both trained serialized models and training sets retained. Confidence values are uncalibrated.

Four supplied source skills added in a separate immutable library: React JSX page, responsive CSS stylesheet, shared JavaScript filtering/validation helpers, and native DOM page event wiring. Source is assistant-authored and verified; knowledge storage is not neural training on code syntax. This does not fine-tune SpikingBrain, Qwen, an SNN or a generative language model. It trains task selection and teaches reusable source artifacts. It cannot invent arbitrary new React components from a request.

## Real builds
React 19.3.0 and ReactDOM already installed on D:. Installed esbuild 0.28.2 in a new D: build-runtime, with npm cache/config/temp on D:. Bundled real App.jsx through esbuild into browser app.js and app.css. App.jsx uses useState and keyed components; source styles.css and interactions.js remain separate. Demo data supplied by assistant. Trained selector routes all three recorded build requests to the expected technology stack; the compiler verifies immutable source skills before copying/building them.

React features: controlled search, category filters, dynamic counts/empty state, theme toggle, mobile navigation, validated local demo form and reset. Vanilla version uses native DOM events and the same filtering/validation functions. Source links/files under react-learning/src; reproducible lab build via node build.cjs using the existing sibling runtime. This lab currently depends on D: sibling paths; not claimed as a standalone portable npm project.

## Root cause and fixes
Initial vanilla 20/23: source CSS display rules overrode the hidden attribute on cards, so search/filter state changed but cards stayed visible. Assistant corrected DOM wiring to explicitly set display:none for nonmatches. Initial failed browser report preserved. Final 23/23. This is a manual verified code correction, not autonomous self-healing. React apps were already 24/24. Vue confusion came from insufficient nearby unsupported-framework examples; corrected classifier trained separately and old model preserved.

## Verification and preservation
Real Chrome at 1440x1000 and 390x844. Tested search, empty results, reset, filters, theme state, menu, invalid/whitespace/valid forms, live status, React reset, anchors, runtime errors, external requests, overflow and separate stylesheet loading. Additional JS tests cover empty collections, combined category/query filters, trimmed/case-insensitive queries and no data mutation. CSS tests check computed multi-column/single-column grids and focus outlines. No external browser requests. Screenshots reviewed.

All 26 earlier branch records preserved; 30 current live and backup records verified. Old domain-selector experiments untouched. Stored source content hashes verified; compiler may normalize UTF-8/newlines when emitting source files, so model/data freezes and immutable source hashes are recorded separately. Four new source skills appended; no old skill overwritten.

## Limits and next proposal
Selector learns technology categories; code remains supplied reusable implementations and parameterized data. CSS-only and JS-only selection verified, plus actual shared stylesheet/function execution; no unrestricted CSS/JS synthesis demonstrated. Tests are by the same assistant and use familiar concepts. Unsupported frameworks remain outside scope. Next step: independently authored requests requiring new behavior, measure correctly selected skills and actual rendered behavior separately, then evaluate a genuine generative code learner if that is the intended research objective.

## Process and artifacts
D:\AI\SpikingBrain-study\experiments\frontend-learning\react-learning
Installed isolated bundler; authored source; trained/froze selector; stored three source skills; compiled JSX; evaluated requests; built/served three apps; tested/fixed vanilla visibility; trained/froze selector v2; fresh comparison; JS/CSS checks; added verified native DOM source skill; audited sources/models/old records; saved notes and archived evidence. All new source, caches, profiles, model files and evidence on D:. Existing C: executables/libraries read only.

![React preview](SpikingBrain Assets/react-learning/react-preview.png)
