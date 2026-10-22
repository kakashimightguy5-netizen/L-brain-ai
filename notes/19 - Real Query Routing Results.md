---
tags: [experiment, clinc150, routing, abstention]
created: 2026-10-06
---
# Real-query routing results

Back to [[17 - Variable Delay and Real Query Protocol]] · [[08 - SpikingBrain Process Log]]

## Dataset and method
CLINC150 crowdsourced English queries from Larson et al. (EMNLP 2019), official clinc/oos-eval pinned revision, CC BY 3.0. This is a public natural-language dataset, not actual user bank data or live customer traffic. Supported domains banking/travel/work; other domains and original out-of-scope queries are rejection tests.

Official train/val/test splits retained after lowercase/whitespace duplicate audit. Original counts {'oos_val': 100, 'val': 3000, 'train': 15000, 'oos_test': 1000, 'test': 4500, 'oos_train': 100}; exclusions {'train': 0, 'oos_train': 0, 'val': 3, 'oos_val': 0, 'test': 2, 'oos_test': 0}. Filtered counts {'train': 15000, 'oos_train': 100, 'val': 2997, 'oos_val': 100, 'test': 4498, 'oos_test': 1000}. Each seed samples 80 train examples per supported intent. Five training subsets share the official test set, so SD reflects training-subset sensitivity, not independent test samples.

TF-IDF word uni/bigrams fitted on selected train only, min_df 2, max_features 6000. C=.1,1,10 selected by known validation accuracy. Rejection threshold uses validation supported/unsupported examples, maximizes coverage with >=95% accepted accuracy and <=10% unsupported acceptance. Empty feature vectors reject. Test data never select C/threshold. Controls 37/37 passing.

![[SpikingBrain Assets/real-query-routing.png]]

## Results
Read accepted accuracy together with coverage. Correct-answer fraction counts rejected supported queries as not answered correctly. Unsupported rejection includes semantically neighbouring categories (e.g. credit-card intents vs banking), so this is deliberately harder than generated topic-word routing.

| Evaluation | Metric | Mean +/- sample SD |
|---|---|---|
| standard | accuracy_without_rejection | 97.96 +/- 0.12% |
| standard | coverage | 84.37 +/- 1.07% |
| standard | accepted_accuracy | 99.98 +/- 0.04% |
| standard | unsupported_rejection | 90.95 +/- 0.29% |
| standard | correct_answer_fraction | 84.36 +/- 1.10% |
| heldout_intents | accuracy_without_rejection | 59.41 +/- 1.58% |
| heldout_intents | coverage | 16.15 +/- 1.47% |
| heldout_intents | accepted_accuracy | 89.01 +/- 4.15% |
| heldout_intents | unsupported_rejection | 90.54 +/- 0.52% |
| heldout_intents | correct_answer_fraction | 14.37 +/- 1.40% |

Standard seed-42 counts: 3600 train, 900 validation, 1350 supported test, 4148 unsupported test. Feature count 4038; classifier coefficients/intercepts 12117. TF-IDF vocabulary/storage adds overhead beyond coefficients.

## Held-out intent transfer
Held out lexicographically last 3 intents/domain before fitting: {'banking': ['spending_history', 'transactions', 'transfer'], 'travel': ['travel_notification', 'travel_suggestion', 'vaccines'], 'work': ['schedule_meeting', 'taxes', 'w2']}. Train on remaining 12 intents/domain; validation excludes held-out intents too. Evaluate those new intents as belonging to the original supported domains. Vocabulary and hyperparameters never fit held-out examples. This tests within-known-domain transfer, not new-domain discovery.

## Manual negation diagnostics (seed 42)
These probes are authored diagnostics, not CLINC samples or tuning data. Unknown means desired abstention. All seeds' probe outputs, including failures, are in JSON.

| Text | Expected | Decision | Outcome |
|---|---|---|---|
| I am not asking about my bank balance. I want to know my paid time off balance. | work | unknown | incorrect |
| Don't book a flight; check when my salary will be paid. | work | unknown | incorrect |
| This is not about my salary; I need the baggage allowance for my flight. | travel | unknown | incorrect |
| Not a flight question: what is the balance of my checking account? | banking | banking | correct |
| Please ignore my bank balance. How do I cook an omelette? | unknown | banking | incorrect |
| I don't want to schedule a meeting; tell me the weather. | unknown | work | incorrect |
| I don't need flight information. Tell me a joke. | unknown | travel | incorrect |
| I don't need a recipe, I need to check my account balance. | banking | banking | correct |

## Limits
A word/bigram model may react to a negated topic mention rather than the intended request. High confidence can be wrong; rejection thresholds can reduce coverage heavily. This is a three-domain classification subset, not a full 150-intent or generative benchmark. Filtering duplicates means results should not be compared directly with published full-dataset scores. No claim of semantic comprehension or production readiness.

## Records and reproducibility
- Source https://github.com/clinc/oos-eval and paper https://aclanthology.org/D19-1131/
- Original data/license/provenance D:\AI\SpikingBrain-study\experiments\datasets\clinc150
- [[SpikingBrain Assets/real-query-routing-results.json]] includes all C/threshold selection, per-category rejection, error samples and probes.
- D:\AI\SpikingBrain-study\experiments\run_real_query_routing.py
- Saved trusted local router: D:\AI\SpikingBrain-study\experiments\clinc-domain-router.joblib
- Rerun `py -3.12 -B 'D:\AI\SpikingBrain-study\experiments\run_real_query_routing.py'`
- sklearn 1.9.0; runtime 10.2 s; script and dataset hashes in JSON. All new artifacts/cache/temp on D:.
