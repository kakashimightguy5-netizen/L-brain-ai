# Unknown-class and bounded request-scope fix

[[29 - Root Cause and Fix Protocol]] · [[27 - Independent Language Challenge Results]]

Four classes trained with explicit unsupported-domain/OOS examples from official training only. Word/character TF-IDF fit train only; C selected using balanced four-class validation accuracy. Empirical candidate thresholds avoid coarse-grid reject-all gap; validation accepted quality>=95%, unknown rejection>=90%. Extra unknown supervision and larger feature space differ from old classifiers; not a matched-compute comparison. Bounded request extractor removes explicit standalone topic denials only when positive clauses remain; intrinsic request negation is preserved. Unrecognized 'Forget X and answer...' forms deliberately remain untreated. No general semantic comprehension claim.

| Official evaluation | Metric | Mean +/- sample SD |
|---|---|---|
| standard | accuracy_without_rejection | 94.74 +/- 0.19% |
| standard | coverage | 95.13 +/- 0.19% |
| standard | accepted_accuracy | 99.60 +/- 0.07% |
| standard | correct_answer_fraction | 94.74 +/- 0.19% |
| standard | unsupported_rejection | 98.23 +/- 0.10% |
| heldout_intents | accuracy_without_rejection | 31.19 +/- 0.71% |
| heldout_intents | coverage | 37.78 +/- 0.94% |
| heldout_intents | accepted_accuracy | 82.57 +/- 1.98% |
| heldout_intents | correct_answer_fraction | 31.19 +/- 0.71% |
| heldout_intents | unsupported_rejection | 98.62 +/- 0.07% |

New wrappers over official held-out questions are dependent contrast families; underlying test queries were previously evaluated. Old300 authored prompts are already-seen regression only. Each record preserves original/selected text and failure. Compare fixed_no_scope vs fixed_scope to isolate parser contribution, old_lexical vs fixed_no_scope to see combined unknown-supervision/feature/calibration change.

| Dataset | Router | Style | Correct decision | Supported coverage | Unknown rejection |
|---|---|---|---:|---:|---:|
| fresh_wrappers | old_lexical | all | 58.57% | 51.67% | 75.83% |
| fresh_wrappers | old_lexical | denial_after | 53.57% | 41.67% | 83.33% |
| fresh_wrappers | old_lexical | do_not_discuss | 51.19% | 41.67% | 75.00% |
| fresh_wrappers | old_lexical | neutral | 90.48% | 90.00% | 91.67% |
| fresh_wrappers | old_lexical | rather_than | 48.81% | 41.67% | 66.67% |
| fresh_wrappers | old_lexical | unsupported_form | 48.81% | 43.33% | 62.50% |
| fresh_wrappers | fixed_no_scope | all | 83.10% | 96.33% | 77.50% |
| fresh_wrappers | fixed_no_scope | denial_after | 78.57% | 98.33% | 62.50% |
| fresh_wrappers | fixed_no_scope | do_not_discuss | 76.19% | 98.33% | 58.33% |
| fresh_wrappers | fixed_no_scope | neutral | 92.86% | 90.00% | 100.00% |
| fresh_wrappers | fixed_no_scope | rather_than | 84.52% | 98.33% | 79.17% |
| fresh_wrappers | fixed_no_scope | unsupported_form | 83.33% | 96.67% | 87.50% |
| fresh_wrappers | fixed_scope | all | 88.10% | 93.00% | 90.00% |
| fresh_wrappers | fixed_scope | denial_after | 78.57% | 98.33% | 62.50% |
| fresh_wrappers | fixed_scope | do_not_discuss | 92.86% | 90.00% | 100.00% |
| fresh_wrappers | fixed_scope | neutral | 92.86% | 90.00% | 100.00% |
| fresh_wrappers | fixed_scope | rather_than | 92.86% | 90.00% | 100.00% |
| fresh_wrappers | fixed_scope | unsupported_form | 83.33% | 96.67% | 87.50% |
| fresh_wrappers | old_embedding | all | 46.19% | 29.33% | 88.33% |
| fresh_wrappers | old_embedding | denial_after | 44.05% | 28.33% | 83.33% |
| fresh_wrappers | old_embedding | do_not_discuss | 46.43% | 30.00% | 87.50% |
| fresh_wrappers | old_embedding | neutral | 47.62% | 30.00% | 91.67% |
| fresh_wrappers | old_embedding | rather_than | 46.43% | 30.00% | 87.50% |
| fresh_wrappers | old_embedding | unsupported_form | 46.43% | 28.33% | 91.67% |
| seen_regression | old_lexical | all | 54.33% | 34.67% | 76.67% |
| seen_regression | old_lexical | negation | 27.00% | 12.00% | 50.00% |
| seen_regression | old_lexical | neutral | 69.00% | 50.00% | 88.00% |
| seen_regression | old_lexical | reframed | 67.00% | 42.00% | 92.00% |
| seen_regression | fixed_no_scope | all | 67.67% | 82.67% | 75.33% |
| seen_regression | fixed_no_scope | negation | 34.00% | 100.00% | 36.00% |
| seen_regression | fixed_no_scope | neutral | 85.00% | 76.00% | 94.00% |
| seen_regression | fixed_no_scope | reframed | 84.00% | 72.00% | 96.00% |
| seen_regression | fixed_scope | all | 82.33% | 69.33% | 95.33% |
| seen_regression | fixed_scope | negation | 78.00% | 60.00% | 96.00% |
| seen_regression | fixed_scope | neutral | 85.00% | 76.00% | 94.00% |
| seen_regression | fixed_scope | reframed | 84.00% | 72.00% | 96.00% |

Implementation checks 31/31. Full training choices, validation calibration and all decisions [[SpikingBrain Assets/fixed-routing-results.json]]. Saved runnable classifier D:\AI\SpikingBrain-study\experiments\fixed-domain-router.joblib and request_scope.py. All new scripts/cache/results on D:. Existing Nomic baseline reused locally; no downloads. Runtime 113.9s.

## Version2 scope correction and separate confirmation
Development wrappers exposed two parser failures: missing punctuation before a suffix denial and an explicit answer marker left untreated. Version2 supports these bounded forms, preserving intrinsic negation. It does not change classifier weights or thresholds. New confirmation uses last20 base queries/domain and last3/category, disjoint from development first20/first3 families; same underlying official test corpus remains historically seen. Confirmation distractors avoid the target supported domain, preventing contradictory topic framing. No confirmation labels tune version2.

[
  {
    "model": "old_lexical",
    "summary": [
      {
        "style": "all",
        "decision_accuracy": 0.44047619047619047,
        "coverage": 0.41,
        "accepted_accuracy": 0.8943089430894309,
        "unknown_rejection": 0.625
      },
      {
        "style": "denial_after",
        "decision_accuracy": 0.38095238095238093,
        "coverage": 0.25,
        "accepted_accuracy": 0.9333333333333333,
        "unknown_rejection": 0.75
      },
      {
        "style": "do_not_discuss",
        "decision_accuracy": 0.35714285714285715,
        "coverage": 0.31666666666666665,
        "accepted_accuracy": 0.7894736842105263,
        "unknown_rejection": 0.625
      },
      {
        "style": "explicit_answer",
        "decision_accuracy": 0.27380952380952384,
        "coverage": 0.31666666666666665,
        "accepted_accuracy": 0.7894736842105263,
        "unknown_rejection": 0.3333333333333333
      },
      {
        "style": "neutral",
        "decision_accuracy": 0.7857142857142857,
        "coverage": 0.7833333333333333,
        "accepted_accuracy": 1,
        "unknown_rejection": 0.7916666666666666
      },
      {
        "style": "rather_than",
        "decision_accuracy": 0.40476190476190477,
        "coverage": 0.38333333333333336,
        "accepted_accuracy": 0.8260869565217391,
        "unknown_rejection": 0.625
      }
    ]
  },
  {
    "model": "fixed_scope_v1",
    "summary": [
      {
        "style": "all",
        "decision_accuracy": 0.8428571428571429,
        "coverage": 0.9666666666666667,
        "accepted_accuracy": 0.8344827586206897,
        "unknown_rejection": 0.9333333333333333
      },
      {
        "style": "denial_after",
        "decision_accuracy": 0.6785714285714286,
        "coverage": 1,
        "accepted_accuracy": 0.65,
        "unknown_rejection": 0.75
      },
      {
        "style": "do_not_discuss",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      },
      {
        "style": "explicit_answer",
        "decision_accuracy": 0.6785714285714286,
        "coverage": 0.9833333333333333,
        "accepted_accuracy": 0.5932203389830508,
        "unknown_rejection": 0.9166666666666666
      },
      {
        "style": "neutral",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      },
      {
        "style": "rather_than",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      }
    ]
  },
  {
    "model": "fixed_scope_v2",
    "summary": [
      {
        "style": "all",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      },
      {
        "style": "denial_after",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      },
      {
        "style": "do_not_discuss",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      },
      {
        "style": "explicit_answer",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      },
      {
        "style": "neutral",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      },
      {
        "style": "rather_than",
        "decision_accuracy": 0.9523809523809523,
        "coverage": 0.95,
        "accepted_accuracy": 0.9824561403508771,
        "unknown_rejection": 1
      }
    ]
  }
]

Reproduction: run_routing_fix_v1.py pins the original scope-v1 module so subsequent scope-v2 edits do not change the recorded v1 experiment. Scope-v2 confirmation remains confirm_scope_fix.py with request_scope.py. The originally executed script source is retained in the archive.

