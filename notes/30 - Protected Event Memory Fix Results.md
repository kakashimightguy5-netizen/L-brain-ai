# Protected first-event memory fix

[[29 - Root Cause and Fix Protocol]] · [[28 - Memory Trace and Language Challenge Findings]]

Engineered positive/negative threshold event neurons write the first spike sign into a protected two-channel synaptic slot. Later events do not overwrite it; next sequence resets memory. Logistic readout trained on fresh examples; threshold selected using independent clean/noisy validation only. No target labels or true pulse times at inference. This is a specialized architecture, not a parameter-matched end-to-end learned recurrent SNN. Original RNN and membrane states are unchanged controls. Five seeds, fresh512example tests/condition. Calibration thresholds [0.8, 0.8, 0.8, 0.65, 0.8].

| Model | Length | Noise | Mean +/- sample SD |
|---|---:|---:|---|
| old_membrane | 32 | 0.1 | 90.94 +/- 0.98% |
| old_membrane | 32 | 0.2 | 79.10 +/- 3.30% |
| old_membrane | 32 | 0.3 | 70.70 +/- 4.24% |
| old_membrane | 96 | 0.1 | 78.32 +/- 3.67% |
| old_membrane | 96 | 0.2 | 69.06 +/- 2.72% |
| old_membrane | 96 | 0.3 | 63.32 +/- 2.02% |
| old_membrane | 192 | 0.1 | 67.70 +/- 2.82% |
| old_membrane | 192 | 0.2 | 60.47 +/- 2.40% |
| old_membrane | 192 | 0.3 | 57.15 +/- 3.31% |
| RNN | 32 | 0.1 | 99.65 +/- 0.32% |
| RNN | 32 | 0.2 | 89.96 +/- 4.15% |
| RNN | 32 | 0.3 | 75.98 +/- 2.47% |
| RNN | 96 | 0.1 | 98.36 +/- 1.16% |
| RNN | 96 | 0.2 | 74.30 +/- 6.28% |
| RNN | 96 | 0.3 | 59.77 +/- 3.71% |
| RNN | 192 | 0.1 | 96.37 +/- 2.66% |
| RNN | 192 | 0.2 | 63.95 +/- 4.60% |
| RNN | 192 | 0.3 | 55.35 +/- 1.77% |
| protected_slot | 32 | 0.1 | 100.00 +/- 0.00% |
| protected_slot | 32 | 0.2 | 99.96 +/- 0.09% |
| protected_slot | 32 | 0.3 | 97.11 +/- 1.53% |
| protected_slot | 96 | 0.1 | 100.00 +/- 0.00% |
| protected_slot | 96 | 0.2 | 99.61 +/- 0.68% |
| protected_slot | 96 | 0.3 | 91.33 +/- 5.98% |
| protected_slot | 192 | 0.1 | 100.00 +/- 0.00% |
| protected_slot | 192 | 0.2 | 99.41 +/- 1.10% |
| protected_slot | 192 | 0.3 | 86.05 +/- 7.40% |

Stress results [{'seed': 42, 'mode': 'independent_second', 'coverage': 1.0, 'original_first_label_accuracy': 1.0, 'rejection': 0.0}, {'seed': 42, 'mode': 'weak_signal', 'coverage': 0.00390625, 'original_first_label_accuracy': 0.001953125, 'rejection': 0.99609375}, {'seed': 42, 'mode': 'no_first', 'coverage': 1.0, 'original_first_label_accuracy': None, 'rejection': 0.0}, {'seed': 42, 'mode': 'no_events', 'coverage': 0.00390625, 'original_first_label_accuracy': None, 'rejection': 0.99609375}, {'seed': 43, 'mode': 'independent_second', 'coverage': 1.0, 'original_first_label_accuracy': 1.0, 'rejection': 0.0}, {'seed': 43, 'mode': 'weak_signal', 'coverage': 0.0078125, 'original_first_label_accuracy': 0.0, 'rejection': 0.9921875}, {'seed': 43, 'mode': 'no_first', 'coverage': 1.0, 'original_first_label_accuracy': None, 'rejection': 0.0}, {'seed': 43, 'mode': 'no_events', 'coverage': 0.0078125, 'original_first_label_accuracy': None, 'rejection': 0.9921875}, {'seed': 44, 'mode': 'independent_second', 'coverage': 1.0, 'original_first_label_accuracy': 1.0, 'rejection': 0.0}, {'seed': 44, 'mode': 'weak_signal', 'coverage': 0.009765625, 'original_first_label_accuracy': 0.00390625, 'rejection': 0.990234375}, {'seed': 44, 'mode': 'no_first', 'coverage': 1.0, 'original_first_label_accuracy': None, 'rejection': 0.0}, {'seed': 44, 'mode': 'no_events', 'coverage': 0.009765625, 'original_first_label_accuracy': None, 'rejection': 0.990234375}, {'seed': 45, 'mode': 'independent_second', 'coverage': 1.0, 'original_first_label_accuracy': 0.994140625, 'rejection': 0.0}, {'seed': 45, 'mode': 'weak_signal', 'coverage': 0.080078125, 'original_first_label_accuracy': 0.0546875, 'rejection': 0.919921875}, {'seed': 45, 'mode': 'no_first', 'coverage': 1.0, 'original_first_label_accuracy': None, 'rejection': 0.0}, {'seed': 45, 'mode': 'no_events', 'coverage': 0.080078125, 'original_first_label_accuracy': None, 'rejection': 0.919921875}, {'seed': 46, 'mode': 'independent_second', 'coverage': 1.0, 'original_first_label_accuracy': 0.998046875, 'rejection': 0.0}, {'seed': 46, 'mode': 'weak_signal', 'coverage': 0.0078125, 'original_first_label_accuracy': 0.001953125, 'rejection': 0.9921875}, {'seed': 46, 'mode': 'no_first', 'coverage': 1.0, 'original_first_label_accuracy': None, 'rejection': 0.0}, {'seed': 46, 'mode': 'no_events', 'coverage': 0.0078125, 'original_first_label_accuracy': None, 'rejection': 0.9921875}]. Weak amplitude events below threshold are undetectable; a missing first event can cause a later event to be stored incorrectly. Noise spikes before the real event also contaminate write-once memory. No-events rejection is a separate measure, not label accuracy. Independent second signs test against exploiting the old opposite-sign shortcut. Full perseed data [[SpikingBrain Assets/protected-slot-results.json]]. Checks 15/15. All new artifacts on D:; no power or full SpikingBrain claims.

## Bounded version2 confirmation
Added exactly-two-event completeness check after version1 development stress failures. Event starts are detected from threshold crossing, not supplied pulse times. Weak/incomplete sequences abstain. Fresh confirmation inputs use different generation seeds; no threshold retuning. This is a specialized two-event contract and cannot be used unchanged for arbitrary pulse counts.

[
  {
    "mode": "normal",
    "coverage_mean": 0.973046875,
    "accuracy_mean": 0.973046875,
    "rejection_mean": 0.026953125
  },
  {
    "mode": "independent_second",
    "coverage_mean": 0.973046875,
    "accuracy_mean": 0.973046875,
    "rejection_mean": 0.026953125
  },
  {
    "mode": "weak_signal",
    "coverage_mean": 0.000390625,
    "accuracy_mean": 0.000390625,
    "rejection_mean": 0.999609375
  },
  {
    "mode": "missing_first",
    "coverage_mean": 0.0265625,
    "accuracy_mean": null,
    "rejection_mean": 0.9734375
  },
  {
    "mode": "no_events",
    "coverage_mean": 0.000390625,
    "accuracy_mean": null,
    "rejection_mean": 0.999609375
  }
]

All per-seed outcomes [[SpikingBrain Assets/protected-memory-v2-results.json]].
