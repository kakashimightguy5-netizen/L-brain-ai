---
tags: [L, fix, counting, brackets]
created: 2026-10-10
status: code done + equivalence-tested; NOT yet trained (no quality claim)
---
# 116 - Integer counter channel (bracket depth) for L
Weak spot (note 114): saturating recurrent units cannot count nesting. Fix: flag `BRACKET=1` adds an embedding of bracket depth (( [ { +1, ) ] } -1, clamped 0..16), zero-initialised so it starts silent.
- Files: fastmem.bracket_depth, l4_model forward, l5_model (bracket flag), l6_step (raw counter in Stepper, GraphStepper ints 10 rows), train_L1 env BRACKET.
- MEASURED (tests/test_bracket.py, float64): depth fn == reference; flag-on zero-init == flag-off (diff 0.0); Stepper == full forward with channel active (3.6e-15). Old step-equivalence test still PASS.
- Not done: training comparison (L-Edge, 3 seeds, BRACKET=0 vs 1) and a bracket-closing accuracy probe. Strings/comments ignored by design.
- Hardware: a 5-bit saturating up/down counter per stream; negligible cost.
