# 135 - Competition, homeostasis and sleep-drift fixes (2026-10-14 IST, MEASURED, cloud CPU)
## Sleep reverse replay + stabilising anchor (note 134 follow-up), 3 seeds, 200 facts
| model | forward | reverse |
|---|---|---|
| L5 plain sleep | 98.7% | 90.8% |
| L5 sleep + anchor 5 (KD to pre-sleep self on random bytes) | 98.5% | 91.2% |
| T plain sleep | 93.8% | 73.8% |
| T sleep + anchor 5 | 98.8% | 90.8% |
| anchor 25 (seed 0): L5 92.5/54.5, T 99.0/81.0 -> too strong for L5 |
Reading: with the anchor both models reach ~91% reverse / ~99% forward. The reversal fix is a training-procedure win, not an L5-only win; L5's edge is that it does not need the anchor (fewer false memories: 54 vs 104).
Consistency filter (replay only memories that came back in the previous dream), seed 0, no anchor: L5 98.5/91.0 (no change), T 94.5/68.5 (worse). Dropped: false memories are themselves consistent.
## Competition (k-winner-take-all lateral inhibition) in the ternary spiking cell, delay-3 copy, H=128, seed 0
Baseline (no competition): 96.7% at ~28% firing.
- Hard k-WTA (losers get no gradient): k=32 73.4% (20% firing), k=16 68.7% (10%), k=8 60.4% (5%). k=32 with H=256: 68.7%.
- k-WTA with straight-through losers (silenced forward, still learn): k=32 93.1% (20%), k=16 85.7% (10%).
Cause of the first failure: losing neurons never received gradient (dead losers). Fix recovered most of it.
## Homeostasis (intrinsic plasticity: threshold rises when a neuron fires above target)
- target 10%, eta 0.05: 64.6% (H=128), 67.6% (H=256) -> too aggressive.
- target 20%, eta 0.01: 89.3% at 23.6% firing; target 25%, eta 0.005: 82.8% at 19.8%.
Reading: homeostasis alone is worse than straight-through k-WTA at the same firing rate. Best sparse option so far: k-WTA STE k=32 (93.1% at 20%, vs 96.7% at 28%).
## Laptop GPU (runner restarted 2026-10-14 ~21:20 IST)
Queued: full-precision fine-tune control (3 seeds) to judge the earlier ternary result cft2_tern_hid_b = 0.9065 bits/byte (vs baseline 0.9359, 1 seed; it had 6000 extra steps the baseline did not); ternary seeds 1-2. 24 redundant limitation jobs marked skipped (already run in cloud at 8000 steps).
