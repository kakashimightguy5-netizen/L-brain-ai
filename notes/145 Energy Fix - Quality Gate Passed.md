# 145 - Energy fix: quality gate PASSED (2026-10-22 IST) - MEASURED, 1 seed

Continues note 144. Test: quality_ab.py, fine-tune 600 steps from LEdge_s0, identical batches, bpb on 6x1024 eval windows.

| variant | bpb before | bpb after | after vs eager | ms/step |
|---|---|---|---|---|
| eager fp32 | 0.93585 | 0.94095 | 0 | 378.5 |
| compiled + STE fp32 | 0.93585 | 0.93530 | -0.6% (better) | 157.9 (2.4x faster) |
| compiled + STE bf16 | 0.93579 | 0.94185 | +0.096% | 112.7 (3.4x faster) |

Gate (<= 0.5% worse than eager): passed by both. 

Caveats (honest): one seed, 600 steps. The -0.6% for fp32 is probably seed/ordering noise: read as "no quality loss", not "gain". Energy numbers from note 144 were taken with other GPU jobs running; clean idle-GPU re-measure still owed.

Combined with note 144: ~2.4-3.4x faster steps, compile_ab energy 4.22 -> 1.83 mJ/token (bf16 compiled) vs the 4.2 eager figure.

Next: (1) wire SPIKE_STE + compile + bf16 into training scripts as default-off flags; (2) 3-seed quality repeat; (3) idle-GPU energy re-measure; (4) give the transformer the same optimisations for a fair gap; (5) then resume the research roadmap.
