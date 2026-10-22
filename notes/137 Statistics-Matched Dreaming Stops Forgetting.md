# 137 - Statistics-matched dreaming stops catastrophic forgetting (2026-10-15 IST, MEASURED, cloud CPU)
Brain principle: sleep replay draws on the STATISTICS of recent experience, not exact recordings; cortex is trained to keep its old responses to such dream inputs (consolidation without storing episodes).
Method (cont_suite2.py, condition kduni, LAM=100): after learning task A, keep ONLY the 256-number byte-frequency histogram of task A (no episodes). While learning task B, the model also matches its frozen old self on 'dream' sequences sampled from that histogram (KL distillation, strength 100). L5 d=128, 4 layers, 936,357 params; 2500 steps A + 2500 steps B; fixed 1024-episode eval; chance 10%.
| seed | A before B | A after B, plain training | A after B, uniform-noise dreams (note 133) | A after B, statistics-matched dreams | B after B (stat dreams) |
|---|---|---|---|---|---|
| 0 | 70.1% | 1.4% | 63.5% | 70.4% | 91.9% |
| 1 | 71.3% | ~5% (random eval) | 70.4% | 72.1% | 90.9% |
| 2 | 73.4% | ~0% (random eval) | 56.2% | 72.9% | 92.0% |
| mean | 71.6% | ~2% | 63.4% | 71.8% | 91.6% |
Plain training learns B to 96.5% (seed 0, fixed eval).
## Reading
- Forgetting of task A is eliminated within eval noise on all 3 seeds (71.8% after vs 71.6% before), with no stored A examples; only 256 numbers kept.
- Cost: task B is about 5 points lower than plain training (91.6% vs 96.5%), and consolidation roughly doubles training compute (MODELLED) plus a frozen copy of the weights during learning.
- Earlier variants for comparison (seed 0 unless noted): uniform-noise dreams 47% mean (strength 1), 63.4% (strength 100, 3 seeds); EWC 21%; binarised EWC 26%; 8 raw episodes 29%; adapters 65% but B only 59%.
- Not yet shown: transformer under the same method (running), smaller strength to recover B (running, 25), longer task sequences A->B->C, language-scale data.

## Transformer under the same method (seed 0, MEASURED)
A before B 36.8%; plain training A after B 0.0% (B 33.0%); statistics-matched dreams A after B 35.5% (B 38.5%).
Reading: the method also stops forgetting in the transformer (96% of A kept), so it is a training-procedure result, not L5-only. L5's advantage here is that it learns both tasks about twice as well (A ~72% vs 37%, B ~92% vs 38%) at matched size.

## Lower dream strength (25), seed 0: A after B 69.8% (before 70.1%), B 94.7% (plain 96.5%). Better trade-off than strength 100 (B 91.9%). Seeds 1-2 running.

## Strength 25, 3 seeds (MEASURED): A after B 69.8 / 72.6 / 73.0% (before 70.1 / 71.3 / 73.4%; mean 71.8 vs 71.6), B 94.7 / 90.6 / 96.5% (mean 93.9%). Recommended setting: 25 (keeps A fully, B costs ~2-3 points vs plain).
