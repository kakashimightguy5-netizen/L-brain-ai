# Latest repair findings and runnable tests

[[40 - E Series Repair Benchmark Results]] · [[35 - Learned Weak Signal Memory Results]] · [[36 - Category Description Routing Results]]

We have tested every specifically named E-series failure and retained the negative results. The strongest practical changes are robust native fine-tuning for noisy perception, consolidation anchored to clean examples, teacher-initialized compression, and replay for continual learning. EWC alone, intrinsic correction and pure nonlinear genome compression remain weak.

![[SpikingBrain Assets/E-repair-overview.png]]

| Area | Matched result | Assessment |
|---|---|---|
| Native continual fine-tuning | 19.11%; replay 92.06% | Earlier examples are needed to avoid forgetting. |
| EWC | Corrected estimator 20.33% alone; independently tuned replay variant 90.83% vs replay 90.11% | Calculation fixed; EWC alone still fails. |
| E2 direct edits | Aggressive clean accuracy 47.67%; protected edits 95.50% | Collapse prevented, but few mistakes are repaired. |
| E2 sleep | Noisy 55.78% → 70.33%; clean 96.78% → 94.67% | Generalization improves; some clean skill is lost. |
| E2 intrinsic correction | Base 55.22%; smoothing 55.44% | Tiny, inconsistent gain; unresolved. |
| E3 perception | Mixed noise 61.17% → 72.83%; fixed noise 56.28% → 75.72% | Robust one-pass native fine-tuning is best overall. Gates cost more here. |
| E5 compression | Teacher 96.78%; rank 8 96.72%, checkpoint 2.85× smaller | Practical compression success. Genome hybrid 95.50%; pure genome remains weak. |
| E6 integration | Shared noisy accuracy 69.28% → 76.00%; matched baseline 75.83% | Improved, approximately tied with baseline. |

Rows belong to separate matched five-seed experiments, so do not compare unrelated row numbers as if they came from one test. These are CPU digit prototypes, not full SpikingBrain language-model fine-tuning. The detailed report includes seed variation, clean-skill tradeoffs, failed candidates and all validation choices.

The earlier memory/router repair remains separate: weak pulses at low noise improve the fraction of correctly answered inputs from 2.70% to 93.13%, but heavy noise and incomplete-event detection remain weak. Category descriptions improve correct answered unfamiliar intents from 31.19% to 75.63%, while unsupported rejection drops from 98.62% to 94.91%. The descriptions explicitly supply the held category names; this is not discovering unnamed categories.

## Runnable comparisons

All five demo modes were executed successfully on the existing CPU runtime. The tool loads saved native model weights; it does not download anything or train during a demo. By default it displays the first example the revised model repaired, which is a demonstration rather than a benchmark sample. Use `--index` to select any saved example, or `--input-json` for a file containing 64 normalized pixel values.

Script: `D:\AI\SpikingBrain-study\experiments\try_E_repairs.py`

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = 'D:\AI\SpikingBrain-study\experiments\temp'
$env:TMP = $env:TEMP
py -3.12 -B D:\AI\SpikingBrain-study\experiments\try_E_repairs.py --mode sleep
```

Other modes: `perception`, `compression`, `integration`, `ewc`. Their saved default seeds are 342, 242, 142 and 242. Custom images must be 8×8 digit inputs; these models do not process free-form language.

## Next research priorities

Keep replay, conservative correction memory and robust fine-tuning as the baseline. Larger next experiments should test routing and consolidation on a real labelled language task, calibration on unfamiliar categories, and reliable rejection of missing/ambiguous events. Pure EWC and intrinsic truth detection need new evidence rather than stronger claims. Full SpikingBrain native tuning requires a verified runnable checkpoint and a resource plan; none of the digit experiments substitutes for that.

All new scripts, checkpoints, cache/temp files, plots, reports and process records are on D:. Detailed benchmark tables: [[40 - E Series Repair Benchmark Results]]. Verification and copies are recorded in [[SpikingBrain Assets/E-repair-manifest.json]].
