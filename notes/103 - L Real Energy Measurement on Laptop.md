---
tags: [L, energy, measurement, sustainability]
created: 2026-10-07
status: measured (single run)
---
# 103 - L-0 real energy measurement on the laptop

Previous: [[102 - L-0 Results First Non-Transformer Model]] · Goals: [[98 - Architecture L Charter]] (sustainability)

## Method
- **Kit:** `D:\AI\SpikingBrain-study\L-architecture\energy-kit\RUN_ENERGY_TEST.ps1`.
- **Hardware:** the user's Windows laptop (i7-13620H), CPU only (no GPU), on battery.
- **Power:** whole-laptop power drain from Windows WMI `BatteryStatus.DischargeRate`, sampled once per second.
- **Phases:**
  - 60 s idle (baseline);
  - 60 s of L-0 generating tokens one at a time;
  - 60 s of the transformer generating tokens one at a time;
  - all at 256 and at 16,384 tokens of context, with 20 s cool-downs between phases.
- **Models:** the trained L-0 and transformer from note 102 (about 1.1M params each), 4 CPU threads.
- **Note:** the transformer's 16K KV cache was filled with stand-in values. This does not change timing or energy.

## Results (measured)
Idle laptop: 22.26 W.

| Context | Model | Tokens/s | Avg power | Extra power over idle | **Energy per token (extra)** | Energy per token (whole laptop) |
|---|---|---|---|---|---|---|
| 256 | Transformer | 131.9 | 28.05 W | 5.80 W | 44.0 mJ | 212.7 mJ |
| 256 | **L-0** | **247.8** | 27.21 W | 4.95 W | **20.0 mJ** | **109.8 mJ** |
| 16K | Transformer | 18.5 | 26.37 W | 4.11 W | 222.1 mJ | 1,423.9 mJ |
| 16K | **L-0** | **190.9** | 26.02 W | 3.77 W | **19.7 mJ** | **136.3 mJ** |

## What it shows
- **L-0's energy per token stays the same at any context length** (20.0 → 19.7 mJ). The transformer's grows 5× (44 → 222 mJ).
- **L-0 uses 2.2× less energy per token on short text and 11× less on long text.** It generates 1.9× and 10× faster.
- **Water:** none from data-centre cooling when run locally. In a data centre, water scales with energy, so the same ratios apply.
- **Hardware:** an ordinary laptop CPU, no GPU, so no new hardware and no new e-waste.

## Limits
- One run, 1 s power sampling, whole-laptop measurement (background activity adds noise).
- Small models (~1M params). Ratios at large scale still need testing.

Raw data: `energy_results.json`, `power.csv`, `markers.csv` in the kit folder.
