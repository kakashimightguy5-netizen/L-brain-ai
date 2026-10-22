---
tags: [L, L-6, optimisation, hardware, L-HW, honest, corrections]
created: 2026-10-09
status: internal research record (in progress — final report in note 114)
---
# 113 - "Fix everything" round + L architecture changes plan (Phases 0–3)

Previous: [[112 - Why Only 5.6 Percent and GPU Speed Fix]] · Plan source: user's `L_architecture_changes_prompt.txt`

## L-6 (all quality fixes) at 9M
| Model | Bits/byte |
|---|---|
| Transformer 9.2M (3 runs) | 1.082 |
| L-5 8.8M (2 runs) | 1.0218 / 1.0218 |
| **L-6 8.86M (seed 0)** | **1.018** (seeds 1–2 running) |

- **Weak spots at 9M, by byte category** (`where_loses.py`): L now beats the transformer in **every** category, including bytes where the hippocampus recall was wrong (2.50 vs 2.59), the second half of the window (0.88 vs 0.94) and digits (2.72 vs 2.86).
- **Bug found and fixed:** newly added input signals started random (N(0,1)) and injected noise. They now start at zero, which explains why the trust/veto/layout tests had hurt.

## Fixes that were verified
| Problem | Fix | Proof |
|---|---|---|
| Language-column misrouting (L-3) | **Soft Bayesian routing**: blend all columns weighted by their running predictive likelihood (decay 0.97, chosen on training passages) | Avg 1.2254 = oracle upper bound 1.2255; Python forgetting +0.025 → **+0.0004**. Cost: runs all columns. The learned-router attempt failed (1.2659) |
| Effort gate destroyed full-depth quality | Post-hoc gate: train the model first, then the heads on the frozen model, then short adaptation + distillation | Full depth 1.3239 → **1.3227 (no loss)**. Gate 0.99: 5.3% compute saved for +0.5%. 0.95: 14% saved for +3.7% (too costly). Longer adaptation running |
| GPU overflow, random-init bug | one job at a time; zero-init | done |

## Phase 0 measurements (RTX 3050, alone on the GPU) — CORRECTIONS
| | L-6 | Transformer |
|---|---|---|
| Train ms/step (32×256) | 617 | **194** |
| Train energy above idle | 2.69 mJ/token | **0.90** |
| Peak VRAM | 4.66 GB | **1.68** |
| Kernel launches / step | 3,361 | **288** |

- **Correction 1:** the earlier "L trains only ~15% slower" was wrong (the transformer run had shared the GPU). The real figure is **3.2× slower per step**.
- **Correction 2:** L-6 is **not sparse** in the membrane (70–74% of spikes non-zero). The FFN is 12–27% per token, but 57–97% across a batch of 32.
- **Forward-time split:** fast-weight memory 57%, spiking layer 32%, hippocampus 5%, trajectory 5%.
- **Membrane recurrence is LINEAR** (a_t and b_t depend only on the input; no spike feedback or reset), so a parallel scan is exact. Code map: `docs/L_code_map.md`.

## Generation (batch 1, token by token) — where L wins
| Context | Transformer KV-cache (eager) | **L (CUDA-graph stepper)** |
|---|---|---|
| 256 | 9.3 ms/token | **1.38 ms** |
| 1,024 | 9.4 ms, 54 mJ above idle | **1.29 ms, 11 mJ** |
| 4,096 | 8.7 ms, 122 mJ | **1.29 ms, 14 mJ** |
| 16,384 | 25.4 ms, **518 mJ**, KV-cache 277 MB | **1.35 ms, 14 mJ, state 0.46 MB** |

- The L stepper equals the full forward to 1e-14 (test), and the graph version equals the plain stepper exactly (0.0).
- **Fairness fix pending:** the transformer decoder above had no CUDA graph. A graph version (static cache = context, masked) is written and verified (2e-15) and is queued.

## Phase 2 optimisation log (each behind a flag)
| Change | Equivalence | Effect | Verdict |
|---|---|---|---|
| Fused delta-rule CUDA kernel (DELTAKERN) | 5e-6 / 1e-14 | 1,164 vs 618 ms | ❌ off |
| CUDA graph, training step | — | no gain | ❌ |
| bf16 autocast | loss drift 1e-3 / 20 steps | −16% time, −20% VRAM | 🟡 needs a full quality run |
| SPIKE_MASK (bool mask saved for backward) | **bit-exact** | VRAM −3.5% | ✅ on for new runs |
| SCAN_MODE=hs (parallel scan) | exact | slower (867 ms), +23% VRAM | ❌ |
| Chunk size, block inverse | exact | slower | ❌ |
| GPU hippocampus/trajectory | 0 mismatches | trajectory 230 → 9 ms | ✅ |

## Phase 3 — L-HW (integer-only inference)
- `hw/l_int.py`: integer path with:
  - FA = 10 / FG = 16 fixed point;
  - int8 per-channel power-of-two weights;
  - 513-entry interpolated LUTs (sigmoid, 2σ−1, SiLU, softplus);
  - rsqrt by leading-one detection + 384-entry mantissa LUT;
  - spike counts as integers;
  - boost ×8 by shift.
  - No exp / div / sqrt / softmax in inference.
- **NumPy golden == PyTorch hw_mode, bit-exact** (`tests/test_hw_int.py`). `golden/l_int.py` (NumPy only) reproduces all exported test vectors bit-exactly.
- `export/export_hw.py` writes `.mem` hex files + `params.npz` + `config.json` + per-byte test vectors (spikes, active FFN indices, state checksums, hippocampus/trajectory, logits, output byte).
- `analysis/cost_model.py` (**ESTIMATE**): at L-6 firing rates, event-driven skipping saves only ~11% of MACs, because most of L is dense by design. State + hash tables + LUTs fit the Tang Nano 20K block RAM; weights stream from SDRAM.
- L-Edge (0.94M, d=128) is training on the cloud for the float vs integer quality comparison.
