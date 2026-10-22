---
tags: [L, L-0, results, non-transformer, spiking, sustainability]
created: 2026-10-07
status: internal research record
---
# 102 - L-0 results: the first L model (no transformer, no attention)

Previous: [[98 - Architecture L Charter]] · [[100 - LLM Problem Atlas]] · [[101 - Training Without GPU Clusters Plan]]

## What L-0 is
- **Cortical sheet:** 4 layers of neurons with a membrane state `h = a·h + b`.
  - The leak `a` is input-gated and lies in [-1, 1], so a neuron can flip sign. This is what lets it track state.
  - Neurons send **integer spike counts** to each other (−4…4 for the state, 0…4 for the feed-forward part), trained with a surrogate gradient.
  - The cost per token is fixed and **does not grow with text length**. There is no KV cache.
- **Hippocampus:** a hash index of the last 3, 6 and 12 bytes.
  - It stores the position where each byte pattern was last seen (pattern separation). On a repeat, it recalls the byte that came next (pattern completion).
  - Each lookup costs O(1), with no scan over the past.
  - A learned **trust gate** decides when to use the recalled byte (self-knowledge).
- **Training:** from scratch on Python source code (CPython stdlib, 1.8 MB), at byte level, 2,000 steps of 16×256 bytes, on 1 CPU thread.
- **Competitor (outside L only):** a same-size transformer (1.23M params) trained identically.

## Results
| Test | Transformer | L core (no hippocampus) | **L-0 full** |
|---|---|---|---|
| Params | 1.23M | 1.01M | 1.14M |
| **Quality: held-out code, bits/byte** (lower is better) | 1.909 | **1.816** | **1.615** |
| **Exact copy of NEW random text**, 127 / 2,048 / 8,192 bytes | 3% / cannot / cannot | 2% / 2% / 2% | **95% / 98% / 96%** |
| **Exact copy of held-out code**, 127 / 2,048 / 8,192 bytes | 68% / cannot / cannot | 71% / 67% / 70% | **94% / 93% / 92%** |
| **State tracking (parity)**, train length 64 → test 64 / 256 / 1,024 (2 seeds) | 56 / 51 / 51% (chance) | — | **100 / 95 / 64%** |
| Same test, L with leak in [0, 1] only (as in Mamba) | — | 59 / 53 / 51% (fails) | — |

"Cannot" means the text is longer than the transformer's trained context of 256 bytes.

### Cost per generated token (CPU, 1 thread, batch 1)
| Context already read | 256 | 1K | 4K | 16K | 64K |
|---|---|---|---|---|---|
| Transformer ms/token | **1.45** | 2.31 | 10.9 | 46.4 | **201** |
| L-0 ms/token | 2.05 | 2.05 | 2.05 | 2.05 | **2.05** |
| Transformer memory (KV cache) | 1.2 MB | 4.8 MB | 19 MB | 78 MB | **304 MB** |
| L-0 memory (state + compact hippocampus index) | 2.5 KB + 6 KB | +24 KB | +96 KB | +384 KB | **2.5 KB + 1.5 MB** |

- **Check with a really filled index:** 3.4 ms/token at both 4K and 64K entries, so the cost is flat. This was measured while other jobs shared the CPU.
- The transformer is faster below about 1K tokens of context. L is **about 100× cheaper per token at 64K** and uses **about 200× less memory**.

### Energy per token (compute-only ESTIMATE, Horowitz 45 nm, memory traffic excluded)
| | 256 context | 4K | 64K |
|---|---|---|---|
| Transformer | 6.7 µJ | 28 µJ | 372 µJ |
| L-0 (event-driven spikes) | **2.6 µJ** | **2.6 µJ** | **2.6 µJ** |

These are estimates. Measured joules on the laptop (CPU power readout) are still needed. Water: no data-centre cooling when L runs locally on a CPU.

### Spike sparsity
- **Feed-forward neurons:** only **12–21% active** per token. This is the part that can run add-only and event-driven.
- **State neurons:** 70–80% active, so not sparse yet. This is the next target.

## L-0b: can layers learn without depending on each other?
| Training | Bits/byte |
|---|---|
| End-to-end backprop (L core) | 1.816 |
| **Local only:** each layer has its own goal, gradient stopped between layers | **1.917** (+0.10) |
| Local + top-down hint | 1.922 (no help) |
| (End-to-end transformer, for reference) | 1.909 |

- **Breaking the layer dependency cost 0.10 bits/byte (5.5%)**. Even so, the decoupled L is **as good as an end-to-end transformer of the same size**. This is the first evidence that L's layers can be trained separately, i.e. on separate machines without a GPU cluster.
- The top-down hint (matching the top prediction) did not help. It only shapes the readouts, not what lower layers send upward. Next to try: a learned forward target (synthetic gradient) or a delayed real gradient from the next layer only.

## Honest limits
- **Very small scale:** about 1M params, 1.8 MB of code, 2,000 steps, a single seed for the language runs. The transformer may be undertrained, and small data can favour recurrent models.
- **The hippocampus index is hand-designed** (exact byte patterns), not learned. It copies exact repeats very well but cannot do fuzzy recall yet. A transformer might learn copying with much more training. Its cost would still grow with context.
- **Training is about 1.8× slower in wall time** (37 vs 21 min). The cause is the Python chunked scan, not the cost per token; a fused kernel would fix it.
- **Parity generalisation fades at 16× the training length** (64%).

## Next
1. Learned hippocampus keys (fuzzy recall), and sparser state spikes.
2. Better decoupled training (forward targets), and columns trained in parallel processes.
3. More seeds, more data, bigger L, and measured energy on the laptop.

## Code
`lab-code-L/`: l_model.py · train_L0.py · train_L0b.py · parity_L0.py · cost_L0.py · energy_L0.py · results_*.json
