---
tags: [architecture, L, charter, non-transformer, spiking]
created: 2026-10-07
status: charter recorded; first build waiting for approval
---
# 98 - Architecture L: Charter

Previous: [[97 - Decision Drop Transformer Base]] · Principles: [[20 - Final Benchmark Summary]]

**Name:** **L**, after the character from *Death Note*.
**Thesis (user's own words, 2026-10-07):** "The entire transformer is an inefficient architecture. We propose a new architecture called L. It does not inherit or use the transformer."

## Hard rules
1. **No transformer anywhere in L:**
   - no self-attention or attention blocks;
   - no transformer layers or KV cache;
   - no pretrained transformer weights;
   - no distillation from a transformer teacher.
2. Trained **from scratch**. Qwen and all earlier transformer work stay archived and untouched.
3. A transformer may appear **only as an outside competitor in benchmarks**, to prove or disprove the thesis, never inside L.

## Why the transformer is inefficient (the claims L must beat, each one measurable)
| Transformer cost | L's answer (brain analogue) |
|---|---|
| Attention cost grows with the square of the context | **Recurrent spiking state**: fixed cost per token (membrane potential + slow synaptic traces) |
| KV-cache memory grows with every token | **Constant memory** per token, plus a small **hippocampus store** for exact recall (copying, names) |
| Every parameter fires for every token | **Spiking sparsity + skill columns**: only the relevant column fires |
| Same compute for easy and hard tokens | **Thalamus effort gate**: think longer only when needed |
| Learning something new overwrites old knowledge | **Append-only columns** (GPT's growth_core) + protected sleep + self-healing |
| Does not know what it does not know | **Self-knowing "mine" + surprise routing + novelty growth** (Rounds 6–8) |

## Building blocks already proven in this vault
- LIF neurons (Round 1, GPT's LIF experts).
- Columns and routing (Round 8: 99% routing, novelty AUROC 0.92–0.97).
- Effort gate (−62.5% compute).
- Sleep and healing (Round 4).
- Abstention (Round 7).
- Append-only growth and self-healing (notes 66–71).

## Build order (not started)
1. **L-0:** a byte-level spiking recurrent language model on code text. Compare it against a same-size transformer (outside competitor) on bits per byte, compute per token, memory vs context length, and speed on CPU.
2. **L-1:** add the hippocampus exact-recall store; test exact copying and long-range recall.
3. **L-2:** add skill columns, surprise routing and novelty growth; test learning new languages without forgetting.
4. **L-3:** add the effort gate, then scale on the RTX 3050 and move to code tasks and tool use.
