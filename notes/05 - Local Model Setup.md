---
tags: [setup, local-models, ollama]
created: 2026-10-05
---
# Local Model Setup

Back to [[00 - Concept and Roadmap]]

## Earlier Ollama inventory (superseded by verification below)
| Model | Size | Note |
|---|---|---|
| **qwen3:14b** | **9.28 GB** | The "14B" model Kakashi asked to remove |
| qwen3:8b | 5.23 GB | Shares template/license files with 14b |
| qwen3.5:9b | 6.59 GB | |
| nomic-embed-text | 0.27 GB | Embedding model |
(D:\AI\Ollama is empty.)

**Verified correction (05 Oct 2026):** qwen3:14b and qwen3:8b were removed successfully with Ollama and their absence was verified. Remaining models: qwen3.5:9b and nomic-embed-text. The table above preserves the earlier inventory.

## Spiking models: can they run locally?
| Model | Base | Runs in Ollama / llama.cpp? | Needs |
|---|---|---|---|
| SpikingBrain-7B (2025) | Upcycled Qwen2.5-7B | ❌ No (custom `GLAswaForCausalLM` architecture, no GGUF) | NVIDIA GPU, PyTorch + flash-attn (Ampere+) or vLLM-HyMeta plugin; ~15+ GB VRAM in BF16; weights on ModelScope (incl. W8ASpike quantised) |
| SpikingBrain2.0-5B (Apr 2026) | Converted Qwen3-4B | ❌ Not mentioned | Optimised for Hopper GPUs (FP8) and neuromorphic chips; code/weights at BICLab GitHub (check release) |

⚠️ **Critique to remember:** SpikingBrain's spikes are **simulated on GPUs** (sparsity packaged for GPUs), not event-driven neuromorphic execution, and the 7B is a converted Qwen2.5 that "trades a little quality" for long-context speed.

## Plan options
1. **Study the code** (architecture, spike encoding) without a GPU, in the cloud workspace.
2. **Run it on a cloud GPU** (Kaggle / Colab / rented A100), since there's a Kaggle setup on the laptop.
3. Run locally **only if** the laptop GPU has enough VRAM (to check: GPU model and VRAM).

## Sources
- SpikingBrain-7B repo: https://github.com/BICLab/SpikingBrain-7B
- GGUF not supported: https://huggingface.co/mradermacher/model_requests/discussions/1425
- Critique: https://articles.akadata.co.uk/past-the-hype-spikingbrain-7b-qwen2-5-7b-in-new-clothes/
- SpikingBrain2.0: https://arxiv.org/html/2604.22575v1

## Device and study verification — 05 Oct 2026
- CPU: Intel i7-13620H. RAM: 15.7 GiB total, approximately 2.3 GiB free during inspection. GPU: RTX 3050 Laptop, 6 GB; approximately 6 GB free during one check. WSL2 Ubuntu installed. Availability changes with running apps.
- Official sources downloaded under D:\AI\SpikingBrain-study. No trained SpikingBrain weights downloaded. All new experiment files, caches and vault artifacts on D:.
- Existing Python 3.12 uses PyTorch 2.14.0+cpu: CUDA is unavailable in this Python environment. CPU encoder demo completed successfully; this does not indicate GPU hardware incompatibility.
- W8ASpike checkpoint header confirms F32 storage, roughly 30.8 GB weights. Its name does not mean a 7 GB packed INT8 checkpoint. Source simulates quantization with floating-point weights.
- 5B instruct configuration uses BF16. No ready-to-run 5B 8-bit or 7B 4-bit checkpoint was verified. Generic CPU offloading does not replace model-specific GPU operators.
- [[07 - SpikingBrain Neuron Experiments]]: 15/15 checks passed, plots and raw results.
- [[08 - SpikingBrain Process Log]]: preparation, execution and verification record.
- [[09 - SpikingBrain Architecture and Prototype Decisions]]: comparison and next prototype design.
- Earlier critique and feasibility statements above are historical context, not new experimental findings.
