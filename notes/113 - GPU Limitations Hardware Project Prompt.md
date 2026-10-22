---
tags: [L, hardware, gpu, accelerator, prompt, project]
created: 2026-10-09
status: prompt for a hardware project built on L's measured GPU limits
---
# 113 - Hardware project prompt: fixing the GPU limits L runs into

Context: [[111 - L-5 Wins at 9M on Laptop GPU]] · [[112 - Why Only 5.6 Percent and GPU Speed Fix]]

## Prompt (copy-paste)

> I am an AI & Data Science engineering student. For a college **hardware project**, I want to design and build a small hardware accelerator (or a GPU improvement) for a new brain-inspired AI model my team built, called **L**.
>
> **What L is (no transformer, no attention):** a byte-level language model that runs like a brain circuit. Each layer has:
> 1. **Spiking state neurons:** a diagonal recurrence `h_t = a_t·h_{t-1} + b_t` with a in [-1, 1], whose output is turned into small integer **spike counts** (−8..+8) with an adaptive (homeostatic) threshold.
> 2. **An error-correcting fast-weight memory (delta rule):** per head, a small matrix `S` (44×44, 8 heads) updated each byte as `S ← a·S + b·(v − a·S·k)·kᵀ`, and read as `S·q`.
> 3. **A short delay line:** a 4-tap convolution.
> 4. **A sparse spiking feed-forward block:** only **2–10% of its neurons fire**.
>
> Next to the layers sits a **hippocampus**: an exact hash index of earlier 3/6/12-byte patterns that recalls what came next, plus a calibrated copy mechanism.
> **Cost per byte is constant** (no cache that grows with text length).
>
> **Measured results** (8.8M parameters, laptop RTX 3050 6 GB):
> - L beats a same-size transformer on held-out Python code by about 6% (1.018 vs 1.082 bits/byte).
> - It copies text from 2,048 bytes back with 95% accuracy.
> - Energy: 20 mJ/token at both short and long text; the transformer uses 44 mJ/token at 256 bytes of context and 222 mJ/token at 16K.
>
> **GPU limits we measured** (these are the problems to solve in hardware):
> 1. **Sequential recurrence.** The membrane scan and the memory update depend on the previous byte. GPUs prefer huge parallel matrix maths, so they run this with only about 11k threads, each looping over time: low utilisation.
> 2. **Sparsity is wasted.** Only 2–10% of feed-forward neurons spike, but the GPU still multiplies every zero (dense compute). Spike counts are small integers, but the GPU computes them in 32-bit float.
> 3. **Many tiny operations.** About 3,300 small GPU kernel launches per training step. The CPU spends time launching them instead of the GPU computing.
> 4. **Small-matrix work.** The memory uses thousands of 64×64 triangular solves and 44×44 updates. These map poorly onto tensor cores, and the backward pass costs about 3× the forward. One memory layer takes 36 ms per step, about 60% of the step time.
> 5. **Irregular memory access.** The hippocampus is a hash/lookup structure (exact pattern match). On a GPU it is done with sorting, which is wasteful for what is really a content-addressable lookup.
> 6. **Small VRAM and spill.** With 6 GB, two training jobs overflow memory. Windows then spills to system RAM and everything runs about 5× slower.
> 7. **Result:** training is still about 15% slower than an equally sized transformer, even though L needs less compute per token in principle.
>
> **What I want from you:**
> 1. Propose 2–3 hardware project designs, from easy to ambitious, that directly attack these limits. Candidates:
>    - an FPGA co-processor for the spiking recurrence and fast-weight memory;
>    - an event-driven sparse spike engine that only computes for neurons that fire;
>    - a content-addressable memory (CAM) block for the hippocampus lookup;
>    - on-chip SRAM holding each layer's state (≈62 KB of memory matrices per layer) so nothing goes back to DRAM between bytes;
>    - integer/low-bit arithmetic for spikes.
> 2. For the best design, give: block diagram, data flow per byte, what runs on the host PC vs the accelerator, the arithmetic precision, memory sizes, and the expected speed and energy versus the RTX 3050.
> 3. Suggest a realistic student build: board options (e.g. PYNQ-Z2 / Zynq, Arty A7, Kria KV260, or an ESP32/RP2040 + FPGA combo), approximate cost in INR, tools (Vivado/Vitis HLS, Verilog/VHDL, or HLS C++), and a week-by-week plan for about 8–10 weeks.
> 4. Explain how we will **measure success**:
>    - bytes/second and energy per byte (mJ/token) vs the laptop GPU;
>    - the same outputs as the PyTorch reference (bit-exact or within a set tolerance);
>    - a live demo: the board continues/copies Python code.
> 5. Point out risks (FPGA resource limits, fixed-point accuracy, host-board bandwidth) and how to reduce them.
> 6. Connect it to sustainability: lower power, less cooling water, and longer-lived small hardware instead of large GPU clusters.
>
> Keep the scope buildable by 3–4 students in one semester. Start with **inference** (running the trained model) on the hardware; training can stay on the GPU.

## Why these are the right targets (from our own measurements)
| Measured limit | Evidence (note) |
|---|---|
| Delta memory = largest cost | 36.5 ms fwd+bwd per layer, the biggest share of a 617 ms step (11c micro-profile) |
| Launch overhead | ~3,300 kernel launches/step (7a profile) |
| Sparsity unused | FFN 2–10% active, state neurons 22–65% active (note 111) |
| VRAM spill | Two 4M jobs = 6.3 GB on 6 GB → ~5× slower (note 112) |
| Speed gap now | L-6 9M: 64 min vs transformer 55 min (was 133 min) |
