# 144 - Energy issue: assessment and fix plan (2026-10-22 IST). User priority: fix energy first, then proceed.
## What the energy problem actually is (from stored results; MEASURED unless marked)
1. TRAINING: L is slower and costlier than a matched transformer on the laptop RTX 3050: profile_L5.json (batch 16 x 256): transformer 291 ms per microbatch, L5 716 ms (after delta-memory speedups; 958 ms before) = 2.5x. Note 121: 2.9x mJ/token, 613 vs 197 ms/step, VRAM 6.66 vs 1.6 GB.
2. Cause: not arithmetic. The step launches ~3,290 small kernels; the top GPU time is elementwise (aten::mul, add_, copy_, clone, div, MulBackward) plus many small mm/bmm. It is launch- and memory-bound (profile_L5.json).
3. What was already tried: fused delta kernel was SLOWER (1164 vs 618 ms/step, speed_kernel.json); bf16 alone 510 vs 617 ms; CUDA graph alone 603; graph+bf16 494 ms (speed_variants.json) = -20%.
4. train_graph.json (energy per token) is NOT reliable: transformer eager fp32 shows 742 mJ/token with std 508 and 137 s/step while its bf16 run shows 1.9 mJ/token; L eager vs graph numbers also swing 2-3x between modes. Do not quote it; rerun cleanly before any claim.
5. Inference: tie at batch 1, L cheaper at long context/batch (note 121). Ternary wide models add ~2.2x add ops (MODELLED).
## Fix plan for training energy (all behind flags, default off, baselines preserved)
F1. Fuse the elementwise chain of each spiking layer + RMSNorm into fewer ops (custom autograd op or torch.compile if the laptop Triton works); target kernel count 3,290 -> <1,000. Check: loss identical to plain within 1e-4 on CPU first.
F2. CUDA graph + bf16 for the whole L5 step (already -20% time) after F1, static shapes.
F3. Mixed batching: bigger batch per launch (L launches are per-token-count independent), measure mJ/token vs batch.
F4. Cheaper spike path: SPIKEK 8 -> 4 or 2 only if bits/byte stays within 0.5% (quality gate).
F5. Truncated-credit training (SLTT-like): drop temporal gradient through neuron state; check quality on 3 seeds.
Success: L training mJ/token (above idle, NVML, n>=5 runs) within 1.5x of the transformer at the same bits/byte, quality within 0.5%. Failure: <20% reduction after F1+F2.
## Limits
Cloud has no GPU; energy can only be measured on the laptop runner, which must be online. CPU can verify numerical equivalence and kernel/op counts only.

## Progress 2026-10-22 IST
- Existing GPU scan is already a custom CuPy kernel (cuscan.py, self-verified), so the 3,290 launches come from the many other small ops (delta memory chunk maths, spike blocks, norms, hippocampus trajectory), not the scan. CPU op count per training step (MEASURED, torch profiler, batch 16 x 256): whole L5 ~11,270 aten ops; one spiking block 1,390; one delta memory 1,253 (many are cheap views).
- torch.compile on one spiking block on CPU (MEASURED): forward output identical (error 0.0), gradient error 6e-6 (float rounding), 73 s compile time. So the code is compile-compatible.
- Written but NOT yet run: code/compile_check.py + jobs/2000_compile_check.json (checks Triton availability on the laptop, kernel count and ms eager vs compiled, no power settings touched). The laptop runner was not picking up jobs (no .started file after ~3 min), so it must be started on the laptop (START_L_RUNNER_v2.ps1) before any GPU energy measurement.

## Finding 2026-10-22 ~12:00 IST: GPU scan kernel was silently OFF in runner jobs (MEASURED from job logs)
Logs of 1500f0/f1/g0 (compress_ft.py, laptop runner) all print: "cuscan self-test failed -> OFF: RuntimeError('element 0 of tensors does not require grad and does not have a grad_fn')". The self-test ran inside a torch.no_grad() context (teacher pass), failed, and the result was cached as VERIFIED=False for the whole process, so training fell back to the slow log-depth scan (many extra kernels, ~8 scan stages per layer). Quality is unaffected (both scans are exact); speed/energy of those jobs is worse than it needed to be. Older energy figures taken through other scripts may or may not have been affected - not checked.
Fix (committed to code/cuscan.py, original mtime guarded): the self-test now runs under torch.enable_grad(). Only jobs started after the fix pick it up; 1500f0/f1/g0 (already running) do not.
Next: compile_check (queued, waits for a free GPU slot), then an A/B of one L5 training step with the scan kernel ON vs OFF to measure the saving before touching anything else.

## compile_check result (MEASURED on laptop RTX 3050, 2026-10-22 ~11:50 IST)
Triton is NOT installed in the laptop venv (torch 2.11.0+cu128): torch.compile fails with TritonMissing, so the compiler-fusion route (F1 via torch.compile) is unavailable unless triton-windows is installed (not tried; compatibility unverified). Eager RMSNorm+spike chain fwd+bwd = 75 CUDA kernels (10.1 ms on a GPU shared with 3 running jobs). Routes left: (a) custom fused kernels via CuPy/NVRTC (already used for the scan), (b) CUDA graph, (c) the scan-kernel fix above.
Next measurement: code/scan_ab.py (interleaved A/B: scan kernel ON vs OFF vs matched transformer; ms/step, kernels/step, NVML energy). Run on laptop: ..\venv\Scripts\python.exe -B scan_ab.py from the code folder. Energy numbers are indicative while other jobs share the GPU.

## scan_ab result (MEASURED on laptop RTX 3050, 2026-10-22 ~11:52 IST; GPU shared with 3 comparison jobs; modes interleaved, 5 reps)
cuscan self-test now passes (max error 1.9e-06 -> ON). One L5 training step, batch 16 x 256:
| | ms/step (median) | CUDA kernels/step | total mJ/token |
|---|---|---|---|
| scan kernel ON (fixed) | 476.3 | 2,365 | 4.374 |
| scan kernel OFF (old behaviour) | 640.4 | 3,257 | 5.922 |
| matched transformer | 101.6 | 258 | 0.956 |
Fix = -26% time, -27% kernels, -26% energy per token. L is still ~4.7x the transformer in time and ~4.6x in energy under this contended GPU (ratio may be inflated by contention; earlier uncontended profile gave 2.5x). 'Above idle' energy is INVALID here (idle reading taken while other jobs ran; values came out negative).
Next: code/graph_ab.py (eager vs CUDA graph, fp32 and bf16, loss-agreement check) to attack the remaining 2,365 launches.

## graph_ab result (MEASURED on laptop RTX 3050, 2026-10-22 ~12:02 IST; GPU shared with 3 comparison jobs; interleaved, 5 reps; batch 16x256)
| config (scan kernel ON) | ms/step | total mJ/token |
|---|---|---|
| eager fp32 | 476.5 | 4.475 |
| CUDA graph fp32 | 431.3 | 3.941 |
| eager bf16 | 431.8 | 4.163 |
| CUDA graph + bf16 | 366.5 | 3.301 |
| transformer eager fp32 | 102.0 | 0.947 |
Versus the pre-fix behaviour (scan OFF, eager fp32: 640.4 ms, 5.922 mJ/token): scan fix + graph + bf16 = -43% time, -44% energy/token. L is still ~3.6x time and ~3.5x energy of the transformer, and the transformer was NOT given graph/bf16 (unfair to the transformer; true gap probably larger).
Caveats: last-loss values differ (2.38-2.71) because graph runners did 3 extra warm-up steps -> NOT a valid equality check; bf16 quality not yet measured (needs the 0.5% bits/byte gate). Above-idle energy invalid (idle taken under load).
Reading: graph replay removes launch overhead yet the step is still 431 ms vs 102 -> the GPU is doing real extra work, not only launching. Next: code/profile_modules.py (GPU time per module family + top kernels) to find the heavy parts.

## profile_modules result (MEASURED, laptop RTX 3050, 2026-10-22 ~12:03 IST, GPU shared; one fp32 training step, scan kernel ON, batch 16x256)
forward 153.6 ms (delta memory x4 = 81.0, spiking blocks x4 = 39.2, other = 33.4 [embeddings, hippocampus boost, head, loss]); backward 324.6 ms; optimizer 2.4 ms; CUDA kernel time 371 ms; 2,366 launches.
By kernel type: elementwise mul/add/copy/neg/fill ~46% of GPU time (memory-bound: top three kernels 41.5 + 36.8 + 30.9 ms, ~160-190 calls each); fp32 SGEMM ~24%; batched triangular solves (delta memory) ~4%; depthwise conv ~2%.
Reading: the cost is memory-bound elementwise work, mostly the delta memory's chunk maths (many (B,H,N,C,C) tensors with C=64), not arithmetic. Lever: smaller DELTA_C halves that elementwise volume (same exact math). Next: code/delta_c_sweep.py (C in 16/32/64/128, equivalence check + ms/step fp32 and bf16). Second target: the 33 ms 'other' (hippocampus boost loop of 3 scatter_adds).

## delta_c_sweep result (MEASURED, laptop RTX 3050, 2026-10-22 ~12:08 IST, GPU shared; eager, batch 16x256, 5 reps) - NEGATIVE
Output identical to C=64 within float rounding (max |logit diff| 6e-7). ms/step fp32: C16 594.4, C32 498.2, C64 476.8, C128 516.4; bf16: C16 505.7, C32 441.9, C64 432.9, C128 474.6. Peak VRAM 0.75 GB. Hypothesis (smaller chunks halve the elementwise volume) was WRONG: more sequential chunk steps and launches cost more than smaller tensors save. DELTA_C=64 stays.
## Triton on Windows (research, UNVERIFIED for torch 2.11)
Project page (github.com/woct0rdho/triton-windows): `pip install -U "triton-windows<3.7"`; table maps torch 2.9->triton 3.5, 2.10->3.6; torch 2.11 not listed (3.6.x is the likely match, unconfirmed). Bundles TinyCC so no MSVC needed for GPU kernels (CPU torch.compile would need a C++ compiler). RTX 30xx supported. Reversible: pip uninstall triton-windows.
Next: install into the laptop venv, rerun code/compile_check.py; if compile works, torch.compile the delta-memory chunk maths and spiking block and re-measure with scan_ab/graph_ab-style A/B.

## Triton installed: compile works (MEASURED, laptop RTX 3050, 2026-10-22 ~12:23 IST)
`pip install -U "triton-windows<3.7"` -> triton-windows 3.6.0.post26 works with torch 2.11.0+cu128. compile_check: RMSNorm+spike chain fwd+bwd 75 -> 11 CUDA kernels, 10.06 -> 1.09 ms (about 9x on this chain; GPU shared), max error 0.0.
## Making the real model compile (cloud, MEASURED correctness)
Whole L5 under dynamo.explain: graph breaks (SpikeCount autograd.Function inside LLayer). Fix: flag SPIKE_STE=1 (default OFF) in l_model.py: spikes() built from plain tensor ops, y = clamp(x, lo-0.5, hi+0.5) + (clamp(round(x), lo, hi) - that).detach(). Checked on 4000 random values, ranges (-8,8) and (0,8): forward max diff 0.0, gradient max diff 0.0, 0 mismatches. With it the spiking block compiles as 1 graph, 0 breaks; DeltaMemory already 1 graph, 0 breaks. Original kept as l_model_before_ste.py (cloud); laptop l_model.py committed with mtime guard.
Next: code/compile_ab.py (per-variant subprocess; eager vs compiled blocks with cuscan or compiler-fused log-depth scan, fp32 and bf16; equivalence of logits/gradients vs eager; ms/step, kernels/step, mJ/token).

## compile_ab result (MEASURED, laptop RTX 3050, 2026-10-22 ~12:36 IST; GPU shared; one subprocess per variant; batch 16x256; step = fwd+bwd+AdamW)
| variant | ms/step | kernels/step | mJ/token | max logit diff vs eager | grad cosine |
|---|---|---|---|---|---|
| eager fp32 | 477.7 | 2,365 | 4.218 | - | - |
| compiled + CuPy scan, fp32 | 278.0 | 1,133 | 2.617 | 0.0459 | 1.00013 |
| compiled + fused log-depth scan, fp32 | 303.3 | 1,189 | 2.788 | 0.0459 | 1.00013 |
| compiled + CuPy scan, bf16 | 192.2 | 1,298 | 1.828 | 0.0948 | 0.99999 |
| compiled + fused scan, bf16 | 207.8 | 1,338 | 1.972 | 0.1152 | 0.99997 |
Best (compiled + CuPy scan + bf16): -60% time, -57% energy/token vs eager fp32; vs the original scan-OFF behaviour (640.4 ms, 5.92 mJ/token earlier run) about -70%. Versus the transformer (eager fp32, 102 ms, 0.95 mJ/token in the earlier run, NOT optimised) L is now ~1.9x in time and energy (was ~4.7x). Compile time 131-185 s for the CuPy variants.
OPEN: outputs are not bit-identical (logit diffs 0.05-0.12, gradients aligned). Quality NOT yet confirmed against the 0.5% bits/byte gate. Next: code/quality_ab.py (trained LEdge_s0: bpb under each numeric path, then 600 fine-tune steps with identical batches).
