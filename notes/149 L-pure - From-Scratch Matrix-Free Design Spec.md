# 149 - L-pure: from-scratch matrix-free, multiply-free design (SPEC, 2026-10-22 IST) - nothing here is tested yet

Why from scratch: pruning a trained L5 into sparse ternary (note 148) lost +17% to +76% bits/byte. A model built from the right primitives from step 0 may learn to use them. Prior art to be honest about: MatMul-free LM (ternary BitLinear + elementwise recurrence), SpikeGPT/SNN LMs, Kanerva sparse distributed memory, BitNet, RWKV/Mamba/Gated DeltaNet (recurrent state), Loihi-style event hardware. What is intended as different: ONE small set of primitives for everything, delta-rule learning kept but made shift/add-only, and a measured joules-per-quality comparison.

## Allowed primitives (the whole model is built only from these)
integer add/subtract, compare, shift (power-of-two scaling), table lookup, saturating counters, XOR/popcount, top-k selection by compare. NO dense matrix multiply, NO general multiplier, NO divide, NO softmax in the deployed forward.

## Blocks
1. Embedding: byte -> table row (small ints). A lookup, not a matrix.
2. Spiking recurrence: v_t = v_{t-1} - (v_{t-1} >> s_c) + I_t ; leak = per-channel shift s_c (learned continuous log-decay, rounded to a power of two at deploy); spike count n_t = clamp(v_t >> p) with p a per-channel power-of-two threshold; reset by subtracting n_t << p. Adds, shifts, compares only.
3. Channel mixing (replaces dense matrices): sparse ternary wiring, each output reads K fixed inputs chosen by a structured topology (butterfly / power-of-two strides + a few random long links) so every channel reaches every other in log2(d) layers; weights in {-1,0,+1}; gain = shift. y_j = sum_k t_jk * n[idx_jk] -> adds only.
4. Delay line: shift register taps (already L's conv, 4-8 taps).
5. Associative memory (replaces the matrix delta state): key -> sparse binary address (sparse ternary projection + top-k compare, k of m rows). WRITE: for each active row r: M[r] += (v - read) >> lr (delta rule with power-of-two learning rate, saturating int counters). READ: o = sum of the k active rows (adds). Keeps "prediction = storage = learning" (candidate M1) with shifts/adds. Decay by shift.
6. Hippocampus: existing hash recall (lookup table), unchanged.
7. Normalisation replaced by homeostatic integer thresholds: p_c drifts by +-1 from the channel's firing rate (no divide, no RMS).
8. Readout: each of 256 byte classes reads K hidden spike counts with ternary weights (adds) -> integer logits; sampling/argmax by compare; training uses float softmax only.

## Training (offline, GPU/CPU) vs deployment
Train with backprop + straight-through estimators through ternary weights, rounded shifts and spike counts (quantisation-aware from step 0, ramped). Deploy the integer graph. Check: integer emulation == float forward within rounding. Local/backprop-free training stays a separate research track (every local rule tested so far cost +14-20%).

## Evaluation ladder (3 seeds each, honest reporting)
1 delayed copy 8 bits (cell sanity)  2 key-value recall N=4..128 (memory block)  3 hop2 chains  4 bytes of Python text bits/byte vs ternary d=192 (0.8287) and fp d=128 same recipe (0.8471)  5 op count per token (MODELLED) and GPU joules/token (MEASURED; GPUs will not reward this design)  6 hardware-spec numbers only from a simulator, never invented.

## Risks (stated up front)
Quality may fall well short of L5 (sparse wiring result in 148); structured topology may still under-mix; shift-only decays/learning rates may be too coarse; GPU energy will likely NOT improve (more small ops) - the case is for custom hardware and is unproven until simulated.
