---
tags: [L, SPARK, FPGA, estimate]
created: 2026-10-10
status: ESTIMATED on paper. No synthesis, no place-and-route, no board run. Not a measurement.
---
# 122 - Does ternary L-Edge fit on SPARK (Sipeed Tang Nano 20K)? Paper estimate

## Board resources (source: CNX Software listing for the Tang Nano 20K, which names GW2AR-18 but lists the part as GW2A-LV18QN88C8I7 - the page is inconsistent; verify against the Gowin datasheet before relying on it)
20,736 LUT4, 15,552 flip-flops, 828 Kb block SRAM (46 blocks, ~103.5 KB), 48 18x18 multipliers, 64 Mbit (8 MB) 32-bit SDR SDRAM, 2 PLLs.

## L-Edge numbers used (MEASURED/ESTIMATED from fix_eval, 1 seed, 1 checkpoint)
Ternary + low-rank 0.25 (tlr25): weights 105.2 KB packed (baseline 740.6 KB), 0 multiplies and 293,388 additions per byte, but +19.9% bits/byte (1.1218 vs 0.9359).

## Fit check (ASSUMPTIONS: 2-bit packed ternary, 100 MHz clock, 16-bit accumulators, 8-bit state)
- Weights: 105.2 KB vs ~103.5 KB on-chip block SRAM -> does NOT fully fit on chip with room for state; stream the remainder from SDRAM (8 MB, ample) or shrink (rank <0.25, drop embeddings).
- Fast-weight state: 4 layers x 4 heads x 32x32 x 8 bit = 16 KB (plus neuron/conv state) -> also contends for block SRAM; likely needs SDRAM or smaller heads.
- Arithmetic: 0 multipliers needed (48 DSPs idle). 64 parallel 16-bit adders ~ 1-2k LUT4 (<10% of the fabric). 293,388 adds/byte / 64 = ~4.6k cycles = ~46 us/byte at 100 MHz -> ~20k bytes/s; 256 adders (~4-8k LUT4) -> ~80k bytes/s. Order of magnitude only.
- Not covered: hippocampus hash tables (orders 3/6/12) and trajectory recall need history storage and hashing logic - unquantified; this is the likely real blocker, together with the +19.9% quality loss.
- Power/energy: no board measurement; the 100.5 nJ/byte chip figure in note 115 is a 45 nm Horowitz-model ESTIMATE, not an FPGA number. Do not quote it as SPARK energy.

## Verdict
Arithmetic and weight storage are plausible on paper (multiply-free, ~105 KB). Unresolved: (1) recall/hash memory, (2) state storage vs block SRAM, (3) the quality loss of tlr25, (4) true FPGA power. Next step is a tiny RTL/HLS sketch of one ternary matvec + resource report, which needs the Gowin toolchain (not available in this cloud container).
