# L-brain-ai - Project L

A brain-inspired, attention-free (non-transformer) language-model research project, built by a solo founder. Goal: strong quality per unit of compute/energy/memory, with a long-term target of dedicated hardware.

**Honest status.** All evidence is small scale (models under ~10M parameters, byte-level, mostly toy tasks plus Python-text bits/byte). Nothing here supports any claim about AGI. L is brain-INSPIRED, not a human brain. Negative results are kept in the notes on purpose.

## Layout
- code/ - model code (L5: spiking diagonal recurrence + delta-rule fast-weight memory + hash recall), training, experiments, energy tools
- cloud_edits/ - newer experimental edits made in the cloud workspace (barcode/sleep/wide/COT flags, NOMAT sparse ternary wiring) and their small result files
- jobs/, runner/ - job files and the GPU job runner
- results/ - small JSON/log results (no checkpoints)
- notes/ - the full research log (numbered notes; latest: 144-149 energy fix, fair comparison, NOMAT, L-pure spec)

## Key measured results (laptop RTX 3050 6GB)
- Training energy, both models compiled + bf16: transformer 0.41 mJ/token, L 0.84 mJ/token (L is ~2x worse on a GPU). Notes 144-146.
- Pruning trained L5 to sparse ternary wiring: +17% (K=32) to +76% (K=4) bits/byte, fails the 0.5% gate. Note 148.
- Wide ternary L5 (d=192) beats same-recipe fp d=128 by 2.2% bits/byte at half the weight memory. Note 136.

Checkpoints and datasets are not included (size); see notes for how they are produced.
