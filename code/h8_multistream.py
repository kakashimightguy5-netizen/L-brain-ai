"""H8 - multi-stream interleaving: B independent byte streams decoded in lock-step share each weight fetch.
Streams = B non-overlapping segments of the EVAL trace (independent positions in held-out code). MEASURED activity, MODELLED traffic:
  dense weights (inp, f1, qkv, o, head) are read once per step for all B streams -> per-token traffic / B
  event weights (out, f2) are read for the UNION of the B streams' active columns
Per-stream state that must stay on chip (ESTIMATED, L-HW formats): membrane state D x 24 bit, conv buffer 3 x D x 16 bit,
delta memory heads x dk x dk x 16 bit, per layer; hippocampus tables are per-stream too but live in external memory (not counted).
Weights at 8-bit. Ratios matter; absolute bytes are model-size dependent."""
import numpy as np
from common import load, macs, tags, save_csv, plt, OUT
rows = []; P = plt(); f, ax = P.subplots(figsize=(6, 3.8))
for tag in tags():
    E = load(tag, 'eval'); L, D, F, H = E['L'], E['D'], E['F'], E['H']; M = macs(E); dk = D // H
    dense_B = L * M['dense_per_layer'] + M['head']
    state_B = L * (D * 3 + 3 * D * 2 + H * dk * dk * 2)                       # bytes per stream
    As, Af = E['state'] != 0, E['ffn'] != 0; T = len(As); ys = []
    for B in (1, 2, 4, 8, 16, 32):
        seg = T // B; us = uf = 0
        for t in range(seg):
            idx = [t + b * seg for b in range(B)]
            us += As[idx].any(0).sum(); uf += Af[idx].any(0).sum()
        ev_B = (us + uf) / seg * D; per_tok = (dense_B + ev_B) / B; single = dense_B + (As.sum() + Af.sum()) / T * D
        rows.append(dict(model=tag, streams=B, weight_bytes_per_step=round(dense_B + ev_B), weight_bytes_per_token=round(per_tok),
                         traffic_vs_1_stream=round(per_tok / single, 4), event_union_vs_sum=round((us + uf) / seg / ((As.sum() + Af.sum()) / T * B), 4),
                         on_chip_state_KB=round(B * state_B / 1024, 1)))
        ys.append(per_tok / single); print(rows[-1])
    ax.plot([1, 2, 4, 8, 16, 32], ys, 'o-', label=tag)
ax.plot([1, 32], [1, 1 / 32], ':', c='gray', label='perfect 1/B'); ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('interleaved streams B')
ax.set_ylabel('weight traffic per token vs 1 stream'); ax.set_title('H8 multi-stream interleaving (MODELLED)'); ax.legend(); f.tight_layout(); f.savefig(f'{OUT}/h8_multistream.png', dpi=110)
save_csv('h8_multistream.csv', rows)
