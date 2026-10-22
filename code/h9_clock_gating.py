"""H9 - clock-gating opportunity: how often are event-driven lanes / blocks / whole units idle?
MEASURED from model activity (EVAL): per-neuron idle share, share of aligned blocks of B neurons with no spike in a token
(a block's lane could be clock-gated for that token), and share of tokens where a layer's whole FFN-f2 unit receives no spike.
MODELLED: upper bound on gate-able MAC share = idle share x event-unit share of all work (dense units are never idle)."""
import numpy as np
from common import load, macs, tags, save_csv, plt, OUT
rows = []; P = plt(); f, ax = P.subplots(figsize=(6, 3.8))
for tag in tags():
    E = load(tag, 'eval'); L, D, F = E['L'], E['D'], E['F']; M = macs(E)
    total_full = L * (M['dense_per_layer'] + M['event_full_per_layer']) + M['head']
    for unit, key, N in (('f2 / FFN spikes', 'ffn', F), ('out / state spikes', 'state', D)):
        A = E[key] != 0; T = len(A); share = L * N * D / total_full; ys = []
        for B in (1, 4, 8, 16, 32, 64):
            if N % B: continue
            idle = 1 - A.reshape(T, L, N // B, B).any(-1).mean()
            rows.append(dict(model=tag, unit=unit, block=B, idle_block_share=round(float(idle), 4), unit_share_of_dense_work=round(share, 4),
                             gateable_share_of_all_work=round(float(idle * share), 4))); ys.append(idle); print(rows[-1])
        ax.plot([b for b in (1, 4, 8, 16, 32, 64) if N % b == 0], ys, 'o-', label=f'{tag} {unit}')
        whole = (~A.any(2)).mean(0)
        rows.append(dict(model=tag, unit=unit, block='whole layer', idle_block_share=round(float(whole.mean()), 4), unit_share_of_dense_work=round(share, 4),
                         gateable_share_of_all_work=round(float(whole.mean() * share), 4))); print(rows[-1])
ax.set_xscale('log', base=2); ax.set_xlabel('clock-gate block size (neurons)'); ax.set_ylabel('idle share (token x block)'); ax.set_title('H9 idle fractions (MEASURED)')
ax.legend(fontsize=7); f.tight_layout(); f.savefig(f'{OUT}/h9_clock_gating.png', dpi=110)
save_csv('h9_clock_gating.csv', rows)
