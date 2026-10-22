"""H1 - energy proportionality: does L's per-token work track how hard the token is?
MEASURED: spike events per token (from the model). MODELLED: MACs per token with event-driven 'out' and 'f2' (common.py).
ESTIMATED: energy with the published 45 nm per-operation energies of Horowitz (ISSCC 2014): 8-bit int mult 0.2 pJ + add 0.03 pJ,
SRAM read (8 KB, 64-bit) ~10 pJ -> ~1.25 pJ/byte. These are for ORDER OF MAGNITUDE only; they are not Tang Nano numbers.
Difficulty = the model's own loss on the token (bits) and its top-1 confidence. Decile analysis on EVAL only (nothing is fitted)."""
import numpy as np
from scipy.stats import spearmanr
from common import load, macs, tags, save_csv, plt, OUT
E_MAC = 0.23e-12; E_BYTE = 1.25e-12      # ESTIMATED (Horowitz 2014, 45 nm), stated assumption
rows = []
for tag in tags():
    r = load(tag, 'eval'); M = macs(r); L, D = r['L'], r['D']
    ns = (r['state'] != 0).sum(2); nf = (r['ffn'] != 0).sum(2)          # (T, L) spike events per layer
    ev = ns.sum(1) + nf.sum(1)                                            # events per token (all layers)
    event_macs = (ns.sum(1) + nf.sum(1)) * D
    dense_macs = L * M['dense_per_layer'] + M['head']
    total = dense_macs + event_macs; full_dense = L * (M['dense_per_layer'] + M['event_full_per_layer']) + M['head']
    energy = total * (E_MAC + E_BYTE)                                     # each MAC reads one 8-bit weight byte
    loss = r['loss']; q = np.quantile(loss, np.linspace(0, 1, 11)); dec = np.clip(np.searchsorted(q, loss, 'right') - 1, 0, 9)
    by = [dict(decile=k, loss_bits=loss[dec == k].mean(), events=ev[dec == k].mean(), work_rel_dense=(total[dec == k] / full_dense).mean()) for k in range(10)]
    rho_ev = spearmanr(ev, loss).correlation; rho_conf = spearmanr(ev, r['conf']).correlation
    res = dict(model=tag, tokens=len(loss),
               events_per_token_mean=ev.mean(), events_p5=np.percentile(ev, 5), events_p95=np.percentile(ev, 95),
               events_dynamic_range_p95_over_p5=np.percentile(ev, 95) / max(1, np.percentile(ev, 5)),
               spearman_events_vs_loss=rho_ev, spearman_events_vs_confidence=rho_conf,
               events_hardest_over_easiest_decile=by[9]['events'] / by[0]['events'],
               work_fraction_of_dense_mean=(total / full_dense).mean(),
               work_hardest_over_easiest_decile=(by[9]['work_rel_dense'] / by[0]['work_rel_dense']),
               dense_share_of_work=(dense_macs / total).mean(),
               loss_hardest_over_easiest_decile=by[9]['loss_bits'] / max(1e-3, by[0]['loss_bits']),
               est_energy_nJ_per_token_mean=energy.mean() * 1e9, est_energy_nJ_dense_ref=full_dense * (E_MAC + E_BYTE) * 1e9)
    rows.append({k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in res.items()})
    save_csv(f'h1_deciles_{tag}.csv', [{k: round(float(v), 4) for k, v in b.items()} for b in by])
    P = plt(); f, ax = P.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([b['loss_bits'] for b in by], [b['events'] for b in by], 'o-'); ax[0].set_xlabel('loss decile mean (bits)'); ax[0].set_ylabel('spike events / token (MEASURED)')
    ax[1].plot(range(10), [b['work_rel_dense'] for b in by], 'o-', label='L (event-driven out+f2)'); ax[1].plot(range(10), [1] * 10, '--', label='dense')
    ax[1].plot(range(10), [b['loss_bits'] / by[9]['loss_bits'] for b in by], ':', label='ideal: work ∝ loss'); ax[1].set_ylim(0, 1.1)
    ax[1].set_xlabel('difficulty decile (EVAL)'); ax[1].set_ylabel('work / dense (MODELLED)'); ax[1].legend(fontsize=7)
    f.suptitle(f'H1 energy proportionality - {tag}'); f.tight_layout(); f.savefig(f'{OUT}/h1_{tag}.png', dpi=110)
    print(rows[-1])
save_csv('h1_summary.csv', rows)
