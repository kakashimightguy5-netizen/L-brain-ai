"""H3 - hot neurons: can a small, FIXED on-chip cache of f2 weight columns ('hot' FFN neurons) serve most spike events?
Hot set ranked by firing frequency on CAL (per layer, same share of each layer, and globally across layers); coverage measured on EVAL.
coverage_events = share of FFN spike events (non-zero counts) whose column is in the hot set (MEASURED);
coverage_mass   = same, weighted by spike count.  Cache size (MODELLED) = hot columns x D weights x bits / 8.
Also reports the full model's f2 storage for comparison, and L-Edge numbers against the Tang Nano 20K block RAM
(GW2AR-18: 828 Kbit BSRAM = 103.5 KB, Gowin datasheet DS226 - quoted, not measured)."""
import numpy as np
from common import load, tags, save_csv, plt, OUT
BSRAM_KB = 828 / 8
FR = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]; rows = []; P = plt(); f, ax = P.subplots(figsize=(6, 4))
for tag in tags():
    C, E = load(tag, 'cal'), load(tag, 'eval'); L, D, F = E['L'], E['D'], E['F']
    fc = (C['ffn'] != 0).mean(0); ev = E['ffn'] != 0; ms = E['ffn'].astype(np.float32)
    rank_l = np.argsort(-fc, 1); flat = np.argsort(-fc.ravel())
    dead_cal = (fc == 0).mean(); dead_eval = ((ev.mean(0) == 0)).mean()
    cov_plot = []
    for fr in FR:
        k = max(1, int(round(fr * F))); hot = np.zeros((L, F), bool)
        for l in range(L): hot[l, rank_l[l, :k]] = True
        hg = np.zeros(L * F, bool); hg[flat[:k * L]] = True; hg = hg.reshape(L, F)
        for name, H in (('per-layer', hot), ('global', hg)):
            ce = (ev & H).sum() / ev.sum(); cm = (ms * H).sum() / ms.sum(); n = H.sum()
            rows.append(dict(model=tag, ranking=name, hot_fraction=fr, hot_columns=int(n), coverage_events=round(float(ce), 4), coverage_mass=round(float(cm), 4),
                             cache_KB_8bit=round(n * D / 1024, 1), cache_KB_4bit=round(n * D / 2048, 1), all_f2_KB_8bit=round(L * F * D / 1024, 1),
                             fits_TangNano_BSRAM_4bit=bool(n * D / 2048 <= BSRAM_KB) if D == 128 else ''))
            if name == 'global': cov_plot.append(ce)
    rows.append(dict(model=tag, ranking='(never fire on CAL / EVAL)', hot_fraction='', hot_columns='', coverage_events=round(float(dead_cal), 4), coverage_mass=round(float(dead_eval), 4),
                     cache_KB_8bit='', cache_KB_4bit='', all_f2_KB_8bit='', fits_TangNano_BSRAM_4bit=''))
    ax.plot(FR, cov_plot, 'o-', label=tag)
    for r in rows:
        if r['model'] == tag: print(r)
ax.plot([0, .5], [0, .5], ':', c='gray', label='uniform'); ax.set_xlabel('hot share of FFN neurons (fitted on CAL)'); ax.set_ylabel('share of EVAL spike events covered')
ax.set_title('H3 hot-neuron coverage'); ax.legend(); f.tight_layout(); f.savefig(f'{OUT}/h3_hot_neurons.png', dpi=110)
save_csv('h3_hot_neurons.csv', rows)
