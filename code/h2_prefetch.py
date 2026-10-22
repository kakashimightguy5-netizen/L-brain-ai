"""H2 - prefetch predictability of the FFN active set (which f2 weight columns step t will need).
All predictors use only information available BEFORE step t's FFN runs: previous steps' active sets and the input byte.
Tables / thresholds are fitted on CAL only; hit rate and waste are measured on EVAL.
  hit   = |pred ∩ actual| / |actual|   (share of needed columns already fetched)
  waste = |pred \\ actual| / |actual|  (extra columns fetched per needed column)
Go rule from the brief: hit >= 70 % and waste <= 1x.  Everything MEASURED from model activity."""
import numpy as np
from common import load, tags, save_csv, plt, OUT
rows = []
for tag in tags():
    C, E = load(tag, 'cal'), load(tag, 'eval'); L = E['L']
    Ac, A = C['ffn'] != 0, E['ffn'] != 0                       # (T, L, F) bool
    nA = A.sum()
    def score(name, P, extra=''):
        inter = (P & A).sum(); pred = P.sum()
        rows.append(dict(model=tag, predictor=name, hit=round(inter / nA, 4), waste=round((pred - inter) / nA, 4),
                         fetched_per_token=round(pred / len(A), 1), needed_per_token=round(nA / len(A), 1),
                         go=bool(inter / nA >= .7 and (pred - inter) / nA <= 1), note=extra)); print(rows[-1])
    jac = ((A[1:] & A[:-1]).sum((1, 2)) / np.maximum(1, (A[1:] | A[:-1]).sum((1, 2)))).mean()
    rows.append(dict(model=tag, predictor='(Jaccard A_t vs A_t-1)', hit=round(float(jac), 4), waste='', fetched_per_token='', needed_per_token='', go='', note='mean over tokens')); print(rows[-1])
    prev = np.concatenate([A[:1] * False, A[:-1]])
    score('last step', prev)
    u = prev.copy()
    for k in (2, 4, 8):
        while True:
            sh = np.concatenate([A[:k] * False, A[:-k]]); u = u | sh; break
        score(f'union of last {k} steps', u)
    # byte-conditional table fitted on CAL: P(neuron active | input byte)
    cnt = np.zeros((256,) + Ac.shape[1:]); np.add.at(cnt, C['inbyte'], Ac); tot = np.bincount(C['inbyte'], minlength=256)[:, None, None]
    pb = cnt / np.maximum(1, tot); glob = Ac.mean(0, keepdims=True)[0]
    pb = np.where(tot > 20, pb, glob)                            # back off to the global rate for rare bytes
    for tau in (0.1, 0.2, 0.3, 0.5):
        score(f'byte table P>={tau}', pb[E['inbyte']] >= tau, 'fitted on CAL')
    score('last step ∪ byte table P>=0.3', prev | (pb[E['inbyte']] >= 0.3), 'fitted on CAL')
    # learned linear predictor (fitted on CAL): active_t ~ W [state spikes of the same layer at t-1, one-hot input byte, 1];
    # threshold chosen on CAL as the one with the highest hit subject to waste <= 1
    hitL, predL = 0, 0
    for l in range(L):
        def feats(R):
            s = np.concatenate([R['state'][:1, l] * 0, R['state'][:-1, l]]).astype(np.float32)
            return np.concatenate([s, np.eye(256, dtype=np.float32)[R['inbyte']], np.ones((len(s), 1), np.float32)], 1)
        Xc, Xe = feats(C), feats(E); Yc = Ac[:, l].astype(np.float32)
        W = np.linalg.solve(Xc.T @ Xc + 10 * np.eye(Xc.shape[1], dtype=np.float32), Xc.T @ Yc); sc, se = Xc @ W, Xe @ W
        best = None
        for th in np.linspace(0.05, 0.6, 23):
            pc = sc >= th; i = (pc & Ac[:, l]).sum(); n = Ac[:, l].sum()
            if (pc.sum() - i) / n <= 1 and (best is None or i / n > best[1]): best = (th, i / n)
        pe = se >= best[0]; hitL += (pe & A[:, l]).sum(); predL += pe.sum()
    rows.append(dict(model=tag, predictor='linear(state spikes t-1, input byte), thr on CAL', hit=round(hitL / nA, 4), waste=round((predL - hitL) / nA, 4),
                     fetched_per_token=round(predL / len(A), 1), needed_per_token=round(nA / len(A), 1), go=bool(hitL / nA >= .7 and (predL - hitL) / nA <= 1), note='fitted on CAL')); print(rows[-1])
    # oracle-ish upper reference: previous layer's set at the SAME step (no lead time for layer 0; little lead time otherwise)
    lay = np.concatenate([A[:, :1] * False, A[:, :-1]], 1)
    score('same step, previous layer (reference: zero lead time)', lay)
save_csv('h2_prefetch.csv', rows)
P = plt(); f, ax = P.subplots(figsize=(6, 4))
for tag in {r['model'] for r in rows}:
    rr = [r for r in rows if r['model'] == tag and r['waste'] != '']
    ax.scatter([r['waste'] for r in rr], [r['hit'] for r in rr], label=tag)
    for r in rr: ax.annotate(r['predictor'][:22], (r['waste'], r['hit']), fontsize=5)
ax.axhline(.7, ls='--', c='g'); ax.axvline(1, ls='--', c='g'); ax.set_xlabel('waste (extra / needed)'); ax.set_ylabel('hit rate'); ax.set_xscale('symlog')
ax.set_title('H2 prefetch predictors (EVAL; go = top-left of green lines)'); ax.legend(); f.tight_layout(); f.savefig(f'{OUT}/h2_prefetch.png', dpi=110)
