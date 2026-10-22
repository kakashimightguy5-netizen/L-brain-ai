"""Why does H1 fail? Three diagnostics on EVAL traces (nothing fitted).
 (a) structural ceiling: share of per-byte MACs that CAN be event-driven at all (out, f2) vs dense (inp, f1, qkv, o)
 (b) what drives spike-event count: input byte identity vs difficulty (variance explained, partial correlation with loss)
 (c) per-layer spike rates for easy vs hard bytes (is there ANY difficulty signal in the spikes?)"""
import numpy as np, json
from scipy.stats import spearmanr
from common import load, tags, OUT
res = {}
for tag in tags():
    E = load(tag, 'eval'); D, F, L = E['D'], E['F'], E['L']
    parts = {'qkv (dense)': 3 * D * D, 'o (dense)': D * D, 'inp (dense)': 2 * D * D, 'f1 (dense)': D * F, 'out (event-capable)': D * D, 'f2 (event-capable)': F * D}
    tot = sum(parts.values()); cap = (parts['out (event-capable)'] + parts['f2 (event-capable)']) / tot
    st, ff = (E['state'] != 0).sum(2), (E['ffn'] != 0).sum(2); ev = st.sum(1) + ff.sum(1); loss = E['loss']; byte = E['byte']
    ib = E['inbyte']; mu = np.bincount(ib, weights=ev, minlength=256) / np.maximum(1, np.bincount(ib, minlength=256))
    r2 = 1 - ((ev - mu[ib]) ** 2).sum() / ((ev - ev.mean()) ** 2).sum()
    resid = ev - mu[ib]; rho_resid = spearmanr(resid, loss).correlation
    easy, hard = loss <= np.quantile(loss, .5), loss >= np.quantile(loss, .8)
    per_layer = {f'L{l}': {'state_easy': round(float((E['state'][easy, l] != 0).mean()), 4), 'state_hard': round(float((E['state'][hard, l] != 0).mean()), 4),
                           'ffn_easy': round(float((E['ffn'][easy, l] != 0).mean()), 4), 'ffn_hard': round(float((E['ffn'][hard, l] != 0).mean()), 4)} for l in range(L)}
    # dense share of per-layer parts (percent of one layer's MACs)
    res[tag] = {'MAC_share_per_layer_%': {k: round(100 * v / tot, 1) for k, v in parts.items()}, 'max_possible_saving_if_event_layers_were_100%_idle': round(cap, 3),
                'actual_event_unit_saving (1 - fired share)': round(1 - (ev.mean() * D) / (L * (parts['out (event-capable)'] + parts['f2 (event-capable)'])), 3),
                'variance_of_events_explained_by_input_byte_R2': round(float(r2), 3), 'spearman(events residual after byte, loss)': round(float(rho_resid), 3),
                'spearman(events, loss) raw': round(float(spearmanr(ev, loss).correlation), 3), 'per_layer_firing_easy_vs_hard': per_layer}
    print(tag, json.dumps(res[tag], indent=1)[:1800])
json.dump(res, open(f'{OUT}/diag_h1.json', 'w'), indent=1)
