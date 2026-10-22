"""Evaluate the 'big-matrix' fixes on the SAME EVAL windows as the baseline (L-Edge s0, 6 x 1024 bytes of the second half).
Variants (cE_* checkpoints in results/fix): tlr25 = ternary + low-rank 0.25, tern, lr25, dspk05 (dense inputs spike-coded), dspk05_tern.
MODELLED per-byte MACs split into multiplies and additions; weight storage bits; zero-input share per dense matrix (dense-spike variants).
ESTIMATED chip energy with 45 nm op energies (Horowitz 2014): mult 0.2 pJ, add 0.03 pJ, 8-bit weight read 1.25 pJ/byte (scaled by bit width)."""
import os, sys, math, json, torch, torch.nn as nn
import torch.nn.utils.parametrize as P
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path[:0] = [ROOT, HERE]
os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem, hwflags
torch.set_num_threads(int(os.environ.get('THREADS', 1)))
va = torch.tensor(list(open('/home/claude/data/py_big_val.txt', 'rb').read()), dtype=torch.long); half = len(va) // 2
X = torch.stack([va[half + i * 1024: half + i * 1024 + 1025] for i in range(6)])
E_MUL, E_ADD, E_B = 0.2e-12, 0.03e-12, 1.25e-12
VAR = {'baseline (LEdge_s0)': ('LEdge_s0.pt', 'none', 0, 0), 'ternary+lowrank0.25': ('results/fix/cE_tlr25_s0.pt', 'ternary', 0.25, 0), 'ternary': ('results/fix/cE_tern_s0.pt', 'ternary', 0, 0),
       'lowrank0.25': ('results/fix/cE_lr25_s0.pt', 'none', 0.25, 0), 'dense-spike 0.5': ('results/fix/cE_dspk05_s0.pt', 'none', 0, 0.5), 'dense-spike 0.5 + ternary': ('results/fix/cE_dspk05_tern_s0.pt', 'ternary', 0, 0.5)}
BITS = {'none': 8, 'ternary': 2}
def build(q, lr, ds):
    m = M5.L5Model(d=128, n_layers=4, heads=4)
    if lr > 0: hwflags.apply_lowrank(m, lr)
    if ds > 0: hwflags.apply_dense_spikes(m, ds)
    if q != 'none': hwflags.apply_qat(m, q, 32)
    return m
out = {}
for name, (ck, q, lr, ds) in VAR.items():
    p = os.path.join(ROOT, ck)
    if not os.path.exists(p): continue
    m = build(q, lr, ds); m.load_state_dict(torch.load(p, map_location='cpu')); m.eval()
    spikes = []; orig = M0.spikes
    def spy(z, lo, hi):
        c = orig(z, lo, hi); spikes.append((c != 0).float().mean().item()); return c
    M0.spikes = spy
    with torch.no_grad():
        inp, y = X[:, :-1], X[:, 1:]; lp = m(inp, fastmem.hippo_batch(inp))[0].log_softmax(-1); bpb = -lp.gather(-1, y[..., None]).mean().item() / math.log(2)
    M0.spikes = orig
    n = len(m.layers); st = sum(spikes[0::2][:n]) / n if ds == 0 else None
    # spikes list order with dense spikes interleaves; take robustly from layer attributes instead
    rs = sum(float(l.last_rate[0]) for l in m.layers) / n; rf = sum(float(l.last_rate[1]) for l in m.layers) / n
    zero = {}
    if ds > 0:
        for w in m._dense_spk: zero.setdefault(id(w), w.zero)
    mac_mul = mac_add = 0.0; wbits = 0
    for nm, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            o, i = mod.weight.shape; parent = nm.rsplit('.', 1)[0]; leaf = nm.split('.')[-1]
            frac = 1.0
            if nm.endswith('.out') or nm.endswith('.lin') and False: frac = rs
            if nm.split('.')[-1] == 'out': frac = rs
            elif nm.split('.')[-1] == 'f2': frac = rf
            # dense-spike wrapper: the Linear sits at <name>.lin ; zero share known from the wrapper
            if nm.endswith('.lin'):
                w = m.get_submodule(nm[:-4]); frac = 1 - w.zero
            if q == 'ternary': mac_add += frac * o * i
            else: mac_mul += frac * o * i
            wbits += o * i * BITS[q]
    joules = mac_mul * (E_MUL + E_ADD) + mac_add * E_ADD + (mac_mul + mac_add) * BITS[q] / 8 * E_B
    out[name] = {'bits_per_byte': round(bpb, 4), 'delta_vs_baseline_%': None, 'multiplies_per_byte': int(mac_mul), 'additions_only_per_byte': int(mac_add),
                 'weight_storage_KB': round(wbits / 8 / 1024, 1), 'chip_nJ_per_byte (ESTIMATED)': round(joules * 1e9, 1),
                 'dense_input_zero_share': {k: round(v, 3) for k, v in zero.items()} if zero else None, 'state_firing': round(rs, 3), 'ffn_firing': round(rf, 3)}
    print(name, out[name], flush=True)
b = out['baseline (LEdge_s0)']['bits_per_byte']
for v in out.values(): v['delta_vs_baseline_%'] = round(100 * (v['bits_per_byte'] / b - 1), 2)
json.dump(out, open(os.path.join(ROOT, 'results/hw_hypotheses/fix_eval.json'), 'w'), indent=1)
