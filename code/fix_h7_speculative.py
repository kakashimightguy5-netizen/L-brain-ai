"""Fix for H7: decide the precision from the CURRENT byte, not the previous one ('speculative 4-bit').
Every byte first runs with 4-bit (mx4 pow2 B32) weights. If its top-1 confidence >= thr the 4-bit answer is kept (weights read once at 4 bit);
otherwise the neural states are rolled back and the byte is re-run at 8 bit (cost = 4-bit pass + 8-bit pass). The hash memories (hippocampus)
are updated once, as in the normal Stepper. Weight-traffic model (MODELLED, bytes relative to an all-8-bit run):
   easy byte: 0.5     redo byte: 0.5 + 1.0 = 1.5
thr fitted on CAL (first half) for target redo shares; bits/byte measured on EVAL (second half). L-Edge, post-training quantisation, 1 checkpoint."""
import os, sys, math, copy, json, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path[:0] = [ROOT, HERE]
os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l6_step, hwflags
torch.set_num_threads(int(os.environ.get('THREADS', 1))); NW = int(os.environ.get('NW', 6)); W = 1024
base = M5.L5Model(d=128, n_layers=4, heads=4).eval(); base.load_state_dict(torch.load(os.path.join(ROOT, 'LEdge_s0.pt'), map_location='cpu'))
def quantised(mode):
    m = copy.deepcopy(base); fq = hwflags.FakeQuant(mode, 32)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear): mod.weight.copy_(fq.quant(mod.weight))
    return m
M8, M4 = quantised('int8'), quantised('mx4_pow2')
va = list(open('/home/claude/data/py_big_val.txt', 'rb').read()); half = len(va) // 2
def snap(st): return [[t.clone() for t in L if t is not None] for L in (st.convbuf, st.S, st.h)]
def restore(st, sn):
    for L, saved in zip((st.convbuf, st.S, st.h), sn):
        for t, v in zip([t for t in L if t is not None], saved): t.copy_(v)
def run(start, thr):
    """thr=None: always 8-bit; thr=-1: always 4-bit; else speculative. Returns bits, conf4 (always computed when thr is not None), redo flags"""
    bits, conf4, redo = [], [], []
    for w in range(NW):
        st = l6_step.Stepper(M8); seg = va[start + w * W: start + w * W + W + 1]
        for t in range(W):
            b = seg[t]; st.seq[0].append(b); r, tr, hit, lay = st._recall(0, len(st.seq[0]) - 1)
            ints = torch.tensor([[b]] + [[c] for c in r] + [[tr], [hit]] + [[c] for c in lay], dtype=torch.long)
            if thr is None: st.m = M8; lp = st.neural(ints)[0]; rd = False; c4 = float('nan')
            else:
                sn = snap(st); st.m = M4; lp = st.neural(ints)[0]; c4 = lp.max().exp().item(); rd = c4 < thr
                if rd: restore(st, sn); st.m = M8; lp = st.neural(ints)[0]
            bits.append(-lp[seg[t + 1]].item() / math.log(2)); conf4.append(c4); redo.append(rd)
    return np.array(bits), np.array(conf4), np.array(redo)
R = {}
b8, _, _ = run(half, None); R['all 8-bit'] = {'bits_per_byte': round(float(b8.mean()), 4), 'redo_share': 0.0, 'weight_traffic_vs_8bit': 1.0}; print(R['all 8-bit'], flush=True)
b4, c4e, _ = run(half, -1.0); R['all 4-bit (mx4 pow2)'] = {'bits_per_byte': round(float(b4.mean()), 4), 'redo_share': 0.0, 'weight_traffic_vs_8bit': 0.5}; print(R['all 4-bit (mx4 pow2)'], flush=True)
_, c4cal, _ = run(0, -1.0)                                       # CAL: 4-bit confidences only (thresholds fitted here)
for share in (0.1, 0.2, 0.3, 0.5):
    thr = float(np.quantile(c4cal, share)); b, c, rd = run(half, thr)
    R[f'speculative, target redo {share:.0%}'] = {'thr_fitted_on_CAL': round(thr, 4), 'bits_per_byte': round(float(b.mean()), 4), 'redo_share': round(float(rd.mean()), 4),
                                                  'weight_traffic_vs_8bit': round(float(0.5 + rd.mean()), 4), 'delta_vs_8bit_%': round(100 * (b.mean() / b8.mean() - 1), 2)}
    print(f'speculative {share:.0%}', R[f'speculative, target redo {share:.0%}'], flush=True)
json.dump(R, open(os.path.join(ROOT, 'results/hw_hypotheses/fix_h7_speculative.json'), 'w'), indent=1); print('done')
