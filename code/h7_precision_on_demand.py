"""H7 - precision on demand: run each byte with 4-bit OR 8-bit weights, chosen from the PREVIOUS byte's top-1 confidence
(confident -> 4-bit, unsure -> 8-bit). Exact sequential simulation: one recurrent Stepper whose weights are swapped per byte
(states carry over, as on hardware). Post-training quantisation of every nn.Linear (hwflags.FakeQuant.quant):
8-bit = per-channel int8 pow2 scale; 4-bit = mx4_pow2 (block 32, pow2 shift per block) [and per-channel int4 as a second 4-bit].
Thresholds are fitted on CAL (first half of the held-out file) to give target 4-bit shares; bits/byte measured on EVAL (second half).
Control: random choice at the SAME 4-bit share. MEASURED (model quality); weight-traffic saving MODELLED = share_4bit x 50 %.
Usage: python h7_precision_on_demand.py CKPT [N_WINDOWS=6] [W=1024]   (L-Edge: d128, 4 layers)"""
import os, sys, math, copy, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path[:0] = [ROOT, HERE]
OUT = os.environ.get('LOUT', os.path.join(ROOT, 'results/hw_hypotheses')); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import l5_model as M5, l6_step, hwflags
torch.set_num_threads(int(os.environ.get('THREADS', 2))); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = sys.argv[1]; NW = int(sys.argv[2]) if len(sys.argv) > 2 else 6; W = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
base = M5.L5Model(d=128, n_layers=4, heads=4).to(dev).eval(); base.load_state_dict(torch.load(ck if os.path.exists(ck) else os.path.join(OUT, ck), map_location=dev))
def quantised(mode):
    m = copy.deepcopy(base); fq = hwflags.FakeQuant(mode, 32)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear): mod.weight.copy_(fq.quant(mod.weight))
    return m
M = {'float': base, '8': quantised('int8'), '4': quantised('mx4_pow2'), '4c': quantised('int4')}
va = list(open(os.path.join(os.environ.get('LDATA', '/home/claude/data'), 'py_big_val.txt'), 'rb').read()); half = len(va) // 2
def run(start, policy, lo='4', rng=None):
    """policy(prev_conf) -> True for low precision. Returns per-byte bits, confidences, low-precision flags."""
    bits, confs, lowf = [], [], []
    for w in range(NW):
        st = l6_step.Stepper(M['8']); seg = va[start + w * W:start + w * W + W + 1]; prev = 0.0
        for t in range(W):
            low = policy(prev, rng); st.m = M[lo] if low is True else M[low] if isinstance(low, str) else M['8']
            lp = st.step([seg[t]])[0]; bits.append(-lp[seg[t + 1]].item() / math.log(2)); prev = lp.max().exp().item()
            confs.append(prev); lowf.append(low is True or low == lo)
    return np.array(bits), np.array(confs), np.array(lowf)
rows = []
def rec(name, b, low, note=''):
    rows.append(dict(policy=name, bits_per_byte=round(float(b.mean()), 4), share_low_precision=round(float(low.mean()), 4),
                     modelled_weight_traffic_vs_8bit=round(1 - 0.5 * float(low.mean()), 4), note=note)); print(rows[-1], flush=True)
# fixed-precision references on EVAL
for name, key in (('float32', 'float'), ('all 8-bit (int8 pc)', '8'), ('all 4-bit (mx4 pow2 B32)', '4'), ('all 4-bit (int4 pc)', '4c')):
    b, c, _ = run(half, lambda p, r, k=key: k); rec(name, b, np.full(len(b), key in ('4', '4c')))
# thresholds from CAL: confidences of the 8-bit model on CAL
_, cc, _ = run(0, lambda p, r: False)
prev_cal = np.concatenate([[0.0], cc[:-1]])
for share in (0.25, 0.5, 0.75):
    thr = float(np.quantile(prev_cal, 1 - share))
    for lo in ('4', '4c'):
        b, _, low = run(half, lambda p, r, th=thr: p >= th, lo); rec(f'confidence-gated ({ "mx4" if lo == "4" else "int4"}), target {share:.0%} low', b, low, f'thr={thr:.4f} fitted on CAL')
        rng = np.random.default_rng(0)
        b, _, low = run(half, lambda p, r, s=share: bool(r.random() < s), lo, rng); rec(f'random control ({ "mx4" if lo == "4" else "int4"}), {share:.0%} low', b, low)
json.dump(rows, open(os.path.join(OUT, 'h7_precision_on_demand.json'), 'w'), indent=1); print('done')
