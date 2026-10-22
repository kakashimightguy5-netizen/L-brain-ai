"""Fix test for H1: make the DENSE projections event-driven by spike-coding their inputs (post-training, NO retraining).
Input x of qkv / o / inp / f1 -> integer count q = clamp(round(x/theta), -8, 8), fed on as q*theta. Zero counts = skipped MACs.
Reports bits/byte and the share of input entries that are zero (= MAC saving on that matrix) for each matrix and theta.
If quality collapses without training, the number shows how much QAT must recover. EVAL = second half; nothing fitted. L-Edge."""
import os, sys, math, json, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path[:0] = [ROOT, HERE]
os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, fastmem
torch.set_num_threads(2)
m = M5.L5Model(d=128, n_layers=4, heads=4).eval(); m.load_state_dict(torch.load(os.path.join(ROOT, 'LEdge_s0.pt'), map_location='cpu'))
va = torch.tensor(list(open('/home/claude/data/py_big_val.txt', 'rb').read()), dtype=torch.long); half = len(va) // 2
X = torch.stack([va[half + i * 1024: half + i * 1024 + 1025] for i in range(6)])
targets = {'qkv': [fw.qkv for fw in m.fw], 'o': [fw.o for fw in m.fw], 'inp': [l.inp for l in m.layers], 'f1': [l.f1 for l in m.layers]}
zero = {}
def make(theta, key):
    def pre(mod, args):
        x = args[0]; q = torch.clamp(torch.round(x / theta), -8, 8); zero.setdefault(key, []).append((q == 0).float().mean().item()); return (q * theta,)
    return pre
@torch.no_grad()
def bpb():
    inp, y = X[:, :-1], X[:, 1:]; lp = m(inp, fastmem.hippo_batch(inp))[0].log_softmax(-1); return -lp.gather(-1, y[..., None]).mean().item() / math.log(2)
R = {'clean': round(bpb(), 4), 'rows': []}
print('clean', R['clean'], flush=True)
for theta in (0.25, 0.5, 1.0):
    for sel in (('f1',), ('inp',), ('qkv',), ('o',), ('f1', 'inp', 'qkv', 'o')):
        hs = []; zero.clear()
        for k in sel: hs += [mod.register_forward_pre_hook(make(theta, k)) for mod in targets[k]]
        b = bpb(); [h.remove() for h in hs]
        row = {'theta_in_rms': theta, 'matrices': '+'.join(sel), 'bits_per_byte': round(b, 4), 'delta_%': round(100 * (b / R['clean'] - 1), 2),
               'zero_input_share': {k: round(sum(v) / len(v), 3) for k, v in zero.items()}}
        R['rows'].append(row); print(row, flush=True)
json.dump(R, open(os.path.join(ROOT, 'results/hw_hypotheses/fix_spike_inputs.json'), 'w'), indent=1)
