"""Per-matrix-group sensitivity of the TRAINED baseline to low-bit weights (no retraining). One group quantised at a time, rest fp32.
MEASURED bits/byte on the fix_eval windows. Tells which matrices must stay 8-bit in a mixed-precision model."""
import os, sys, math, json, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem, hwflags
torch.set_num_threads(int(os.environ.get('THREADS', 2))); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.environ.get('LDATA', '/home/claude/data'); va = torch.tensor(list(open(os.path.join(DATA, 'py_big_val.txt'), 'rb').read()), dtype=torch.long); half = len(va) // 2
X = torch.stack([va[half + i * 1024: half + i * 1024 + 1025] for i in range(6)]).to(DEV)
class QLin(nn.Module):
    def __init__(s, lin, mode):
        super().__init__(); s.W = hwflags.FakeQuant(mode, 32).quant(lin.weight.data.float()); s.b = None if lin.bias is None else lin.bias.data.clone(); s.in_features, s.out_features = lin.in_features, lin.out_features
    def forward(s, x): return F.linear(x, s.W, s.b)
def load(): m = M5.L5Model(d=128, n_layers=4, heads=4); m.load_state_dict(torch.load(next(p for p in (os.path.join(os.environ.get('LOUT', '.'), 'LEdge_s0.pt'), os.path.join(HERE, 'LEdge_s0.pt')) if os.path.exists(p)), map_location='cpu')); return m.to(DEV).eval()
def bpb(m):
    with torch.no_grad():
        inp, y = X[:, :-1], X[:, 1:]; lp = m(inp, fastmem.hippo_batch(inp))[0].log_softmax(-1); return -lp.gather(-1, y[..., None]).mean().item() / math.log(2)
def quant(m, plan):   # plan: {group name: mode}
    for parent in list(m.modules()):
        for nm, mode in plan.items():
            lin = getattr(parent, nm, None)
            if isinstance(lin, nn.Linear): setattr(parent, nm, QLin(lin, mode))
    return m
base = bpb(load()); res = {'baseline': base}; print('baseline', round(base, 4), flush=True)
for g in hwflags.ALL_NAMES:
    for mode in ('int4', 'ternary'):
        b = bpb(quant(load(), {g: mode})); res[f'{g}_{mode}'] = b; print(g, mode, round(b, 4), f'{(b/base-1)*100:+.2f}%', flush=True)
for name, plan in {'tern_inp_f1_int8_rest': {'inp': 'ternary', 'f1': 'ternary', 'out': 'int8', 'f2': 'int8', 'qkv': 'int8', 'o': 'int8'},
                   'int4_all': {g: 'int4' for g in hwflags.ALL_NAMES}, 'int4_all_int8_out_f2': {'inp': 'int4', 'f1': 'int4', 'qkv': 'int4', 'o': 'int4', 'out': 'int8', 'f2': 'int8'}}.items():
    b = bpb(quant(load(), plan)); res[name] = b; print(name, round(b, 4), f'{(b/base-1)*100:+.2f}%', flush=True)
OUT = os.environ.get('LOUT', 'results'); os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, 'ptq_sens.json'), 'w'))
