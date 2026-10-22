"""Post-training quantisation of the TRAINED baseline, with and without a Hadamard input rotation (QuaRot-style, MambaQuant/QuaRot motivation).
y = W x = (W H)(H x) for orthonormal symmetric H (exact in fp). Rotation only on the 128-wide-input matrices fed by normed activations
(inp, qkv, f1); out/f2 take spike counts and are NOT rotated (rotation would destroy integer spikes). MEASURED bits/byte on the fix_eval windows."""
import os, sys, math, json, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem, hwflags
torch.set_num_threads(int(os.environ.get('THREADS', 2))); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.environ.get('LDATA', '/home/claude/data'); va = torch.tensor(list(open(os.path.join(DATA, 'py_big_val.txt'), 'rb').read()), dtype=torch.long); half = len(va) // 2
X = torch.stack([va[half + i * 1024: half + i * 1024 + 1025] for i in range(6)]).to(DEV)
def had(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)
class QLin(nn.Module):
    def __init__(s, lin, mode, rot):
        super().__init__(); W = lin.weight.data.float(); s.b = None if lin.bias is None else lin.bias.data.clone(); s.rot = rot and W.shape[1] == 128
        s.H = had(128).to(W.device) if s.rot else None
        if s.rot: W = W @ s.H
        s.W = hwflags.FakeQuant(mode, 32).quant(W) if mode != 'none' else W; s.in_features, s.out_features = lin.in_features, lin.out_features
    def forward(s, x):
        if s.rot: x = x @ s.H
        return F.linear(x, s.W, s.b)
def load(): m = M5.L5Model(d=128, n_layers=4, heads=4); m.load_state_dict(torch.load(next(p for p in (os.path.join(os.environ.get('LOUT', '.'), 'LEdge_s0.pt'), os.path.join(HERE, 'LEdge_s0.pt')) if os.path.exists(p)), map_location='cpu')); return m.to(DEV).eval()
def bpb(m):
    with torch.no_grad():
        inp, y = X[:, :-1], X[:, 1:]; lp = m(inp, fastmem.hippo_batch(inp))[0].log_softmax(-1); return -lp.gather(-1, y[..., None]).mean().item() / math.log(2)
base = bpb(load()); res = {'baseline': base}; print('baseline', round(base, 4), flush=True)
for mode in ('int8', 'int4', 'ternary'):
    for rot in (False, True):
        m = load()
        for parent in list(m.modules()):
            for nm in hwflags.ALL_NAMES:
                lin = getattr(parent, nm, None)
                if isinstance(lin, nn.Linear): setattr(parent, nm, QLin(lin, mode, rot))
        b = bpb(m); k = f'{mode}_rot{int(rot)}'; res[k] = b; print(k, round(b, 4), f'{(b/base-1)*100:+.2f}%', flush=True)
OUT = os.environ.get('LOUT', 'results'); os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, 'ptq_rot.json'), 'w'))
