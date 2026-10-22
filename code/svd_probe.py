"""Quality-preserving weight reduction probe: SVD-factorise the TRAINED baseline (no retraining) at several ranks, eval on the fix_eval windows.
Tells how much rank the trained matrices really need. MEASURED bits/byte; params counted exactly."""
import os, sys, math, json, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE; sys.path[:0] = [ROOT]
os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem, hwflags
va = torch.tensor(list(open(os.path.join(os.environ.get('LDATA','/home/claude/data'),'py_big_val.txt'), 'rb').read()), dtype=torch.long); half = len(va) // 2
X = torch.stack([va[half + i * 1024: half + i * 1024 + 1025] for i in range(6)])
def bpb(m):
    m.eval()
    with torch.no_grad():
        inp, y = X[:, :-1], X[:, 1:]; lp = m(inp, fastmem.hippo_batch(inp))[0].log_softmax(-1); return -lp.gather(-1, y[..., None]).mean().item() / math.log(2)
def load():
    m = M5.L5Model(d=128, n_layers=4, heads=4); m.load_state_dict(torch.load(os.path.join(os.environ.get('LOUT','.'),'LEdge_s0.pt'), map_location='cpu')); return m
base = bpb(load()); print('baseline', round(base, 4), M0.nparams(load()))
res = {'baseline': base}
for names, tag in [(hwflags.DENSE_NAMES, 'dense4'), (hwflags.ALL_NAMES, 'all6')]:
    for frac in (0.75, 0.5, 0.375, 0.25):
        m = load()
        for parent in list(m.modules()):
            for nm in names:
                lin = getattr(parent, nm, None)
                if isinstance(lin, nn.Linear):
                    o, i = lin.weight.shape; r = max(4, round(frac * min(o, i)))
                    U, S, Vh = torch.linalg.svd(lin.weight.data, full_matrices=False)
                    A = nn.Linear(i, r, bias=False); B = nn.Linear(r, o, bias=lin.bias is not None)
                    A.weight.data = (Vh[:r] * S[:r, None].sqrt()); B.weight.data = (U[:, :r] * S[:r].sqrt())
                    if lin.bias is not None: B.bias.data = lin.bias.data.clone()
                    setattr(parent, nm, nn.Sequential(A, B))
        b = bpb(m); res[f'{tag}_{frac}'] = b; print(tag, frac, round(b, 4), f'{(b/base-1)*100:+.1f}%', M0.nparams(m), flush=True)
json.dump(res, open(os.path.join(os.environ.get('LOUT','results'),'svd_probe.json'), 'w'))
