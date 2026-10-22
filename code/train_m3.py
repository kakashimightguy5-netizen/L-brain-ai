"""M3 experiment: L with domain-routed experts on a 3-domain stream (Python / JavaScript / C).
Modes (same data, steps, LR; FFN total width equal for dense/token/segment):
  dense        : normal spiking FFN (all of it active)                       - quality reference
  dense_small  : FFN of width total/E (same ACTIVE compute as one expert)     - equal-active-compute reference
  token        : E experts, top-1 per byte (Switch-style token MoE baseline)
  segment      : E experts, one per 64-byte segment, chosen from the previous segment (ours)
Usage: python train_m3.py MODE SEED [STEPS]  -> results/m3_<MODE>_s<SEED>.json"""
import os, sys, time, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
OUT = os.environ.get('LOUT', os.path.join(os.path.dirname(HERE), 'results')); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import l5_model as M5, fastmem, experts as EX
mode, seed = sys.argv[1], int(sys.argv[2]); STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000; E = 4
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(seed); T, BS = 256, 16
DATA = os.environ.get('LDATA', '/home/claude/data')
ld = lambda f: torch.tensor(list(open(os.path.join(DATA, f), 'rb').read()), dtype=torch.long)
TR = [ld('py_big_train.txt'), ld('js_train.txt'), ld('c_train.txt')]; VA = [ld('py_big_val.txt'), ld('js_val.txt'), ld('c_val.txt')]
NAMES = ['Python', 'JavaScript', 'C']
def batch(srcs, g=None, dom=None):
    d = torch.randint(0, 3, (BS,), generator=g) if dom is None else torch.full((BS,), dom)
    xs = [srcs[int(k)][i:i + T + 1] for k, i in zip(d, [int(torch.randint(0, len(srcs[int(k)]) - T - 1, (1,), generator=g)) for k in d])]
    return torch.stack(xs).to(dev), d
m = M5.L5Model(d=144, n_layers=4, heads=4).to(dev)
W = m.layers[0].f1.out_features
if mode == 'dense_small':
    for l in m.layers:
        l.f1 = nn.Linear(144, W // E).to(dev); l.f2 = nn.Linear(W // E, 144).to(dev); l.theta2 = nn.Parameter(torch.full((W // E,), 0.5, device=dev))
elif mode in ('token', 'segment'): EX.install(m, E, mode, W)
opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS, pct_start=0.05)
t0 = time.time()
for it in range(STEPS):
    m.train(); x, _ = batch(TR); inp, y = x[:, :-1], x[:, 1:]
    lg = m(inp, fastmem.hippo_batch(inp))[0]; loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)) + 0.03 * m.rate_cost()
    if mode in ('token', 'segment'): loss = loss + 0.01 * sum(l.moe.aux for l in m.layers)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    if it % 250 == 0: print(mode, seed, it, round(loss.item(), 3), f'{time.time() - t0:.0f}s', flush=True)
R = {'mode': mode, 'seed': seed, 'experts': E, 'params_total': sum(p.numel() for p in m.parameters()), 'minutes': round((time.time() - t0) / 60, 1)}
ffn_total = sum(p.numel() for l in m.layers for n, p in l.named_parameters() if n.startswith(('f1', 'f2', 'theta2', 'moe.experts')))
R['ffn_params_total'] = ffn_total; R['ffn_params_active_per_byte'] = ffn_total // E if mode in ('token', 'segment') else ffn_total
m.eval(); per = {}; choices = {i: [] for i in range(len(m.layers))}; labels = []
with torch.no_grad():
    for dom in range(3):
        g = torch.Generator().manual_seed(100 + dom); tot = 0
        for _ in range(12):
            x, _ = batch(VA, g, dom); inp, y = x[:, :-1], x[:, 1:]
            lg = m(inp, fastmem.hippo_batch(inp))[0]; tot += F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).item()
            if mode in ('token', 'segment'):
                for i, l in enumerate(m.layers): choices[i].append(l.moe.last_choice.cpu())
                labels.append(torch.full(y.shape, dom))
        per[NAMES[dom]] = round(tot / 12 / math.log(2), 4)
R['bits_per_byte'] = per; R['bits_per_byte_mean'] = round(float(np.mean(list(per.values()))), 4)
if mode in ('token', 'segment'):
    lab = torch.cat(labels).reshape(-1).numpy(); R['routing'] = {}
    def mi(a, b, na, nb):
        j = np.zeros((na, nb)); np.add.at(j, (a, b), 1); j /= j.sum(); pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
        nz = j > 0; return float((j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum()), j
    Hd = math.log(3)
    for i in range(len(m.layers)):
        c = torch.cat(choices[i]); sw = (c[:, 1:] != c[:, :-1]).float().mean().item() * 1000
        I, J = mi(c.reshape(-1).numpy(), lab, E, 3)
        R['routing'][f'layer{i}'] = {'MI(expert;domain)/H(domain)': round(I / Hd, 4), 'switches_per_1000_bytes': round(sw, 1),
                                    'confusion [expert x domain] (share of bytes)': np.round(J, 4).tolist()}
json.dump(R, open(os.path.join(OUT, f'm3_{mode}_s{seed}.json'), 'w'), indent=1); print(json.dumps(R, indent=1))
