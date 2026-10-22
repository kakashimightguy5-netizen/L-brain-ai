"""Backprop-free training test (vault 132). Same task as gate_lm.py dense: next byte from K prior bytes (bits in).
Methods: bp (backprop reference) | dfa (direct feedback alignment: fixed random feedback of output error to each hidden layer, no weight transport, no backward pass through W)
 | local (each layer has its own fixed random readout + local loss; no gradient crosses layers) | hebb_ro (random frozen hidden layers, only readout trained by delta rule)
Usage: python local_learn.py METHOD H [K] [STEPS] [SEED]. MEASURED bits/byte on held-out val."""
import os, sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
M, H = sys.argv[1], int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 8; STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 4000; SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
torch.manual_seed(SEED); torch.set_num_threads(1); DATA = '/home/claude/data'; BS = 256; LR = float(os.environ.get('LR', 2e-3))
tr = torch.frombuffer(bytearray(open(os.path.join(DATA, 'py_big_train.txt'), 'rb').read(16_000_000)), dtype=torch.uint8).long()
va = torch.frombuffer(bytearray(open(os.path.join(DATA, 'py_big_val.txt'), 'rb').read()), dtype=torch.uint8).long(); va = va[len(va) // 2: len(va) // 2 + 200_000]
SH = torch.arange(8); bits = lambda x: ((x[..., None] >> SH) & 1).float().flatten(-2)
def batch(src, B):
    ix = torch.randint(0, len(src) - K - 1, (B,)); w = torch.stack([src[i:i + K + 1] for i in ix]); return bits(w[:, :K]), w[:, K]
L1, L2, LO = nn.Linear(K * 8, H), nn.Linear(H, H), nn.Linear(H, 256)
B1, B2 = torch.randn(256, H) / 16, torch.randn(256, H) / 16     # fixed random feedback (DFA)
R1, R2 = torch.randn(H, 256) / math.sqrt(H), torch.randn(H, 256) / math.sqrt(H)   # fixed random local readouts
if M == 'local': ps = list(L1.parameters()) + list(L2.parameters()) + list(LO.parameters())
elif M == 'hebb_ro': ps = list(LO.parameters())
else: ps = list(L1.parameters()) + list(L2.parameters()) + list(LO.parameters())
opt = torch.optim.Adam(ps, LR); t0 = time.time()
def fwd(x): h1 = F.relu(L1(x)); h2 = F.relu(L2(h1)); return h1, h2, LO(h2)
@torch.no_grad()
def bpb():
    tot = 0.0; n = 0
    for i in range(0, 40_000, 2000):
        w = torch.stack([va[j:j + K + 1] for j in range(i, i + 2000)]); tot += F.cross_entropy(fwd(bits(w[:, :K]))[2], w[:, K], reduction='sum').item(); n += 2000
    return tot / n / math.log(2)
for it in range(STEPS):
    x, y = batch(tr, BS); opt.zero_grad()
    if M == 'bp': F.cross_entropy(fwd(x)[2], y).backward()
    elif M == 'hebb_ro':
        with torch.no_grad(): h1 = F.relu(L1(x)); h2 = F.relu(L2(h1))
        F.cross_entropy(LO(h2), y).backward()
    elif M == 'dfa':
        h1p = L1(x); h1 = F.relu(h1p); h2p = L2(h1.detach()); h2 = F.relu(h2p); out = LO(h2.detach())
        e = (out.softmax(-1) - F.one_hot(y, 256).float()).detach() / BS
        F.cross_entropy(out, y).backward()                                   # trains LO from its own local error
        h2p.backward((e @ B2) * (h2p > 0).float().detach() * 1.0, retain_graph=True) if False else None
        (h2 * ((e @ B2) * (h2p > 0).float()).detach()).sum().backward()      # layer 2: random-feedback error x local activity
        (h1 * ((e @ B1) * (h1p > 0).float()).detach()).sum().backward()      # layer 1
    elif M == 'pc':   # predictive coding (Whittington & Bogacz 2017 style): value nodes relax to minimise local prediction errors; weight updates use only local error x pre-activity
        with torch.no_grad():
            a0 = x; p1 = L1(a0); x1 = F.relu(p1); p2 = L2(x1); x2 = F.relu(p2); out = LO(x2)
            v1, v2 = x1.clone(), x2.clone(); IT = int(os.environ.get('PC_IT', 20)); g = float(os.environ.get('PC_LR', 0.1))
            for _ in range(IT):
                p1 = L1(a0); p2 = L2(v1); out = LO(v2)
                e1 = v1 - F.relu(p1); e2 = v2 - F.relu(p2); eo = (out.softmax(-1) - F.one_hot(y, 256).float()) / BS   # output error, weak nudge (scaled 1/BS) keeps PC close to the exact gradient
                d2 = eo @ LO.weight                                   # top-down signal to layer 2 through the SAME synapses (symmetric weights)
                d1 = (e2 * (p2 > 0).float()) @ L2.weight
                v2 = v2 - g * (e2 + d2); v1 = v1 - g * (e1 - d1)
            p1 = L1(a0); p2 = L2(v1); out = LO(v2); e1 = v1 - F.relu(p1); e2 = v2 - F.relu(p2); eo = (out.softmax(-1) - F.one_hot(y, 256).float()) / BS
            # local Hebbian-with-error updates (grad = -(post error x f') outer pre)
            gw = [(-(e1 * (p1 > 0).float())).t() @ a0, (-(e2 * (p2 > 0).float())).t() @ v1, eo.t() @ v2]
            gb = [(-(e1 * (p1 > 0).float())).sum(0), (-(e2 * (p2 > 0).float())).sum(0), eo.sum(0)]
        for lyr, w_, b_ in zip((L1, L2, LO), gw, gb): lyr.weight.grad = w_; lyr.bias.grad = b_
        if os.environ.get('PC_DBG') and it % 300 == 0:
            gb_ = torch.autograd.grad(F.cross_entropy(fwd(x)[2], y), [L1.weight, L2.weight, LO.weight]); print(it, [round(F.cosine_similarity(a.flatten(), b.flatten(), 0).item(), 3) for a, b in zip(gw, gb_)], [round((a.norm() / b.norm()).item(), 3) for a, b in zip(gw, gb_)], flush=True)
    elif M == 'local':
        h1 = F.relu(L1(x)); l1 = F.cross_entropy(h1 @ R1, y); h2 = F.relu(L2(h1.detach())); l2 = F.cross_entropy(h2 @ R2, y); l3 = F.cross_entropy(LO(h2.detach()), y); (l1 + l2 + l3).backward()
    opt.step()
R = {'method': M, 'H': H, 'K': K, 'steps': STEPS, 'seed': SEED, 'bpb': round(bpb(), 4), 'secs': round(time.time() - t0)}; print(json.dumps(R), flush=True)
os.makedirs('/home/claude/ll', exist_ok=True); json.dump(R, open(f'/home/claude/ll/ll_{M}_{H}_s{SEED}.json', 'w'))
