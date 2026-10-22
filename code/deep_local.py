"""Depth test for layer-local learning with LEARNED local readouts (vault note 132). Next byte from K prior bytes; MLP with D hidden layers of width H.
bp     : ordinary backprop through all layers
local  : every hidden layer has its own learned readout (Linear H->256) and learns only from that local loss; gradients never cross a layer
         (like each cortical area predicting the next input with its own output pathway). Deployed model = last layer's readout.
Usage: python deep_local.py METHOD D H STEPS SEED   -> MEASURED bits/byte on held-out val (also reports each layer's local readout)."""
import os, sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
M, D, H, STEPS, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); K = 8
torch.manual_seed(SEED); torch.set_num_threads(1); DATA = '/home/claude/data'; BS = 256
tr = torch.frombuffer(bytearray(open(os.path.join(DATA, 'py_big_train.txt'), 'rb').read(16_000_000)), dtype=torch.uint8).long()
va = torch.frombuffer(bytearray(open(os.path.join(DATA, 'py_big_val.txt'), 'rb').read()), dtype=torch.uint8).long(); va = va[len(va) // 2: len(va) // 2 + 200_000]
SH = torch.arange(8); bits = lambda x: ((x[..., None] >> SH) & 1).float().flatten(-2)
def batch(src, B):
    ix = torch.randint(0, len(src) - K - 1, (B,)); w = torch.stack([src[i:i + K + 1] for i in ix]); return bits(w[:, :K]), w[:, K]
layers = nn.ModuleList([nn.Linear(K * 8 if i == 0 else H, H) for i in range(D)]); heads = nn.ModuleList([nn.Linear(H, 256) for _ in range(D)])
opt = torch.optim.Adam(list(layers.parameters()) + list(heads.parameters()), float(os.environ.get('LR', 2e-3))); t0 = time.time()
RES = os.environ.get('RES') == '1'
BLK = int(os.environ.get('BLOCK', 1))   # BLOCK flag: gradient flows inside blocks of BLK layers (local microcircuits), detached between blocks
def forward(x, detach):
    hs = []
    for i, l in enumerate(layers):
        xi = x.detach() if (detach and i % BLK == 0) else x; h = F.relu(l(xi)); x = h + xi if (RES and i > 0) else h; hs.append(x)   # RES flag: residual connections
    return hs
@torch.no_grad()
def bpb():
    tot = [0.0] * D; n = 0
    for i in range(0, 40_000, 2000):
        w = torch.stack([va[j:j + K + 1] for j in range(i, i + 2000)]); hs = forward(bits(w[:, :K]), False)
        for k in range(D): tot[k] += F.cross_entropy(heads[k](hs[k]), w[:, K], reduction='sum').item()
        n += 2000
    return [t / n / math.log(2) for t in tot]
for it in range(STEPS):
    x, y = batch(tr, BS); opt.zero_grad()
    if M == 'bp': F.cross_entropy(heads[-1](forward(x, False)[-1]), y).backward()
    elif M == 'overlap':   # each layer's local loss reaches back through TWO layers (overlapping 2-layer windows): gradient crosses at most one synapse boundary
        hs_ = forward(x, True); ins = [x] + hs_[:-1]; loss = 0
        for k in range(D):
            def blk(i, z): h = F.relu(layers[i](z)); return h + z if (RES and i > 0) else h
            W_ = int(os.environ.get('WIN', 2)); st_ = max(0, k - W_ + 1); hk = ins[st_].detach() if st_ > 0 else x   # WIN: how many layers each local loss reaches back through
            for i_ in range(st_, k + 1): hk = blk(i_, hk)
            loss = loss + F.cross_entropy(heads[k](hk), y)
        loss.backward()
    else: hs_ = forward(x, True); sum(F.cross_entropy(heads[k](hs_[k]), y) for k in range(D) if (k + 1) % BLK == 0).backward()
    opt.step()
b = bpb(); R = {'method': M, 'block': BLK, 'res': RES, 'D': D, 'H': H, 'steps': STEPS, 'seed': SEED, 'bpb_final': round(b[-1], 4), 'bpb_per_layer': [round(v, 4) for v in b], 'secs': round(time.time() - t0)}
print(json.dumps(R), flush=True); os.makedirs('/home/claude/ll', exist_ok=True); json.dump(R, open(f'/home/claude/ll/deep_{M}_D{D}_H{H}_B{BLK}{"_res" if RES else ""}_s{SEED}.json', 'w'))
