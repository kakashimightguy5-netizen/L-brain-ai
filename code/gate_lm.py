"""Matrix-free test (vault note 131): byte-level next-byte prediction from the previous K bytes, no attention, no recurrence.
 gate  : differentiable logic-gate network (Petersen et al. 2022/24 style): fixed random wiring, each node = one of 16 two-input boolean gates (softmax-relaxed in training, argmax at test), output = popcount per byte class. No multiplications at inference.
 dense : plain MLP on the same K*8 input bits (matrix multiplications), the matched dense reference.
Usage: python gate_lm.py gate W D [K] [STEPS] [SEED]   |   python gate_lm.py dense H 0 [K] [STEPS] [SEED]
Reports bits/byte on the held-out val file (soft + hard for gate), modelled storage bits, and operation counts. MEASURED on CPU; storage/ops are MODELLED counts."""
import os, sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
MODE, A1, A2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 8; STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3000; SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
torch.manual_seed(SEED); torch.set_num_threads(int(os.environ.get('THREADS', 1))); DATA = os.environ.get('LDATA', '/home/claude/data'); BS = int(os.environ.get('BS', 256)); LR = float(os.environ.get('LR', 0.02 if MODE == 'gate' else 2e-3))
tr = torch.frombuffer(bytearray(open(os.path.join(DATA, 'py_big_train.txt'), 'rb').read(16_000_000)), dtype=torch.uint8).long()
va = torch.frombuffer(bytearray(open(os.path.join(DATA, 'py_big_val.txt'), 'rb').read()), dtype=torch.uint8).long(); va = va[len(va) // 2: len(va) // 2 + 200_000]
SH = torch.arange(8)
def bits(x): return ((x[..., None] >> SH) & 1).float().flatten(-2)   # [B,K,8] -> [B,K*8]
def batch(src, B):
    ix = torch.randint(0, len(src) - K - 1, (B,)); w = torch.stack([src[i:i + K + 1] for i in ix]); return bits(w[:, :K]), w[:, K]
def gates(a, b):   # 16 relaxed boolean functions of a,b in [0,1]; returns [..., 16]
    ab = a * b; return torch.stack([torch.zeros_like(a), ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab, 1 - (a + b - ab), 1 - (a + b - 2 * ab), 1 - b, 1 - b + ab, 1 - a, 1 - a + ab, 1 - ab, torch.ones_like(a)], -1)
TT = gates(torch.tensor([0., 0., 1., 1.]), torch.tensor([0., 1., 0., 1.]))   # [4,16] truth tables
class LogicLayer(nn.Module):
    def __init__(s, nin, nout):
        super().__init__(); s.register_buffer('ia', torch.randperm(max(nin, nout))[:nout] % nin); s.register_buffer('ib', (s.ia + 1 + torch.randint(0, nin - 1, (nout,))) % nin)
        w = torch.randn(nout, 16) * 0.1; w[:, 3] += 2.0 if os.environ.get('RESID', '1') == '1' else 0.0; s.w = nn.Parameter(w)   # RESID flag: start near pass-through
    def forward(s, x, hard=False):
        a, b = x[:, s.ia], x[:, s.ib]
        if hard: return TT[(2 * a + b).long(), s.w.argmax(-1)[None, :].expand(x.size(0), -1)]   # truth-table lookup: pure boolean, no arithmetic
        return (gates(a, b) * F.softmax(s.w, -1)[None]).sum(-1)
class GateLM(nn.Module):
    def __init__(s, W, D, nin, G=int(os.environ.get("GROUP", 8))):
        super().__init__(); s.G = G; dims = [nin] + [W] * (D - 1) + [256 * G]; s.layers = nn.ModuleList([LogicLayer(dims[i], dims[i + 1]) for i in range(D)]); s.tau = float(os.environ.get("TAU", 0.5))
    def forward(s, x, hard=False):
        for l in s.layers: x = l(x, hard)
        return x.view(x.size(0), 256, s.G).sum(-1) / s.tau
if MODE == 'gate':
    W, D = A1, A2; m = GateLM(W, D, K * 8); nodes = W * (D - 1) + 256 * m.G; store_bits = nodes * 4; ops = nodes   # 4 bits/gate id; wiring regenerable from a seed; 1 boolean op per node
else:
    H = A1; m = nn.Sequential(nn.Linear(K * 8, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 256)); npar = sum(p.numel() for p in m.parameters()); store_bits = npar * 8; ops = npar   # int8 storage; 1 MAC per weight
opt = torch.optim.Adam(m.parameters(), LR); t0 = time.time()
@torch.no_grad()
def bpb(hard):
    m.eval(); tot = 0.0; n = 0
    for i in range(0, 40_000, 2000):
        ix = torch.arange(i, i + 2000); w = torch.stack([va[j:j + K + 1] for j in ix]); lg = m(bits(w[:, :K]), hard) if MODE == 'gate' else m(bits(w[:, :K]))
        tot += F.cross_entropy(lg, w[:, K], reduction='sum').item(); n += len(ix)
    m.train(); return tot / n / math.log(2)
for it in range(STEPS):
    x, y = batch(tr, BS); loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % max(1, STEPS // 5) == 0: print(it + 1, 'loss_bits', round(loss.item() / math.log(2), 4), f'{time.time()-t0:.0f}s', flush=True)
R = {'mode': MODE, 'A1': A1, 'A2': A2, 'K': K, 'steps': STEPS, 'seed': SEED, 'store_bits_MODELLED': store_bits, 'ops_per_byte_MODELLED': ops, 'bpb_soft': round(bpb(False), 4)}
if MODE == 'gate': R['bpb_hard'] = round(bpb(True), 4)
R['secs'] = round(time.time() - t0); print(json.dumps(R), flush=True)
OUT = os.environ.get('LOUT', 'results'); os.makedirs(OUT, exist_ok=True); json.dump(R, open(os.path.join(OUT, f'gatelm_{MODE}_{A1}_{A2}_K{K}_s{SEED}.json'), 'w'))
