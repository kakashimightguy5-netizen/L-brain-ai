"""L-pure prototype (note 149): only add/sub/compare/shift/lookup in the deployed forward. Training uses float + straight-through estimators.
Blocks: StructTern (sparse ternary wiring, pow2 gain), shift-leak scan neuron with spike counts, associative memory (binary sparse address, delta rule with add/shift)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

def round_ste(x): return x + (torch.round(x) - x).detach()
def tern_ste(w):
    sc = w.abs().mean(-1, keepdim=True).clamp(min=1e-8); q = torch.clamp(torch.round(w / sc), -1, 1); return w / sc + (q - w / sc).detach()
def pow2_ste(logg): return torch.exp2(round_ste(logg))            # gain = power of two = a shift

class StructTern(nn.Module):
    """y_j = 2^g_j * sum_k t_jk x[idx_jk]; idx = power-of-two strides (butterfly-like reach) + random long links; t in {-1,0,1}. adds only when deployed."""
    def __init__(s, din, dout, K, seed=0):
        super().__init__(); K = min(K, din); g = torch.Generator().manual_seed(seed + 17 * din + dout)
        strides = [0] + [2 ** i for i in range(int(math.log2(max(din, 2))))]; nS = min(len(strides), K - K // 4 if K > 4 else K)
        j = torch.arange(dout)[:, None] * din // dout
        idx = torch.cat([(j + torch.tensor(strides[:nS])[None]) % din, torch.randint(0, din, (dout, K - nS), generator=g)], 1)
        s.register_buffer('idx', idx); s.w = nn.Parameter(torch.randn(dout, K, generator=g) * 0.5); s.logg = nn.Parameter(torch.full((dout,), -math.log2(math.sqrt(K))))
        s.din, s.dout, s.K = din, dout, K
    def forward(s, x): return (x[..., s.idx] * tern_ste(s.w)).sum(-1) * pow2_ste(s.logg)
    def adds(s): return s.dout * s.K

class ShiftLeakSpike(nn.Module):
    """h_t = h_{t-1} - (h_{t-1} >> sh) + u_t (leak by power-of-two shift), spike count n = clamp(round(h / 2^p)). Linear scan, then quantised."""
    def __init__(s, d, K=8):
        super().__init__(); s.sh = nn.Parameter(torch.linspace(0.5, 5.5, d)); s.p = nn.Parameter(torch.zeros(d)); s.K = K
    def forward(s, u):
        a = 1 - torch.exp2(-round_ste(s.sh).clamp(0.0, 7.0)); h = torch.zeros_like(u[:, 0]); out = []
        for t in range(u.size(1)): h = a * h + u[:, t]; out.append(h)
        h = torch.stack(out, 1); th = torch.exp2(round_ste(s.p))
        return torch.clamp(round_ste(h / th), -s.K, s.K) * th

class AssocMem(nn.Module):
    """Sparse distributed memory with delta rule. address = top-k of ternary-wired scores (binary mask over m rows); read = sum of k rows (>>log2 k);
    write (pair prev->cur): rows[mask_prev] += lr * (v_cur - read_prev). lr, k powers of two -> shifts. Keeps 'prediction = storage = learning'."""
    def __init__(s, d, m=64, k=4, Kw=16, lr=1.0):
        super().__init__(); s.m, s.k, s.lr = m, k, lr; s.addr = StructTern(d, m, Kw, seed=1); s.val = StructTern(d, d, Kw, seed=2); s.out = StructTern(d, d, Kw, seed=3)
    def forward(s, x):
        B, T, D = x.shape; sc = s.addr(x); top = sc.topk(s.k, -1).indices; mask = torch.zeros_like(sc).scatter(-1, top, 1.0)
        mask = mask + (torch.sigmoid(sc) - torch.sigmoid(sc).detach())              # straight-through so the address function learns
        v = s.val(x); M = x.new_zeros(B, s.m, D); outs = []; mp = rp = None
        for t in range(T):
            if mp is not None: M = M + s.lr * mp[:, :, None] * (v[:, t] - rp)[:, None, :]       # delta write for pair (t-1 -> t)
            r = torch.einsum('bm,bmd->bd', mask[:, t], M) / s.k; outs.append(r); mp, rp = mask[:, t], r
        return s.out(torch.stack(outs, 1))

class Block(nn.Module):
    def __init__(s, d, K, mem=True, dense=False, m=64):
        super().__init__(); s.dense = dense
        mk = (lambda a, b: nn.Linear(a, b, bias=False)) if dense else (lambda a, b: StructTern(a, b, K, seed=a * 3 + b))
        s.inp, s.out = mk(d, d), mk(d, d); s.neu = ShiftLeakSpike(d); s.mem = AssocMem(d, m=m) if mem else None
        s.f1, s.f2 = mk(d, 2 * d), mk(2 * d, d)
    def forward(s, x):
        x = x + s.out(s.neu(s.inp(x)))
        if s.mem is not None: x = x + s.mem(x)
        return x + s.f2(torch.clamp(round_ste(F.relu(s.f1(x))), 0, 8))
    def adds(s): return sum(m.adds() for m in s.modules() if isinstance(m, StructTern))

class LPure(nn.Module):
    def __init__(s, vocab, d=64, n_layers=2, K=16, mem=True, dense=False):
        super().__init__(); s.emb = nn.Embedding(vocab, d); s.blocks = nn.ModuleList([Block(d, K, mem, dense) for _ in range(n_layers)])
        s.head = nn.Linear(d, vocab, bias=False) if dense else StructTern(d, vocab, K * 2, seed=9)
    def forward(s, tok):
        x = s.emb(tok)
        for b in s.blocks: x = b(x)
        return s.head(x)
    def adds_per_token(s): return sum(b.adds() for b in s.blocks) + (0 if isinstance(s.head, nn.Linear) else s.head.adds())
