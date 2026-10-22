"""Architecture L, version L-0. No attention, no transformer components.
Core layer ("cortical sheet"):
  - membrane state h_t = a_t * h_{t-1} + b_t  (diagonal, input-gated leak; a_t in [-1, 1] so a neuron can FLIP sign -> state tracking)
  - neurons communicate by SPIKE COUNTS (integer 0..K / -K..K per step, straight-through surrogate gradient) -> sparse, event-driven
  - channel mixing by a spiking feed-forward (dendrite-like) block
  Cost per token is constant: O(D^2) per layer, independent of how long the text is. No KV cache.
Hippocampus (episodic index, optional):
  - hash index of the last k bytes (pattern separation) -> position where that pattern last occurred -> the byte that followed it
    (pattern completion). O(1) dict lookup per token, no scan over the past.
  - a learned gate ('do I trust this memory?') mixes the recalled byte into the prediction.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
V = 256

class SpikeCount(torch.autograd.Function):
    """Integer spike count in [lo, hi]; gradient passes straight through inside the range (surrogate)."""
    MASK = bool(int(__import__('os').environ.get('SPIKE_MASK', 0)))   # flag: save a 1-byte bool mask instead of the fp32 input
    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.lo, ctx.hi = lo, hi; ctx.mask_mode = SpikeCount.MASK
        if ctx.mask_mode: ctx.save_for_backward((x > lo - 0.5) & (x < hi + 0.5))
        else: ctx.save_for_backward(x)
        return torch.clamp(torch.round(x), lo, hi)
    @staticmethod
    def backward(ctx, g):
        t, = ctx.saved_tensors
        m = t if ctx.mask_mode else ((t > ctx.lo - 0.5) & (t < ctx.hi + 0.5))
        return g * m.to(g.dtype), None, None
SPIKE_STE = bool(int(__import__('os').environ.get('SPIKE_STE', 0)))   # flag (default off): compile-friendly straight-through spike count (same forward, same gradient window) built from plain tensor ops, no autograd.Function
def spikes(x, lo, hi):
    if SPIKE_STE:
        xc = torch.clamp(x, lo - 0.5, hi + 0.5)                         # gradient 1 inside (lo-0.5, hi+0.5), 0 outside = SpikeCount.backward window
        return xc + (torch.clamp(torch.round(x), lo, hi) - xc).detach()
    return SpikeCount.apply(x, lo, hi)

class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * s.w

class LLayer(nn.Module):
    def __init__(s, d, neg=True, K=None, spiking=True, homeo=False):
        K = K or int(__import__('os').environ.get('SPIKEK', 4))
        super().__init__(); s.neg, s.K, s.spiking, s.homeo = neg, K, spiking, homeo
        s.n1 = RMSNorm(d); s.inp = nn.Linear(d, 2 * d); s.out = nn.Linear(d, d)
        s.n2 = RMSNorm(d); s.f1 = nn.Linear(d, 3 * d); s.f2 = nn.Linear(3 * d, d)
        s.theta = nn.Parameter(torch.full((d,), 0.5)); s.theta2 = nn.Parameter(torch.full((3 * d,), 0.5))
        with torch.no_grad(): s.inp.bias[d:].fill_(3.0)   # start with slow leak (memory)
        s.last_rate = []
    def gate(s, z):
        a = torch.sigmoid(z)
        return 2 * a - 1 if s.neg else a
    @staticmethod
    def scan_hs(a, b, h0):
        """Exact log-depth parallel scan (Hillis-Steele) of h_t = a_t h_{t-1} + b_t. Only products/sums: stable, GPU-friendly."""
        b = torch.cat([b[:, :1] + a[:, :1] * h0[:, None], b[:, 1:]], 1); k = 1; T = b.size(1)
        while k < T:
            b = torch.cat([b[:, :k], a[:, k:] * b[:, :-k] + b[:, k:]], 1)
            a = torch.cat([a[:, :k], a[:, k:] * a[:, :-k]], 1); k *= 2
        return b, b[:, -1]
    def scan(s, a, b, h0=None, C=8):
        """Exact chunked scan of h_t = a_t h_{t-1} + b_t (same result as the step loop, fewer python steps).
        Within a chunk: transfer M[t,k] = prod_{j=k+1..t} a_j = exp(L_t - L_k) * S_t * S_k (signs kept separately; stable)."""
        B, T, D = b.shape; h = torch.zeros(B, D, device=b.device) if h0 is None else h0; out = []
        if T == 1: h = a[:, 0] * h + b[:, 0]; return h[:, None], h
        for c0 in range(0, T, C):
            ac, bc = a[:, c0:c0 + C], b[:, c0:c0 + C]; n = ac.size(1)
            la = torch.log(ac.abs().clamp(min=1e-6)); sa = torch.where(ac < 0, -1.0, 1.0)
            Lc = la.cumsum(1); Sc = sa.cumprod(1)
            diff = Lc[:, :, None, :] - Lc[:, None, :, :]                       # (B, t, k, D)
            mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=b.device))[None, :, :, None]
            M = torch.where(mask, torch.exp(diff.clamp(max=0)), torch.zeros((), device=b.device)) * (Sc[:, :, None, :] * Sc[:, None, :, :])
            hc = (M * bc[:, None, :, :]).sum(2) + torch.exp(Lc) * Sc * h[:, None, :]
            out.append(hc); h = hc[:, -1]
        return torch.cat(out, 1), h
    def forward(s, x, h0=None):
        u, z = s.inp(s.n1(x)).chunk(2, -1); a = s.gate(z); b = (1 - a.abs()) * u
        h, hT = s.scan(a, b, h0)
        if s.spiking and s.homeo:   # homeostatic gain control: threshold scales with the layer's current activity (brain-like adaptive threshold)
            g = h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt().detach(); th = s.theta.abs().clamp(min=0.05)
            sp = spikes(h / (g * th), -s.K, s.K) * th
        elif s.spiking: sp = spikes(h / s.theta.abs().clamp(min=0.05), -s.K, s.K) * s.theta.abs().clamp(min=0.05)
        else: sp = h
        x = x + s.out(sp)
        g = s.f1(s.n2(x))
        if s.spiking: g2 = spikes(F.relu(g) / s.theta2.abs().clamp(min=0.05), 0, s.K) * s.theta2.abs().clamp(min=0.05)
        else: g2 = F.relu(g)
        if not s.training: s.last_rate = [(sp != 0).float().mean().item(), (g2 != 0).float().mean().item()]
        return x + s.f2(g2), hT

class Hippocampus:
    """Episodic hash index over raw bytes. For each position t and order k: did the last k bytes occur before?
    If yes, recall the byte that followed the most recent occurrence. O(1) per token (python dict)."""
    ORDERS = (3, 6, 12)
    @staticmethod
    def recall(seq):  # seq: list/1-D tensor of ints -> LongTensor (len(ORDERS), T) recalled next byte (256 = nothing)
        seq = seq.tolist() if torch.is_tensor(seq) else seq; T = len(seq); out = []
        for k in Hippocampus.ORDERS:
            idx = {}; r = [256] * T
            for t in range(T):
                if t + 1 >= k:
                    key = tuple(seq[t + 1 - k:t + 1])
                    if key in idx: r[t] = seq[idx[key] + 1]
                    if t + 1 < T: idx[key] = t   # store AFTER lookup: position t's pattern -> next byte seq[t+1]
            out.append(r)
        return torch.tensor(out)
    @staticmethod
    def batch(X): return torch.stack([Hippocampus.recall(x) for x in X], 1)   # (orders, B, T)

class LModel(nn.Module):
    def __init__(s, d=160, n_layers=4, neg=True, hippo=True, spiking=True, homeo=False):
        super().__init__(); s.hippo = hippo; s.emb = nn.Embedding(V, d)
        s.layers = nn.ModuleList([LLayer(d, neg, spiking=spiking, homeo=homeo) for _ in range(n_layers)]); s.norm = RMSNorm(d); s.head = nn.Linear(d, V)
        if hippo:
            no = len(Hippocampus.ORDERS); s.remb = nn.ModuleList([nn.Embedding(V + 1, d) for _ in range(no)])
            s.trust = nn.Linear(d, no)          # 'do I trust this memory?' (self-knowing gate)
    def forward(s, x, rec=None, state=None):
        z = s.emb(x)
        if s.hippo: z = z + sum(e(r) for e, r in zip(s.remb, rec))
        new = []
        for i, l in enumerate(s.layers):
            z, h = l(z, None if state is None else state[i]); new.append(h)
        hz = s.norm(z); logits = s.head(hz)
        if s.hippo:
            w = F.softplus(s.trust(hz))           # (B,T,orders)
            boost = torch.zeros_like(logits)
            for k, r in enumerate(rec):
                valid = (r < 256).float()
                boost = boost.scatter_add(-1, r.clamp(max=255).unsqueeze(-1), (w[..., k] * valid * 8.0).unsqueeze(-1))
            logits = logits + boost
        return logits, new

# ---------------- transformer control (outside competitor, NOT part of L) ----------------
class TBlock(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d); s.n2 = nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        B, T, D = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + s.mlp(s.n2(x))
class TModel(nn.Module):
    def __init__(s, d=152, n_layers=4, h=4, T=256):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.zeros(T, d)); s.blocks = nn.ModuleList([TBlock(d, h) for _ in range(n_layers)])
        s.n = nn.LayerNorm(d); s.head = nn.Linear(d, V)
    def forward(s, x, rec=None, state=None):
        z = s.emb(x) + s.pos[:x.size(1)]
        for b in s.blocks: z = b(z)
        return s.head(s.n(z)), None
def nparams(m): return sum(p.numel() for p in m.parameters())
