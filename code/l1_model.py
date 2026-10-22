"""Architecture L, version L-1 (no attention, no transformer components).
Changes vs L-0:
  1. compiled C++ membrane scan (training ~12x faster in the scan)
  2. spike-rate energy budget on the state neurons (loss penalty on mean |spike count|) -> sparser firing
  3. FUZZY hippocampus: besides exact byte patterns (orders 3/6/12), it indexes the layer-2 brain state with
     locality-sensitive hashing (sign of fixed random projections, 2 tables x 16 bits). Similar situations land in the
     same bucket -> recall what came next last time, even if the bytes differ (typos, renamed variables).
     Still O(1) per token: one hash + one dict lookup; no scan over the past.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import l_model as M0, lscan
V = 256

class LLayer1(M0.LLayer):
    def scan(s, a, b, h0=None):
        B, T, D = b.shape; h0 = torch.zeros(B, D, device=b.device) if h0 is None else h0
        if b.is_cuda and __import__('os').environ.get('SCAN_MODE', 'cuscan') == 'hs':
            return M0.LLayer.scan_hs(a, b, h0)      # flag: log-depth parallel associative scan (exact, no custom kernel)
        if b.is_cuda:
            try:
                import cuscan
                if cuscan.AVAILABLE: return cuscan.scan(a, b, h0, M0.LLayer.scan_hs)   # GPU kernel (self-verified)
            except ImportError: pass
            return M0.LLayer.scan_hs(a, b, h0)   # log-depth parallel scan
        if not lscan.AVAILABLE: return M0.LLayer.scan_hs(a, b, h0)   # exact chunked scan (GPU-friendly)
        h = lscan.scan(a, b, h0); return h, h[:, -1]
    def forward(s, x, h0=None):
        u, z = s.inp(s.n1(x)).chunk(2, -1); a = s.gate(z); b = (1 - a.abs()) * u
        mk = getattr(s, 'mask', None)
        if mk is not None: a = torch.where(mk > 0, a, torch.ones_like(a)); b = b * mk   # skipped token: neuron keeps its state unchanged
        h, hT = s.scan(a, b, h0); th = s.theta.abs().clamp(min=0.05)
        gain = h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt().detach() if s.homeo else 1.0   # homeostatic threshold
        cnt = M0.spikes(h / (gain * th), -s.K, s.K); s.rate_cost = cnt.abs().mean()          # energy budget term
        x = x + s.out(cnt * th)
        if getattr(s, 'moe', None) is not None:           # flag domain_experts (meta/experts.py): routed spiking expert FFNs
            x = x + s.moe(s.n2(x)); s.rate_cost = s.rate_cost + s.moe.rate
            if getattr(s, 'record', False): s.tok_rate = ((cnt != 0).float().mean(-1), torch.zeros_like(cnt[..., 0]))
            return x, hT
        g = s.f1(s.n2(x)); th2 = s.theta2.abs().clamp(min=0.05)
        G = getattr(s, 'spike_groups', 1)
        if G > 1:   # flag spike_groups (H4): groups of G adjacent FFN neurons fire or stay silent together
            v = F.relu(g) / th2; sel = (v.view(*v.shape[:-1], -1, G).mean(-1) >= 0.5).float()
            cnt2 = M0.spikes(v, 0, s.K) * sel.repeat_interleave(G, -1); s.group_sel = sel
        else:
            cnt2 = M0.spikes(F.relu(g) / th2, 0, s.K)
        s.rate_cost = s.rate_cost + cnt2.mean()
        if not s.training: s.last_rate = [(cnt != 0).float().mean().item(), (cnt2 != 0).float().mean().item()]
        if getattr(s, 'record', False): s.tok_rate = ((cnt != 0).float().mean(-1), (cnt2 != 0).float().mean(-1))   # per-token firing (metacognition signal)
        return x + s.f2(cnt2 * th2), hT

class FuzzyIndex:
    """LSH over brain state: key = sign pattern of fixed random projections. Most recent position per bucket."""
    def __init__(s, D, tables=2, bits=16, seed=1234):
        g = torch.Generator().manual_seed(seed); s.R = [torch.randn(D, bits, generator=g) for _ in range(tables)]
        s.pow = (2 ** torch.arange(bits)).long()
    def keys(s, z):   # z (B,T,D) -> list of (B,T) int keys
        return [((z @ R.to(z.device) > 0).long() * s.pow.to(z.device)).sum(-1) for R in s.R]
    @staticmethod
    def recall(keys, seq):  # keys (B,T) ints, seq (B,T) bytes -> recalled next byte (B,T), 256 = none
        out = torch.full_like(seq, 256); K = keys.tolist(); S = seq.tolist()
        for i in range(len(K)):
            idx = {}; ki, si, oi = K[i], S[i], out[i]
            row = [256] * len(ki)
            for t, k in enumerate(ki):
                p = idx.get(k)
                if p is not None and p + 1 < len(si): row[t] = si[p + 1]
                idx[k] = t
            out[i] = torch.tensor(row)
        return out

class L1Model(nn.Module):
    def __init__(s, d=160, n_layers=4, fuzzy=True, mid=2, homeo=False):
        super().__init__(); s.fuzzy, s.mid = fuzzy, mid; s.emb = nn.Embedding(V, d)
        s.layers = nn.ModuleList([LLayer1(d, homeo=homeo) for _ in range(n_layers)]); s.norm = M0.RMSNorm(d); s.head = nn.Linear(d, V)
        no = len(M0.Hippocampus.ORDERS); s.remb = nn.ModuleList([nn.Embedding(V + 1, d) for _ in range(no)])
        if fuzzy: s.lsh = FuzzyIndex(d); s.femb = nn.ModuleList([nn.Embedding(V + 1, d) for _ in s.lsh.R])
        s.trust = nn.Linear(d, no + (len(s.lsh.R) if fuzzy else 0)); s.use_fuzzy = fuzzy; s.use_exact = True
    def forward(s, x, rec, block=None):
        """block: decoupled training block size (None = end-to-end). Returns logits, list of per-layer states (for local heads)."""
        if not s.use_exact: rec = torch.full_like(rec, 256)
        z = s.emb(x) + sum(e(r) for e, r in zip(s.remb, rec)); states = []; frec = []
        for i, l in enumerate(s.layers):
            if block and i > 0 and i % block == 0: z = z.detach()
            if s.fuzzy and i == s.mid:
                with torch.no_grad(): frec = [FuzzyIndex.recall(k, x) for k in s.lsh.keys(F.normalize(z, dim=-1))]
                if not s.use_fuzzy: frec = [torch.full_like(f, 256) for f in frec]
                z = z + sum(e(r) for e, r in zip(s.femb, frec))
            z, _ = l(z); states.append(z)
        hz = s.norm(z); logits = s.head(hz); w = F.softplus(s.trust(hz)); boost = torch.zeros_like(logits)
        for k, r in enumerate(list(rec) + list(frec)):
            boost = boost.scatter_add(-1, r.clamp(max=255).unsqueeze(-1), (w[..., k] * (r < 256).float() * 8.0).unsqueeze(-1))
        return logits + boost, states
    def rate_cost(s): return sum(l.rate_cost for l in s.layers) / len(s.layers)

class Trajectory:
    """Hippocampal trajectory replay: once an exact pattern match places us at position p of an earlier passage, keep
    following that passage forward (p+1, p+2, ...) even through mismatching bytes (typos, renamed names), until a new
    exact match relocates us. Also returns a 'last recall was right' flag (0 = wrong, 1 = right, 2 = no track).
    O(1) per token."""
    @staticmethod
    def recall(seq, orders=(6, 12)):
        T = len(seq); r = [256] * T; hit = [2] * T; idx = {k: {} for k in orders}; ptr = None; last_pred = None
        for t in range(T):
            if last_pred is not None: hit[t] = 1 if last_pred == seq[t] else 0
            found = None
            for k in sorted(orders, reverse=True):
                if t + 1 >= k:
                    p = idx[k].get(tuple(seq[t + 1 - k:t + 1]))
                    if p is not None and found is None: found = p
            for k in orders:
                if t + 1 >= k and t + 1 < T: idx[k][tuple(seq[t + 1 - k:t + 1])] = t
            if found is not None: ptr = found
            elif ptr is not None: ptr = ptr + 1
            if ptr is not None and ptr + 1 < t + 1: r[t] = seq[ptr + 1]; last_pred = r[t]
            else: last_pred = None
        return r, hit
    @staticmethod
    def batch(X):
        R, H = zip(*[Trajectory.recall(x.tolist()) for x in X]); return torch.tensor(R), torch.tensor(H)

class L1TModel(L1Model):
    """L-1 with trajectory replay instead of LSH fuzzy memory."""
    def __init__(s, d=160, n_layers=4, gated=False, homeo=False):
        super().__init__(d, n_layers, fuzzy=False, homeo=homeo); s.temb = nn.Embedding(V + 1, d); s.hemb = nn.Embedding(3, d)
        s.trust = nn.Linear(d, len(M0.Hippocampus.ORDERS) + 1); s.use_track = True; s.gated = gated
        if gated: s.tgate = nn.Linear(d, 1)   # 'is the replayed track still on course?' (from the hit flag) -> scales its influence
    def forward(s, x, rec, block=None):
        tr, hit = Trajectory.batch(x.cpu()); tr, hit = tr.to(x.device), hit.to(x.device)
        if not s.use_track: tr = torch.full_like(tr, 256); hit = torch.full_like(hit, 2)
        if not s.use_exact: rec = torch.full_like(rec, 256)
        he = s.hemb(hit); g = torch.sigmoid(s.tgate(he)) if s.gated else 1.0
        z = s.emb(x) + sum(e(r) for e, r in zip(s.remb, rec)) + g * s.temb(tr) + he; states = []
        for i, l in enumerate(s.layers):
            if block and i > 0 and i % block == 0: z = z.detach()
            z, _ = l(z); states.append(z)
        hz = s.norm(z); logits = s.head(hz); w = F.softplus(s.trust(hz)); boost = torch.zeros_like(logits)
        if s.gated: w = torch.cat([w[..., :-1], w[..., -1:] * g], -1)
        for k, r in enumerate(list(rec) + [tr]):
            boost = boost.scatter_add(-1, r.clamp(max=255).unsqueeze(-1), (w[..., k] * (r < 256).float() * 8.0).unsqueeze(-1))
        return logits + boost, states
