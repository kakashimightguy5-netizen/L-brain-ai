"""Architecture L, version L-5 candidate. Fixes the 'capacity ceiling' found at 4x training budget (note 109):
 1. ERROR-CORRECTING FAST-WEIGHT MEMORY (delta rule; brain analogue: error-driven synaptic plasticity).
      S_t = a_t * S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T     o_t = S_t q_t
    The memory first reads what it already stores for key k_t and writes only the difference -> no blurring pile-up.
    Exact chunk-parallel training via a unit-lower-triangular solve per chunk (verified vs the token loop).
 2. SHORT DELAY LINE (depthwise causal conv, kernel 4; analogue: dendritic / axonal delays) on q, k, v:
    exact access to the last few bytes. Constant cost per token, no look-back over the whole text.
Cost per token stays CONSTANT. Everything else (spiking diagonal neurons, hippocampus, gated trajectory, homeostasis) is L-4."""
import math, torch, torch.nn as nn, torch.nn.functional as F
import os
import l_model as M0, l1_model as M1, l4_model as M4
DELTA_V = int(os.environ.get('DELTA_V', 2))
RECORD = False; LAST_SURPRISE = None      # flag: record per-token internal signals (off = identical computation)

def delta_chunk(q, k, v, la, b, S, C=64):
    """q,k,v: (B,H,T,dk) ; la: (B,H,T) log decay ; b: (B,H,T) write strength in (0,1) ; S: (B,H,dv,dk). Returns o, S."""
    B, H, T, dk = q.shape; outs = []
    for c0 in range(0, T, C):
        qc, kc, vc, lc, bc = q[:, :, c0:c0 + C], k[:, :, c0:c0 + C], v[:, :, c0:c0 + C], la[:, :, c0:c0 + C], b[:, :, c0:c0 + C]
        n = qc.size(2); g = lc.cumsum(-1)                                            # log gamma_t
        Gm = g[..., :, None] - g[..., None, :]                                       # log gamma_t/gamma_i
        tril0 = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device), -1); tril1 = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))
        Gs = Gm.masked_fill(~tril0, float('-inf')).exp(); Gi = Gm.masked_fill(~tril1, float('-inf')).exp()
        A = torch.eye(n, device=q.device) + bc[..., :, None] * (kc @ kc.transpose(-1, -2)) * Gs   # unit lower triangular
        rhs = bc[..., None] * (vc - g.exp()[..., None] * (kc @ S.transpose(-1, -2)))           # b (V - gamma K S0^T)
        U = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
        o = g.exp()[..., None] * (qc @ S.transpose(-1, -2)) + ((qc @ kc.transpose(-1, -2)) * Gi) @ U
        outs.append(o)
        w = (g[..., -1:] - g).exp()[..., None]
        S = S * g[..., -1].exp()[..., None, None] + (U * w).transpose(-1, -2) @ kc
    return torch.cat(outs, 2), S


def delta_chunk2(q, k, v, la, b, S, C=64):
    """Same math as delta_chunk, reorganised (WY / UT form) so that ALL chunks are processed in parallel by big batched
    matmuls + ONE batched triangular solve; only a tiny dk x dk recurrence runs chunk by chunk."""
    B, H, T, dk = q.shape; dv = v.size(-1); pad = (-T) % C
    if pad:
        q, k, v = [F.pad(t, (0, 0, 0, pad)) for t in (q, k, v)]; la = F.pad(la, (0, pad)); b = F.pad(b, (0, pad))
    N = q.size(2) // C
    q, k, v = [t.reshape(B, H, N, C, -1) for t in (q, k, v)]; la, b = la.reshape(B, H, N, C), b.reshape(B, H, N, C)
    g = la.cumsum(-1); ge = g.exp()                                                # gamma_t within chunk
    Gm = g[..., :, None] - g[..., None, :]
    tril0 = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), -1); tril1 = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    Gs = Gm.masked_fill(~tril0, float('-inf')).exp(); Gi = Gm.masked_fill(~tril1, float('-inf')).exp()
    A = torch.eye(C, device=q.device, dtype=q.dtype) + b[..., :, None] * (k @ k.transpose(-1, -2)) * Gs
    rhs = torch.cat([b[..., None] * v, (b * ge)[..., None] * k], -1)                # solve for U0 and W together
    X = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True); U0, W = X[..., :dv], X[..., dv:]
    P = (q @ k.transpose(-1, -2)) * Gi
    O0 = P @ U0; Qe = ge[..., None] * q - P @ W                                    # o_n = O0_n + Qe_n S_n^T
    w = (g[..., -1:] - g).exp()[..., None]; Kw = w * k
    M1 = U0.transpose(-1, -2) @ Kw; M2 = W.transpose(-1, -2) @ Kw; gC = ge[..., -1][..., None, None]   # S' = gC S + M1 - S M2
    Ss = []
    for n in range(N):
        Ss.append(S); S = gC[:, :, n] * S + M1[:, :, n] - S @ M2[:, :, n]
    Sn = torch.stack(Ss, 2)                                                         # state at start of each chunk
    o = O0 + Qe @ Sn.transpose(-1, -2)
    if RECORD:                                    # metacognition signal: delta-rule surprise ||v - a S k|| per token (= ||u|| / b)
        U = U0 - W @ Sn.transpose(-1, -2)
        global LAST_SURPRISE; LAST_SURPRISE = (U.norm(dim=-1) / b.clamp(min=1e-4)).reshape(B, H, N * C)[:, :, :T].mean(1)
    return o.reshape(B, H, N * C, dv)[:, :, :T], S

def delta_loop(q, k, v, la, b, S):
    outs = []
    for t in range(q.size(2)):
        a = la[:, :, t].exp()[..., None, None]; kt, vt, qt, bt = k[:, :, t], v[:, :, t], q[:, :, t], b[:, :, t][..., None]
        Sa = a * S; S = Sa + (bt * (vt - (Sa @ kt[..., None])[..., 0]))[..., None] * kt[..., None, :]
        outs.append((S @ qt[..., None])[..., 0])
    return torch.stack(outs, 2), S

class DeltaMemory(nn.Module):
    def __init__(s, d, heads=4, C=None, conv=4):
        C = C or int(os.environ.get('DELTA_C', 64))
        super().__init__(); s.h, s.dk, s.C, s.K = heads, d // heads, C, conv; inner = heads * (d // heads)   # inner = d when heads divides d
        s.n = M0.RMSNorm(d); s.qkv = nn.Linear(d, 3 * inner, bias=False); s.ga = nn.Linear(d, heads); s.gb = nn.Linear(d, heads); s.o = nn.Linear(inner, d)
        s.conv = nn.Conv1d(3 * inner, 3 * inner, conv, groups=3 * inner, bias=False) if conv else None
        with torch.no_grad():
            s.ga.bias.fill_(3.0)
            if conv: s.conv.weight.zero_(); s.conv.weight[:, 0, -1] = 1.0          # start as identity (current byte only)
    def forward(s, x):
        B, T, D = x.shape; H, dk = s.h, s.dk; xn = s.n(x); z = s.qkv(xn)
        if s.conv is not None: z = s.conv(F.pad(z.transpose(1, 2), (s.K - 1, 0))).transpose(1, 2)
        z = F.silu(z)
        q, k, v = z.view(B, T, 3, H, dk).permute(2, 0, 3, 1, 4)
        k = F.normalize(k, dim=-1); q = F.normalize(q, dim=-1)
        la = F.logsigmoid(s.ga(xn)).transpose(1, 2); b = torch.sigmoid(s.gb(xn)).transpose(1, 2)
        mk = getattr(s, 'nowrite_mask', None)                  # NOWRITE skim: tokens that stopped thinking do not write the fast-weight memory (flag; None = off)
        if mk is not None: b = b * mk[..., 0][:, None, :]
        if x.is_cuda and int(os.environ.get('DELTAKERN', 0)):
            import deltakern                                    # fused on-chip kernel (self-verified, falls back if wrong)
            o = deltakern.delta(q, k, v, la, b, lambda *a_: delta_chunk2(*a_, C=s.C)) if deltakern.AVAILABLE else delta_chunk2(q, k, v, la, b, x.new_zeros(B, H, dk, dk), s.C)[0]
        else:
            o, _ = (delta_chunk2 if DELTA_V == 2 else delta_chunk)(q, k, v, la, b, x.new_zeros(B, H, dk, dk), s.C)
            if RECORD: s.last_surprise = LAST_SURPRISE
        return x + s.o(o.transpose(1, 2).reshape(B, T, H * dk))

class L5Model(M4.L4Model):
    def __init__(s, d=144, n_layers=4, heads=4, ffn_mult=2, conv=4, rhit=False, tsign=False, colpos=False, mixcopy=False, bracket=False, copyfeat=False):
        super().__init__(d=d, n_layers=n_layers, heads=heads, ffn_mult=ffn_mult)
        s.fw = nn.ModuleList([DeltaMemory(d, heads, conv=conv) for _ in range(n_layers)])
        s.rhit = rhit
        s.tsign, s.colpos, s.mixcopy, s.bracket, s.copyfeat = tsign, colpos, mixcopy, bracket, copyfeat
        if copyfeat: s.cfemb = nn.Embedding(10, d); nn.init.zeros_(s.cfemb.weight)   # copy-gate signal, starts silent
        if bracket: s.bracketemb = nn.Embedding(17, d); nn.init.zeros_(s.bracketemb.weight)   # counter channel, starts silent
        if tsign:
            s.veto = nn.Linear(d, s.trust.out_features)
            with torch.no_grad(): s.veto.weight.zero_(); s.veto.bias.fill_(-4.0)          # starts as 'no veto'
        if colpos: s.colemb, s.indemb, s.pindemb = nn.Embedding(65, d), nn.Embedding(33, d), nn.Embedding(33, d)   # layout place cells
        if rhit: s.rhemb = nn.ModuleList([nn.Embedding(3, d) for _ in M0.Hippocampus.ORDERS])   # 'was this memory right last time?'
        # new input signals start SILENT (zero), so they cannot inject noise into a working network; they grow only if useful
        for e in ([s.colemb, s.indemb, s.pindemb] if colpos else []) + (list(s.rhemb) if rhit else []):
            nn.init.zeros_(e.weight)
    def forward(s, x, rec, block=None):
        if not s.rhit: return super().forward(x, rec, block)
        # self-knowing recall: for every hippocampus order, tell the network whether that memory's LAST recall was right
        # (0 wrong / 1 right / 2 no recall). Lets the trust gate learn when to ignore a memory (fixes over-trusted wrong recalls).
        prev = torch.cat([torch.full_like(rec[:, :, :1], 256), rec[:, :, :-1]], 2)
        rh = torch.where(prev < 256, (prev == x.unsqueeze(0)).long(), torch.full_like(prev, 2))
        s._rh_add = sum(e(h) for e, h in zip(s.rhemb, rh))
        return super().forward(x, rec, block)

if __name__ == '__main__':
    torch.manual_seed(0); B, H, T, dk = 2, 3, 150, 8
    q, k, v = torch.randn(B, H, T, dk, dtype=torch.float64), F.normalize(torch.randn(B, H, T, dk, dtype=torch.float64), dim=-1), torch.randn(B, H, T, dk, dtype=torch.float64)
    la = F.logsigmoid(torch.randn(B, H, T, dtype=torch.float64) + 3); b = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    S0 = torch.randn(B, H, dk, dk, dtype=torch.float64) * 0.1
    o1, s1 = delta_chunk(q, k, v, la, b, S0, 64); o2, s2 = delta_loop(q, k, v, la, b, S0); o3, s3 = delta_chunk2(q, k, v, la, b, S0, 64)
    print('v1 max err out', (o1 - o2).abs().max().item(), 'state', (s1 - s2).abs().max().item())
    print('v2 max err out', (o3 - o2).abs().max().item(), 'state', (s3 - s2).abs().max().item())
    for d in (144, 232): print(d, M0.nparams(L5Model(d=d, n_layers=4 if d == 144 else 6)), M0.nparams(M4.L4Model(d=d, n_layers=4 if d == 144 else 6)))
