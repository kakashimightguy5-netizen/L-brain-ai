"""Architecture L, version L-4 candidate: adds a FAST-WEIGHT MEMORY to every layer (brain analogue: short-term synaptic
plasticity / Hebbian fast weights). Each head keeps a small matrix S (dk x dk) that is rewritten as text is read:
   S_t = alpha_t * S_{t-1} + v_t k_t^T        (write: 'this key goes with this value', with learned forgetting alpha_t)
   o_t = S_t q_t                               (read: ask the memory)
Cost per token is CONSTANT (dk*dk per head), no cache that grows, no look-back over past tokens.
(Honest naming: mathematically this is the 'fast weights' / 'linear attention' family, not softmax attention.)
Training uses an exact chunk-parallel form; inference can run token by token with S as state.
The spiking diagonal neurons (sign-flipping, state tracking), hippocampus, trajectory replay, homeostasis are kept."""
import math, torch, torch.nn as nn, torch.nn.functional as F
import os, fastmem
import l_model as M0, l1_model as M1
FASTMEM = int(os.environ.get('FASTMEM', 1))

class FastWeights(nn.Module):
    def __init__(s, d, heads=4, C=64):
        super().__init__(); s.h, s.dk, s.C = heads, d // heads, C
        s.n = M0.RMSNorm(d); s.qkv = nn.Linear(d, 3 * d, bias=False); s.ga = nn.Linear(d, heads); s.o = nn.Linear(d, d)
        with torch.no_grad(): s.ga.bias.fill_(3.0)              # start with slow forgetting
    def forward(s, x):
        B, T, D = x.shape; H, dk = s.h, s.dk
        q, k, v = s.qkv(s.n(x)).view(B, T, 3, H, dk).permute(2, 0, 3, 1, 4)        # (B,H,T,dk)
        k = F.normalize(k, dim=-1); q = q / math.sqrt(dk)
        la = F.logsigmoid(s.ga(x)).transpose(1, 2)                                 # (B,H,T) log forget gate
        mk = getattr(s, 'mask', None)
        if mk is not None: m_ = mk[..., 0][:, None, :]; v = v * m_[..., None]; la = la * m_      # skipped token: no write, no forgetting
        S = x.new_zeros(B, H, dk, dk); outs = []
        for c0 in range(0, T, s.C):
            qc, kc, vc, lc = q[:, :, c0:c0 + s.C], k[:, :, c0:c0 + s.C], v[:, :, c0:c0 + s.C], la[:, :, c0:c0 + s.C]
            g = lc.cumsum(-1); n = qc.size(2)
            decay = (g[..., :, None] - g[..., None, :]).masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool, device=x.device)), float('-inf')).exp()
            o = ((qc @ kc.transpose(-1, -2)) * decay) @ vc + (qc @ S.transpose(-1, -2)) * g.exp()[..., None]
            outs.append(o)
            w = (g[..., -1:] - g).exp()[..., None]                                     # how much each write survives to chunk end
            S = S * g[..., -1].exp()[..., None, None] + (vc * w).transpose(-1, -2) @ kc
        o = torch.cat(outs, 2).transpose(1, 2).reshape(B, T, D)
        return x + s.o(o)

class L4Model(M1.L1TModel):
    def __init__(s, d=144, n_layers=4, heads=4, ffn_mult=2):
        super().__init__(d=d, n_layers=n_layers, gated=True, homeo=True)
        s.fw = nn.ModuleList([FastWeights(d, heads) for _ in range(n_layers)])
        for l in s.layers:                                  # slimmer feed-forward to keep parameters matched
            l.f1 = nn.Linear(d, ffn_mult * d); l.f2 = nn.Linear(ffn_mult * d, d); l.theta2 = nn.Parameter(torch.full((ffn_mult * d,), 0.5))
    def forward(s, x, rec, block=None):
        if FASTMEM: tr, hit = fastmem.traj_batch(x)
        else: tr, hit = M1.Trajectory.batch(x.cpu()); tr, hit = tr.to(x.device), hit.to(x.device)
        he = s.hemb(hit); g = torch.sigmoid(s.tgate(he))
        z = s.emb(x) + sum(e(r) for e, r in zip(s.remb, rec)) + g * s.temb(tr) + he; states = []
        ra = getattr(s, '_rh_add', None)
        if ra is not None: z = z + ra; s._rh_add = None
        if getattr(s, 'colpos', False):
            c_, i_, p_ = fastmem.layout(x); z = z + s.colemb(c_) + s.indemb(i_) + s.pindemb(p_)
        if getattr(s, 'bracket', False): z = z + s.bracketemb(fastmem.bracket_depth(x))
        if getattr(s, 'copyfeat', False): z = z + s.cfemb(fastmem.copy_feat(rec, tr))
        for i, l in enumerate(s.layers):
            if block and i > 0 and i % block == 0: z = z.detach()
            z = s.fw[i](z); z, _ = l(z); states.append(z)
        hz = s.norm(z); logits = s.head(hz)
        if getattr(s, 'mixcopy', False):
            # CALIBRATED COPY: mix in PROBABILITY space. weights = softmax over [own prediction, memory 1..4]; an unavailable
            # memory gets weight 0. Training makes each weight ~ P(that memory is right | context), so a wrong recall costs
            # only log(1/(1-weight)) instead of being forced by a large logit boost.
            mem = list(rec) + [tr]; t_ = s.trust(hz); t_ = torch.cat([t_[..., :-1], t_[..., -1:] + torch.log(g.clamp(min=1e-6))], -1)
            avail = torch.stack([(r < 256) for r in mem], -1)
            lw = torch.cat([torch.zeros_like(t_[..., :1]), t_.masked_fill(~avail, float('-inf'))], -1).log_softmax(-1)
            lp = logits.log_softmax(-1) + lw[..., :1]
            for k, r in enumerate(mem):
                add = torch.full_like(lp, float('-inf')).scatter(-1, r.clamp(max=255).unsqueeze(-1), lw[..., k + 1:k + 2])
                lp = torch.logaddexp(lp, add)
            return lp, states
        w = F.softplus(s.trust(hz))
        if getattr(s, 'tsign', False): w = w - F.softplus(s.veto(hz))          # signed trust: a memory can also be VETOED
        w = torch.cat([w[..., :-1], w[..., -1:] * g], -1)
        boost = torch.zeros_like(logits)
        for k, r in enumerate(list(rec) + [tr]):
            boost = boost.scatter_add(-1, r.clamp(max=255).unsqueeze(-1), (w[..., k] * (r < 256).float() * 8.0).unsqueeze(-1))
        return logits + boost, states

class L4EModel(L4Model):
    """L-4 + EFFORT GATE (thalamus-like): after each layer a 'halt' head predicts whether this layer's answer already equals
    the final answer. At inference, a token whose halt score exceeds the threshold stops thinking: deeper layers SKIP it
    (their neurons and fast-weight memory keep their state unchanged for that token) -> less compute on easy tokens."""
    def __init__(s, d=144, n_layers=4, heads=4):
        super().__init__(d, n_layers, heads)
        s.enorm = nn.ModuleList([M0.RMSNorm(d) for _ in range(n_layers - 1)]); s.halt = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_layers - 1)])
        s.etrust = nn.ModuleList([nn.Linear(d, len(M0.Hippocampus.ORDERS) + 1) for _ in range(n_layers - 1)])
    def readout(s, hz, trust, rec, tr, g):
        logits = s.head(hz); w = F.softplus(trust(hz)); w = torch.cat([w[..., :-1], w[..., -1:] * g], -1); boost = torch.zeros_like(logits)
        for k, r in enumerate(list(rec) + [tr]):
            boost = boost.scatter_add(-1, r.clamp(max=255).unsqueeze(-1), (w[..., k] * (r < 256).float() * 8.0).unsqueeze(-1))
        return logits + boost
    def forward(s, x, rec, block=None, thr=None):
        if FASTMEM: tr, hit = fastmem.traj_batch(x)
        else: tr, hit = M1.Trajectory.batch(x.cpu()); tr, hit = tr.to(x.device), hit.to(x.device)
        he = s.hemb(hit); g = torch.sigmoid(s.tgate(he))
        z = s.emb(x) + sum(e(r) for e, r in zip(s.remb, rec)) + g * s.temb(tr) + he
        outs, halts = [], []; active = torch.ones(x.shape + (1,), device=x.device); used = torch.zeros(x.shape, device=x.device); final = None
        for i, l in enumerate(s.layers):
            if thr is not None: l.mask = active; s.fw[i].mask = active
            zn = s.fw[i](z); zn, _ = l(zn); z = zn if thr is None else torch.where(active > 0, zn, z)
            used = used + active[..., 0]
            if i < len(s.layers) - 1:
                hz = s.enorm[i](z); lg = s.readout(hz, s.etrust[i], rec, tr, g); outs.append(lg); hl = s.halt[i](hz)[..., 0]; halts.append(hl)
                if thr is not None:
                    stop = (torch.sigmoid(hl) > thr).float()[..., None] * active
                    final = lg * stop if final is None else final + lg * stop
                    active = active * (1 - stop)
        for l in s.layers: l.mask = None
        for f in s.fw: f.mask = None
        last = s.readout(s.norm(z), s.trust, rec, tr, g); outs.append(last)
        if thr is not None: final = (final if final is not None else 0) + last * active; return final, used.mean().item() / len(s.layers)
        return outs, halts

class L4E2Model(L4EModel):
    """Effort gate v2 ('skim'): a token that stops thinking still WRITES into deeper layers' memories (neuron state +
    fast-weight write, using its current representation) but skips their outputs and feed-forward. Memory stays intact.
    Training practises random early exits so deeper layers learn to live with skimmed tokens.
    Cost of a skimmed layer = 6/11 of a full layer: memory qkv 3 + memory read o 1 + membrane inp 2 (of 11 d^2).
    (corrected 2026-10-09 from 5/11: the memory READ is needed because a skimmed layer's membrane input includes it;
     compute_used numbers reported before this date used 5/11 and are ~4 % optimistic for skimmed tokens)"""
    SKIM_COST = 6 / 11
    def forward(s, x, rec, block=None, thr=None, practice_p=0.0):
        if FASTMEM: tr, hit = fastmem.traj_batch(x)
        else: tr, hit = M1.Trajectory.batch(x.cpu()); tr, hit = tr.to(x.device), hit.to(x.device)
        he = s.hemb(hit); g = torch.sigmoid(s.tgate(he))
        z = s.emb(x) + sum(e(r) for e, r in zip(s.remb, rec)) + g * s.temb(tr) + he
        nL = len(s.layers); outs, halts = [], []; active = torch.ones(x.shape + (1,), device=x.device); cost = torch.zeros(x.shape, device=x.device); final = None
        rand_exit = None
        if s.training and practice_p > 0 and torch.rand(()) < practice_p:   # practise: 30% of tokens exit at a random layer
            ex = torch.randint(1, nL, x.shape, device=x.device); rand_exit = torch.where(torch.rand(x.shape, device=x.device) < 0.3, ex, torch.full_like(ex, nL))
        nw = os.environ.get('NOWRITE', '0') == '1' and (thr is not None or rand_exit is not None)
        for i, l in enumerate(s.layers):
            if nw: s.fw[i].nowrite_mask = active
            zn = s.fw[i](z); zn, _ = l(zn); z = torch.where(active > 0, zn, z)
            cost = cost + active[..., 0] + (1 - active[..., 0]) * (4 / 11 if nw else s.SKIM_COST)
            if i < nL - 1:
                hz = s.enorm[i](z); lg = s.readout(hz, s.etrust[i], rec, tr, g); outs.append(lg); hl = s.halt[i](hz)[..., 0]; halts.append(hl)
                if thr is not None:
                    stop = (torch.sigmoid(hl) > thr).float()[..., None] * active
                    final = lg * stop if final is None else final + lg * stop; active = active * (1 - stop)
                elif rand_exit is not None:
                    active = active * (rand_exit > i + 1).float()[..., None]
        for f in s.fw: f.nowrite_mask = None
        last = s.readout(s.norm(z), s.trust, rec, tr, g); outs.append(last)
        if thr is not None: final = (final if final is not None else 0) + last * active; return final, cost.mean().item() / nL
        return outs, halts
