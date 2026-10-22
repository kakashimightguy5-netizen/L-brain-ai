"""M3 - domain-routed expert modules for L's spiking feed-forward (brain analogy: specialised cortical areas).
Each expert is a small spiking FFN (f1 -> integer spike counts -> f2). Routing modes:
  'segment' : ONE expert per 64-byte segment, chosen from the mean input of the PREVIOUS segment (causal, slow summary).
              Experts not chosen stay completely silent for the whole segment (whole-unit power gating in hardware).
  'token'   : Switch-style top-1 per byte (the token-level MoE baseline, same experts, same active params).
Output is scaled by the chosen expert's router probability (gradient path to the router) + a load-balancing loss."""
import torch, torch.nn as nn, torch.nn.functional as F
import l_model as M0

class ExpertFFN(nn.Module):
    def __init__(s, d, width, K_spike):
        super().__init__(); s.f1 = nn.Linear(d, width); s.f2 = nn.Linear(width, d); s.theta2 = nn.Parameter(torch.full((width,), 0.5)); s.K = K_spike
    def forward(s, x):
        th2 = s.theta2.abs().clamp(min=0.05); cnt = M0.spikes(F.relu(s.f1(x)) / th2, 0, s.K)
        return s.f2(cnt * th2), cnt

class MoE(nn.Module):
    def __init__(s, d, total_width, n_exp, mode, K_spike, seg=64):
        super().__init__(); s.mode, s.E, s.seg = mode, n_exp, seg
        s.experts = nn.ModuleList([ExpertFFN(d, total_width // n_exp, K_spike) for _ in range(n_exp)]); s.router = nn.Linear(d, n_exp)
        s.aux = torch.zeros(()); s.last_choice = None; s.rate = torch.zeros(())
    def forward(s, x):
        B, T, D = x.shape
        if s.mode == 'token': logits = s.router(x)                                           # (B,T,E)
        else:
            nseg = (T + s.seg - 1) // s.seg; pad = nseg * s.seg - T
            xm = F.pad(x, (0, 0, 0, pad)).view(B, nseg, s.seg, D).mean(2)                     # segment means
            prev = torch.cat([torch.zeros_like(xm[:, :1]), xm[:, :-1]], 1)                    # causal: decide from the previous segment
            logits = s.router(prev).repeat_interleave(s.seg, 1)[:, :T]
        p = logits.softmax(-1); choice = p.argmax(-1); s.last_choice = choice
        out = torch.zeros_like(x); fired = torch.zeros((), device=x.device)
        for e, ex in enumerate(s.experts):
            msk = choice == e
            if msk.any():
                y, cnt = ex(x[msk]); out[msk] = y * p[msk][:, e:e + 1]; fired = fired + (cnt != 0).float().sum()
        frac = F.one_hot(choice, s.E).float().mean((0, 1)); s.aux = s.E * (frac * p.mean((0, 1))).sum()   # Switch load-balancing loss
        s.rate = fired / (B * T * s.experts[0].f1.out_features)
        return out

def install(model, n_exp, mode, total_width=None):
    """replace each layer's spiking FFN by a MoE with the same TOTAL feed-forward width (equal total params)."""
    for l in model.layers:
        W = total_width or l.f1.out_features
        l.moe = MoE(l.f1.weight.size(1), W, n_exp, mode, l.K).to(l.f1.weight.device)
        del l.f1, l.f2, l.theta2
    return model
