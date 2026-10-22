"""NOMAT: sparse ternary wiring - replaces dense matrices (no matrix multiply, adds only in the deployed forward)."""
import torch, torch.nn as nn
# ---- NOMAT: sparse ternary wiring (no dense matrix, no multiply in the deployed forward) ----
class SparseTernLinear(nn.Module):
    """Each output reads only K fixed inputs (chosen by |w| from a trained dense layer), with weight in {-1,0,+1}; per-output gain g.
    y_j = g_j * sum_{k<K} t_jk * x[idx_jk] + b_j. Deployed: K gathers + adds per output (no matrix multiply). Training: gather + STE.
    Cost per token: out*K adds (vs out*in MACs)."""
    def __init__(s, lin, K):
        super().__init__(); W = lin.weight.data; o, i = W.shape; K = min(K, i); s.in_features, s.out_features, s.K = i, o, K
        idx = W.abs().topk(K, dim=1).indices; s.register_buffer('idx', idx)
        w = W.gather(1, idx); s.w = nn.Parameter(w.clone()); s.g = nn.Parameter(w.abs().mean(1, keepdim=True).clamp(min=1e-6))
        s.bias = nn.Parameter(lin.bias.data.clone()) if lin.bias is not None else None
    def forward(s, x):
        sc = s.w.abs().mean(1, keepdim=True).clamp(min=1e-8); t = torch.clamp(torch.round(s.w / sc), -1, 1); t = s.w / sc + (t - s.w / sc).detach()
        xg = x[..., s.idx]                                  # (..., out, K) gather
        y = (xg * t).sum(-1) * s.g.squeeze(1)
        return y + s.bias if s.bias is not None else y
def apply_sparse_tern(model, K, names=None):
    """replace nn.Linear named in `names` (default: every Linear except the output head) with SparseTernLinear. Returns (adds_before_MACs, adds_after)."""
    before = after = 0
    for parent in list(model.modules()):
        for nm, ch in list(parent.named_children()):
            if isinstance(ch, nn.Linear) and ch.weight.dim() == 2 and (names is None and nm != 'head' or names is not None and nm in names):
                o, i = ch.weight.shape; new = SparseTernLinear(ch, K); setattr(parent, nm, new); before += o * i; after += o * new.K
    return before, after
