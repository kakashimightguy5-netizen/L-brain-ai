"""Hardware-oriented training flags (all OFF by default; reference path unchanged).
QAT (H6): fake-quantised weights with a straight-through estimator, registered as a parametrization on every nn.Linear:
  int8      : per-output-channel symmetric int8, power-of-two scale
  int4      : per-output-channel symmetric int4 (-8..7), power-of-two scale
  mx4_pow2  : 4-bit ints, ONE power-of-two (shift) scale per block of B consecutive input weights (MXFP4-like microscaling)
  mx4_float : same blocks, free float scale (reference)
  ternary   : BitNet-b1.58-style weights in {-1, 0, +1} x per-row absmean scale -> a matrix-vector product is additions only;
              with L's integer spike counts as inputs, the spike-driven projections need NO multiplier at all.
LOWRANK (fix for L's big dense matrices): replace the DENSE projections (memory qkv, o; membrane inp; FFN f1) by a rank-r
factorisation W ~ U V (r = frac x min(in, out)). The spike-driven projections (out, f2) are left alone (already event-driven)."""
import math, torch, torch.nn as nn
import torch.nn.utils.parametrize as P

def _pow2(s): return torch.exp2(torch.ceil(torch.log2(s.clamp(min=1e-12))))
class FakeQuant(nn.Module):
    def __init__(s, mode, B=32):
        super().__init__(); s.mode, s.B = mode, B; s.alpha = 1.0     # alpha<1 = progressive quantisation (blend), default 1 = unchanged
    def quant(s, w):
        if s.mode == 'ternary':
            sc = w.abs().mean(1, keepdim=True).clamp(min=1e-8); return torch.clamp(torch.round(w / sc), -1, 1) * sc
        if s.mode in ('int8', 'int4'):
            qmax = 127 if s.mode == 'int8' else 7
            sc = _pow2(w.abs().amax(1, keepdim=True) / qmax); return torch.clamp(torch.round(w / sc), -qmax - 1, qmax) * sc
        out, inn = w.shape; B = s.B; pad = (-inn) % B
        wp = torch.nn.functional.pad(w, (0, pad)).view(out, -1, B)
        sc = wp.abs().amax(-1, keepdim=True) / 7
        sc = _pow2(sc) if s.mode == 'mx4_pow2' else sc.clamp(min=1e-12)
        q = torch.clamp(torch.round(wp / sc), -8, 7) * sc
        return q.view(out, -1)[:, :inn]
    def forward(s, w): return w + s.alpha * (s.quant(w) - w).detach()          # straight-through estimator
def apply_qat(model, mode, B=32, skip=()):
    n = 0
    for nm, mod in model.named_modules():
        if skip and nm.split('.')[-1] in skip: continue          # QAT_SKIP: keep e.g. head/trust in full precision
        if isinstance(mod, nn.Linear): P.register_parametrization(mod, 'weight', FakeQuant(mode, B)); n += 1
    return n
def storage_bits(model, mode, B=32):
    """weight storage incl. scales (5-bit shift per scale for pow2, 16-bit for float scales; int8/int4 one scale per row)."""
    bits = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            o, i = mod.weight.shape
            if mode == 'none': bits += o * i * 32
            elif mode == 'ternary': bits += o * i * 2 + o * 16          # 2-bit packing (1.58 bit entropy) + fp16 row scale
            elif mode == 'int8': bits += o * i * 8 + o * 5
            elif mode == 'int4': bits += o * i * 4 + o * 5
            else: bits += o * i * 4 + o * math.ceil(i / B) * (5 if mode == 'mx4_pow2' else 16)
    return bits

DENSE_NAMES = ('qkv', 'o', 'inp', 'f1')
ALL_NAMES = DENSE_NAMES + ('f2', 'out')   # LOWRANK_ALL=1 also factorises the spike-driven f2/out (default off)
def apply_lowrank(model, frac=0.25, names=None):
    """factorise every DENSE projection of an L model (DeltaMemory.qkv/.o, LLayer.inp/.f1) into two thin matrices.
    Returns (MACs before, MACs after) for one token through those matrices."""
    before = after = 0; names = names or DENSE_NAMES
    for parent in list(model.modules()):
        for name in names:
            lin = getattr(parent, name, None)
            if isinstance(lin, nn.Linear) and lin.weight.dim() == 2:
                o, i = lin.weight.shape; r = max(4, int(round(frac * min(o, i))))
                U = nn.Linear(i, r, bias=False); V = nn.Linear(r, o, bias=lin.bias is not None)
                nn.init.normal_(U.weight, std=i ** -0.5); nn.init.normal_(V.weight, std=r ** -0.5)
                if lin.bias is not None: nn.init.zeros_(V.bias)
                seq = nn.Sequential(U, V).to(lin.weight.device); seq.in_features, seq.out_features = i, o
                setattr(parent, name, seq); before += o * i; after += r * (i + o)
    return before, after

class SpikeIn(nn.Module):
    """DENSE_SPIKE flag: make a dense projection EVENT-DRIVEN by spike-coding its input (integer counts in [-K, K], learned
    step theta, straight-through gradient). Zero counts are skipped MACs. self.rate = mean |count| (energy term),
    self.zero = share of zero inputs (the MAC saving on this matrix)."""
    def __init__(s, lin, theta0=0.5, K=8):
        super().__init__(); import l_model as M0
        s.lin, s.K, s.M0 = lin, K, M0; s.theta = nn.Parameter(torch.tensor(float(theta0))); s.rate = torch.zeros(()); s.zero = 0.0
        s.in_features, s.out_features = getattr(lin, 'in_features', None), getattr(lin, 'out_features', None)
    def forward(s, x):
        th = s.theta.abs().clamp(min=0.05); q = s.M0.spikes(x / th, -s.K, s.K)
        s.rate = q.abs().mean(); s.zero = float((q == 0).float().mean())
        return s.lin(q * th)
def apply_dense_spikes(model, theta0=0.5, K=8):
    """wrap qkv / o (DeltaMemory) and inp / f1 (LLayer) with SpikeIn. Returns the list of wrappers (for the energy term)."""
    out = []
    for parent in list(model.modules()):
        for name in DENSE_NAMES:
            lin = getattr(parent, name, None)
            if isinstance(lin, (nn.Linear, nn.Sequential)) and not isinstance(lin, SpikeIn):
                w = SpikeIn(lin, theta0, K); setattr(parent, name, w); out.append(w)
    model._dense_spk = out; return out

def apply_tie_head(model):
    """TIE_HEAD flag: output head shares the embedding matrix (saves V*d weights). Returns params saved."""
    if getattr(model.head, 'weight', None) is None or model.head.weight.shape != model.emb.weight.shape: return 0
    n = model.head.weight.numel(); model.head.weight = model.emb.weight; return n
def apply_share_layers(model, names=('inp', 'out', 'f1', 'f2')):
    """SHARE_LAYERS flag: every LLayer reuses layer 0's projection matrices (norms/gates/thetas stay per-layer).
    Same FLOPs, (n_layers-1)/n_layers fewer stored weights on those matrices. Returns params saved."""
    L = list(model.layers); saved = 0
    for nm in names:
        base = getattr(L[0], nm)
        for l in L[1:]:
            saved += sum(p.numel() for p in getattr(l, nm).parameters()); setattr(l, nm, base)
    return saved

def apply_qat_map(model, mapping, B=32, skip=()):
    """QAT_MAP flag: per-matrix-group QAT mode, e.g. {'f1':'ternary','f2':'ternary','*':'int4'}. Group = the first path component that is one of ALL_NAMES
    (so factorised f1.0/f1.1 both follow 'f1'). Linears outside those groups use mapping.get('other', 'none'). Returns counts per mode."""
    cnt = {}
    for nm, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or nm.split('.')[-1] in skip: continue
        parts = nm.split('.'); g = next((p for p in parts if p in ALL_NAMES), None)
        mode = mapping.get(g, mapping.get('*', 'none')) if g else mapping.get('other', 'none')
        if mode != 'none': P.register_parametrization(mod, 'weight', FakeQuant(mode, B)); cnt[mode] = cnt.get(mode, 0) + 1
    return cnt


def apply_qat_emb(model, mode, B=32):
    """QAT_EMB flag: fake-quantise nn.Embedding tables (per-row, same quantisers as Linear). Default off. Returns count."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Embedding): P.register_parametrization(mod, 'weight', FakeQuant(mode, B)); n += 1
    return n
