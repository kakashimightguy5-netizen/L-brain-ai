"""L-pure byte LM test (note 149/151): mixing question isolated. No memory block / no hippocampus here (loop too slow, separate test).
Variants: MIX=sparse (StructTern wiring, K) | dense (nn.Linear control). Same width/depth. bits/byte on the fixed 6x1024 eval windows of the project (second half of py_big_val)."""
import os, sys, math, json, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from lpure import StructTern, round_ste, tern_ste, pow2_ste
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MIX = os.environ.get('MIX', 'sparse'); K = int(os.environ.get('K', 32)); D = int(os.environ.get('D', 128)); NL = int(os.environ.get('NL', 4)); STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
QUANT = int(os.environ.get('QUANT', 1)); LEVELS = int(os.environ.get('LEVELS', 1)); SEED = int(os.environ.get('SEED', 0)); RAMP = float(os.environ.get('RAMP', 0.5)); TAG = os.environ.get('TAG', f'lp_{MIX}_K{K}_d{D}_s{SEED}'); BS, T = 16, 256; LR = float(os.environ.get('LR', 2e-3))
torch.manual_seed(SEED)
SCHED = {'frac': 0.0, 'alpha': 1.0}                          # gradual-pruning progress 0..1, set by the training loop
class MaskTern(nn.Module):
    """GRADUAL PRUNING (fix for note 148): start dense, anneal each row down to K connections by magnitude while training; ternary weights + pow2 gain.
    Training = masked dense matmul (GPU-fast); deployed = K adds per output (sparse list). Mask is re-derived from |w| every step (connections can move)."""
    def __init__(s, din, dout, Kf, g0=-1.0):
        super().__init__(); s.w = nn.Parameter(torch.randn(dout, din) / math.sqrt(din)); s.logg = nn.Parameter(torch.full((dout,), float(g0))); s.din, s.dout, s.Kf = din, dout, min(Kf, din)
    def k_now(s): return max(s.Kf, int(round(s.din + (s.Kf - s.din) * min(1.0, SCHED['frac']))))
    def forward(s, x):
        k = s.k_now(); w = s.w
        if k < s.din:
            th = w.abs().topk(k, dim=1).values[:, -1:]; w = w * (w.abs() >= th)
        al = SCHED['alpha']; sc = w.abs().sum(1, keepdim=True).clamp(min=1e-8) / max(k, 1); q = torch.clamp(torch.round(w / sc), -1, 1)
        for i_ in range(1, LEVELS): q = q + 0.5 ** i_ * torch.clamp(torch.round((w / sc - q) / 0.5 ** i_), -1, 1)      # 2nd ternary term at half scale: weight levels {-1.5..1.5} step 0.5 = one add + one shifted add
        t = w / sc + al * (q - w / sc).detach()      # QUANT RAMP: alpha 0 = float, 1 = ternary
        lg = s.logg - 0.5 * math.log2(max(k, 2)); lg = lg + al * (torch.round(lg) - lg).detach()
        return F.linear(x, t) * torch.exp2(lg)      # gain follows the CURRENT fan-in (sqrt(k) normalisation as a shift)
    def adds(s): return s.dout * s.Kf * LEVELS      # MODELLED upper bound: each extra level = one more shifted add per connection
def make_mk(mix): return (lambda a, b: nn.Linear(a, b, bias=False)) if mix == 'dense' else (lambda a, b: MaskTern(a, b, K)) if mix == 'mask' else (lambda a, b: StructTern(a, b, K, seed=a * 3 + b))
class LeakScan(nn.Module):
    """h_t = a h_{t-1} + u_t, a = 1 - 2^-shift (power-of-two leak), computed in parallel as a causal kernel a^(t-s). Then spike count with pow2 threshold."""
    def __init__(s, d, Ksp=8):
        super().__init__(); s.sh = nn.Parameter(torch.linspace(1.0, 6.5, d)); s.p = nn.Parameter(torch.zeros(d)); s.Ksp = Ksp
    def forward(s, u):
        B, Tn, Dd = u.shape; a = 1 - torch.exp2(-round_ste(s.sh).clamp(1.0, 7.0)); la = torch.log(a)
        ar = torch.arange(Tn, device=u.device); dt = (ar[:, None] - ar[None]).float(); mask = dt >= 0; dtc = dt.clamp(min=0); cs = 16 if Tn > 512 else Dd
        hs = []
        for c0 in range(0, Dd, cs):
            Km = torch.exp(la[c0:c0 + cs, None, None] * dtc[None]) * mask[None]                        # (cs, T, T)
            hs.append(torch.einsum('bsc,cts->btc', u[..., c0:c0 + cs], Km))
        h = torch.cat(hs, -1); th = torch.exp2(round_ste(s.p))
        return torch.clamp(round_ste(h / th), -s.Ksp, s.Ksp) * th
class Taps(nn.Module):
    """causal 4-tap delay line with ternary taps (add-only)."""
    def __init__(s, d, n=4):
        super().__init__(); s.w = nn.Parameter(torch.randn(d, n) * 0.5); s.logg = nn.Parameter(torch.zeros(d)); s.n = n
    def forward(s, x):
        t = tern_ste(s.w) * pow2_ste(s.logg)[:, None]; xp = F.pad(x, (0, 0, s.n - 1, 0)); return sum(xp[:, s.n - 1 - k: xp.size(1) - k] * t[:, k] for k in range(s.n))
class Blk(nn.Module):
    def __init__(s, d, mk):
        super().__init__(); s.inp, s.out = mk(d, d), mk(d, d); s.sc = LeakScan(d); s.taps = Taps(d); s.f1, s.f2 = mk(d, 2 * d), mk(2 * d, d)
    def forward(s, x):
        x = x + s.out(s.sc(s.inp(x))) + s.taps(x)
        return x + s.f2(torch.clamp(round_ste(F.relu(s.f1(x))), 0, 8))
class LM(nn.Module):
    def __init__(s, mix=None, d=None):
        mix = mix or MIX; d = d or D; mk = make_mk(mix)
        super().__init__(); s.emb = nn.Embedding(256, d); s.b = nn.ModuleList([Blk(d, mk) for _ in range(NL)])
        s.head = nn.Linear(d, 256, bias=False) if mix == 'dense' else MaskTern(d, 256, K * 2, g0=-4.0) if mix == 'mask' else StructTern(d, 256, K * 2, seed=9)
        s.n = nn.ModuleList([nn.LayerNorm(d) for _ in range(NL)])
    def forward(s, x):
        z = s.emb(x)
        for b, n in zip(s.b, s.n): z = z + (b(n(z)) - n(z))
        return s.head(z)
if __name__ == '__main__':
    DATA = os.environ.get('LDATA', '/home/claude/data'); ld = lambda p: torch.tensor(list(open(p, 'rb').read()), dtype=torch.long)
    tr, va = ld(os.path.join(DATA, 'py_big_train.txt')), ld(os.path.join(DATA, 'py_big_val.txt')); half = len(va) // 2
    X = torch.stack([va[half + i * 1024: half + i * 1024 + 1025] for i in range(6)]).to(DEV)
    TCH = os.environ.get('TEACHER', ''); KDW = float(os.environ.get('KDW', 0.5)) if TCH else 0.0; tch = None
    if TCH:
        tch = LM('dense', int(os.environ.get('TEACHER_D', D))).to(DEV); tch.load_state_dict(torch.load(os.path.join(os.environ.get('LOUT', '.'), TCH), map_location=DEV)); tch.eval()
        for p_ in tch.parameters(): p_.requires_grad_(False)
    m = LM().to(DEV); npar = sum(p.numel() for p in m.parameters())
    adds = sum(x.adds() for x in m.modules() if isinstance(x, (StructTern, MaskTern))) if MIX != 'dense' else None
    print(TAG, 'params', npar, 'adds/token', adds, flush=True)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    def bpb():
        m.eval()
        with torch.no_grad(): lg = m(X[:, :-1]); r = F.cross_entropy(lg.reshape(-1, 256), X[:, 1:].reshape(-1)).item() / math.log(2)
        m.train(); return r
    t0 = time.time(); log = []
    for it in range(STEPS):
        SCHED['frac'] = it / (0.6 * STEPS); SCHED['alpha'] = 0.0 if QUANT == 0 else (1.0 if RAMP <= 0 else min(1.0, it / (RAMP * STEPS))); ix = torch.randint(0, len(tr) - T - 1, (BS,)); x = torch.stack([tr[i:i + T + 1] for i in ix]).to(DEV)
        lg = m(x[:, :-1]); loss = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1))
        if tch is not None:
            with torch.no_grad(): tl = tch(x[:, :-1])
            loss = (1 - KDW) * loss + KDW * F.kl_div(lg.log_softmax(-1).reshape(-1, 256), tl.softmax(-1).reshape(-1, 256), reduction='batchmean')
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if (it + 1) % max(1, STEPS // 6) == 0 or it == STEPS - 1:
            b = bpb(); log.append((it + 1, b)); print(TAG, 'step', it + 1, 'train_loss_bits', round(loss.item() / math.log(2), 3), 'val_bpb', round(b, 4), f'{time.time() - t0:.0f}s', flush=True)
    os.makedirs(os.environ.get('LOUT', '.'), exist_ok=True); torch.save(m.state_dict(), os.path.join(os.environ.get('LOUT', '.'), TAG + '.pt'))
    json.dump({'tag': TAG, 'mix': MIX, 'K': K, 'D': D, 'NL': NL, 'steps': STEPS, 'seed': SEED, 'params': npar, 'adds_per_token': adds, 'final_bpb': log[-1][1], 'log': log, 'teacher': TCH, 'sec_per_step': (time.time() - t0) / STEPS, 'device': DEV}, open(os.path.join(os.environ.get('LOUT', '.'), TAG + '.json'), 'w'))
