"""[B1b variant: adds pseudo-rehearsal KD conditions, no stored A examples] Continual-learning suite (vault note 130, experiment B1): does a model forget task A while learning task B, and does hippocampus-style replay help?
Usage: python cont_suite.py MODEL SEED [STEPS_A] [STEPS_B]   MODEL: L5 | T
Task A: in-context key->value recall, keys from letters a-m, values are digits.  Task B: keys from letters n-z, values are letters a-j.  4 pairs per episode.
Train A, then branch the SAME A-trained weights into conditions and train each on B:
  none    sequential fine-tune on B only (catastrophic-forgetting baseline)
  replay  B batches mix in 25% task-A episodes drawn from a small fixed buffer (64 episodes), the hippocampal-replay analogue
Reported (whole-answer accuracy, chance 10% for A and ~10% for B): A after A, then for each condition A after B and B after B. MEASURED.
LCK flag (default off) is not used here; runs are short."""
import os, sys, json, time, copy, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]; os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem
MODEL, SEED = sys.argv[1], int(sys.argv[2]); SA = int(sys.argv[3]) if len(sys.argv) > 3 else 2500; SB = int(sys.argv[4]) if len(sys.argv) > 4 else 2500
torch.manual_seed(SEED); torch.set_num_threads(int(os.environ.get('THREADS', 1))); DEV = 'cpu'
LET = [ord(c) for c in 'abcdefghijklmnopqrstuvwxyz']; DIG = [ord(c) for c in '0123456789']
def batch(task, B, lo=2, hi=4):
    X, M = [], []
    for _ in range(B):
        n = int(torch.randint(lo, hi + 1, (1,))); ks = LET[:13] if task == 'A' else LET[13:]
        idx = torch.randperm(169)[:n].tolist(); keys = [(ks[i // 13], ks[i % 13]) for i in idx]
        vals = [DIG[v] if task == 'A' else LET[v] for v in torch.randint(0, 10, (n,)).tolist()]; s = []
        for (a, b), v in zip(keys, vals): s += [a, b, ord(':'), v, ord(';')]
        q = int(torch.randint(0, n, (1,))); a, b = keys[q]; s += [ord('?'), a, b, ord('='), vals[q]]; X.append(s); m = [0] * len(s); m[-1] = 1; M.append(m)
    return torch.tensor(X), torch.tensor(M)
def new_model():
    return M5.L5Model(d=128, n_layers=4, heads=4) if MODEL == 'L5' else M0.TModel(d=128, n_layers=4, h=4, T=1024)
def fwd(m, x): return m(x, fastmem.hippo_batch(x))[0] if MODEL == 'L5' else m(x)[0]
@torch.no_grad()
def acc(m, task):
    m.eval(); X, Mk = batch(task, 64, 4, 4); lg = fwd(m, X[:, :-1]); ok = (lg.argmax(-1) == X[:, 1:]).float(); mk = Mk[:, 1:].float(); return ((ok * mk).sum(1) == mk.sum(1)).float().mean().item()
def train(m, steps, task, buf=None, rfrac=0.0, lr=2e-3):
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for it in range(steps):
        m.train(); X, Mk = batch(task, 32, 4, 4)
        if buf is not None and rfrac > 0:
            nr = int(32 * rfrac); j = torch.randint(0, buf[0].size(0), (nr,)); X = torch.cat([X[nr:], buf[0][j]]) if X.size(1) == buf[0].size(1) else X
            Mk = torch.cat([Mk[nr:], buf[1][j]]) if Mk.size(1) == buf[1].size(1) else Mk
        lg = fwd(m, X[:, :-1]); mk = Mk[:, 1:].float(); loss = (F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1), reduction='none') * mk.reshape(-1)).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if (it + 1) % max(1, steps // 4) == 0: print(task, it + 1, round(loss.item(), 4), flush=True)

CK = os.path.join(os.environ.get('LOUT', 'results'), f'contA_{MODEL}_s{SEED}.pt'); os.makedirs(os.environ.get('LOUT', 'results'), exist_ok=True)
FMT_KEYS = LET[:13]
def noise_batch(n=32, L=24): return torch.randint(0, 256, (n, L))
def fmt_batch(n=32):   # A-format grammar prior with RANDOM content (leaks the task format; labelled as such)
    X = []
    for _ in range(n):
        s = []
        for _ in range(4): s += [FMT_KEYS[int(torch.randint(0, 13, (1,)))], FMT_KEYS[int(torch.randint(0, 13, (1,)))], ord(':'), DIG[int(torch.randint(0, 10, (1,)))], ord(';')]
        s += [ord('?'), FMT_KEYS[int(torch.randint(0, 13, (1,)))], FMT_KEYS[int(torch.randint(0, 13, (1,)))], ord('='), DIG[int(torch.randint(0, 10, (1,)))]]; X.append(s)
    return torch.tensor(X)
@torch.no_grad()
def self_pool(teacher, n=256, L=24):   # sequences sampled from the A-trained model itself starting from one random byte
    out = []
    for _ in range(n // 32):
        x = torch.randint(0, 256, (32, 1)); teacher.eval()
        for _ in range(L - 1): lg = fwd(teacher, x)[:, -1]; x = torch.cat([x, torch.multinomial(lg.softmax(-1), 1)], 1)
        out.append(x)
    return torch.cat(out)
def fisher(t, n=40):
    t.train(); [p.requires_grad_(True) for p in t.parameters()]; Fi = [torch.zeros_like(p) for p in t.parameters()]
    for _ in range(n):
        X, Mk = batch('A', 16, 4, 4); lg = fwd(t, X[:, :-1]); mk = Mk[:, 1:].float(); l = (F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1), reduction='none') * mk.reshape(-1)).sum() / mk.sum().clamp(min=1); t.zero_grad(); l.backward()
        for f, p in zip(Fi, t.parameters()):
            if p.grad is not None: f += p.grad.detach() ** 2 / n
    t.zero_grad(); return Fi
UNI = None
def unigram_A(n=64):   # 256 byte counts of task A (the only thing kept from A: input statistics, no episodes)
    c = torch.zeros(256)
    for _ in range(n): X, _ = batch('A', 32, 4, 4); c += torch.bincount(X.flatten(), minlength=256).float()
    return (c + 1e-3) / c.sum()
def train_kd(m, teacher, steps, kind, pool=None, lam=float(os.environ.get('LAM', 1.0)), lr=float(os.environ.get('BLR', 2e-3))):
    global UNI
    if kind == 'kduni' and UNI is None: UNI = unigram_A()
    FI = fisher(copy.deepcopy(teacher)) if kind in ('ewc', 'ewcbin') else None
    AR = int(os.environ.get('ADAPTER', 0)); ps_ = list(m.parameters())
    if AR:   # frozen slow weights + small low-rank adapters (new 'plastic' synapses) on every Linear; only adapters learn B
        for p_ in m.parameters(): p_.requires_grad_(False)
        ps_ = []
        for mod in list(m.modules()):
            if isinstance(mod, torch.nn.Linear):
                o_, i_ = mod.weight.shape; A_ = torch.nn.Parameter(torch.randn(AR, i_) / i_ ** 0.5); B_ = torch.nn.Parameter(torch.zeros(o_, AR)); mod.adA, mod.adB = A_, B_; ps_ += [A_, B_]
                mod.register_forward_hook(lambda md, inp, out: out + (inp[0] @ md.adA.t()) @ md.adB.t())
        print('adapter params', sum(p.numel() for p in ps_), flush=True)
    if kind == 'ewcbin':
        allf = torch.cat([f.flatten() for f in FI]); thr = torch.quantile(allf[torch.randperm(allf.numel())[:200000]], 1 - float(os.environ.get('FROZEN', 0.3))); FM = [(f < thr).float() for f in FI]   # 1 = free to change, 0 = frozen (1-bit importance mask)
    p0 = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.AdamW(ps_, lr, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for it in range(steps):
        m.train(); X, Mk = batch('B', 32, 4, 4)
        lg = fwd(m, X[:, :-1]); mk = Mk[:, 1:].float(); loss = (F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1), reduction='none') * mk.reshape(-1)).sum() / mk.sum().clamp(min=1)
        if kind == 'ewcbin':
            opt.zero_grad(); loss.backward()
            for p_, mk_ in zip(m.parameters(), FM):
                if p_.grad is not None: p_.grad.mul_(mk_)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); continue
        if kind in ('l2anchor', 'ewc'):
            loss = loss + (float(os.environ.get('L2LAM', 1.0)) * sum(((p - q) ** 2).sum() for p, q in zip(m.parameters(), p0)) if kind == 'l2anchor' else float(os.environ.get('EWCLAM', 100.0)) * sum((f * (p - q) ** 2).sum() for f, p, q in zip(FI, m.parameters(), p0))); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); continue
        P = (torch.multinomial(UNI, 32 * 24, replacement=True).view(32, 24)) if kind == 'kduni' else noise_batch() if kind == 'kdnoise' else (batch('B', 32, 4, 4)[0] if kind == 'kdb' else (fmt_batch() if kind == 'kdfmt' else pool[torch.randint(0, pool.size(0), (32,))]))
        with torch.no_grad(): tl = fwd(teacher, P[:, :-1])
        sl = fwd(m, P[:, :-1]); kd = F.kl_div(sl.log_softmax(-1).reshape(-1, 256), tl.softmax(-1).reshape(-1, 256), reduction='batchmean')
        loss = loss + lam * kd
        if os.environ.get('RBUF'):
            j = torch.randint(0, BX8.size(0), (8,)); lr_ = fwd(m, BX8[j][:, :-1]); mr = BM8[j][:, 1:].float(); loss = loss + float(os.environ['RBUF']) * (F.cross_entropy(lr_.reshape(-1, lr_.size(-1)), BX8[j][:, 1:].reshape(-1), reduction='none') * mr.reshape(-1)).sum() / mr.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if (it + 1) % max(1, steps // 4) == 0: print(kind, it + 1, round(loss.item(), 4), round(kd.item(), 4), flush=True)
t0 = time.time(); m = new_model(); R = {'model': MODEL, 'seed': SEED, 'stepsA': SA, 'stepsB': SB, 'params': M0.nparams(m)}
if os.path.exists(CK): m.load_state_dict(torch.load(CK)); print('loaded A ckpt', flush=True)
else: train(m, SA, 'A'); torch.save(m.state_dict(), CK)
def acc_fixed(mm_, task, n=16):   # fixed eval set (same episodes before/after), 16x64 = 1024 episodes, lower noise
    with torch.random.fork_rng():
        torch.manual_seed(424242 + (0 if task == 'A' else 1)); return round(sum(acc(mm_, task) for _ in range(n)) / n, 4)
R['A_after_A'] = round(sum(acc(m, 'A') for _ in range(4)) / 4, 4); R['A_after_A_fixed1024'] = acc_fixed(m, 'A')
teacher = copy.deepcopy(m).eval()
for p_ in teacher.parameters(): p_.requires_grad_(False)
pool = self_pool(teacher)
torch.manual_seed(SEED + 99); BX8, BM8 = batch('A', 8, 4, 4)
OUT = os.environ.get('LOUT', 'results'); fn = os.path.join(OUT, f'cont2_{MODEL}_s{SEED}' + os.environ.get('FTAG', '') + '.json')
for cond in os.environ.get('CONDS', 'none,kdnoise,kdself,kdfmt').split(','):
    cond0 = cond; cond = cond.split('@')[0]
    mm = copy.deepcopy(m); torch.manual_seed(SEED + 7)
    if cond == 'none': train(mm, SB, 'B')
    elif cond == 'replay8': b8 = (BX8, BM8); train(mm, SB, 'B', b8, 0.25)
    else: train_kd(mm, teacher, SB, cond, pool)
    R[cond0 + '_fixed1024'] = {'A_after_B': acc_fixed(mm, 'A'), 'B_after_B': acc_fixed(mm, 'B')}
    R[cond0] = {'A_after_B': round(sum(acc(mm, 'A') for _ in range(4)) / 4, 4), 'B_after_B': round(sum(acc(mm, 'B') for _ in range(4)) / 4, 4)}
    R['secs'] = round(time.time() - t0); print(json.dumps(R), flush=True); json.dump(R, open(fn, 'w'))
