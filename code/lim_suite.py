"""Limitation test suite (transformer-limitations map T13/T14/T15/T19/T54): L5 vs matched transformer, same params class, same data, 3 seeds.
Usage: python lim_suite.py MODEL TASK SEED [STEPS]   MODEL: L5 | T      TASK: kv | kvupd | hop2 | reverse
 kv      in-context key->value recall; train on 4-8 pairs, test on 8..128 pairs (length extrapolation, T15/T14)
 kvupd   same key rewritten several times, query the LATEST value; train 4-8 events, test up to 128 (T24/T25, stale-history reuse)
 hop2    two-hop in-context lookup (k1->k2, k2->v) with distractors; train <=4 chains, test up to 32 (T19)
 reverse weight-memorised facts 'AB>CD;' trained forward only; test reverse direction 'CD>' -> 'AB' (T54, reversal curse)
All MEASURED accuracy; loss only on the answer tokens. Results -> LOUT/lim_<MODEL>_<TASK>_s<seed>.json"""
import os, sys, json, math, time, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]; os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem
MODEL, TASK, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]); STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
torch.manual_seed(SEED); torch.set_num_threads(int(os.environ.get('THREADS', 2))); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
g = torch.Generator().manual_seed(SEED + 12345)
L = lambda s: [ord(c) for c in s]; LET = L('abcdefghijklmnopqrstuvwxyz'); DIG = L('0123456789')
def key(n):   # n distinct 2-letter keys
    idx = torch.randperm(676)[:n]; return [(LET[i // 26], LET[i % 26]) for i in idx.tolist()]
def kv_batch(B, npairs):
    X, M = [], []
    for _ in range(B):
        ks = key(npairs); vs = torch.randint(0, 10, (npairs,)).tolist(); s = []
        for (a, b), v in zip(ks, vs): s += [a, b, ord(':'), DIG[v], ord(';')]
        q = torch.randint(0, npairs, (1,)).item(); a, b = ks[q]; s += [ord('?'), a, b, ord('='), DIG[vs[q]]]; X.append(s); m = [0] * len(s); m[-1] = 1; M.append(m)
    return torch.tensor(X), torch.tensor(M)
def kvupd_batch(B, nev):
    # keys are re-assigned over time; the query asks for the MOST RECENT value of a key that was written >=2 times (T24 conflicting memory / stale-history reuse)
    X, M = [], []
    for _ in range(B):
        pool = key(max(2, nev // 2)); evs = [(pool[int(torch.randint(0, len(pool), (1,)))], int(torch.randint(0, 10, (1,)))) for _ in range(nev)]
        cnt = {}; [cnt.__setitem__(k, cnt.get(k, 0) + 1) for k, _ in evs]; multi = [k for k, c in cnt.items() if c >= 2] or [evs[0][0]]
        qk = multi[int(torch.randint(0, len(multi), (1,)))]; ans = [v for k, v in evs if k == qk][-1]; s = []
        for (a, b), v in evs: s += [a, b, ord(':'), DIG[v], ord(';')]
        s += [ord('?'), qk[0], qk[1], ord('='), DIG[ans]]; X.append(s); m = [0] * len(s); m[-1] = 1; M.append(m)
    return torch.tensor(X), torch.tensor(M)
def hop2_batch(B, nch):
    X, M = [], []
    for _ in range(B):
        ks = key(2 * nch); vs = torch.randint(0, 10, (nch,)).tolist(); facts = []
        for c in range(nch):
            k1, k2 = ks[2 * c], ks[2 * c + 1]; facts.append([k1[0], k1[1], ord(':'), k2[0], k2[1], ord(';')]); facts.append([k2[0], k2[1], ord(':'), DIG[vs[c]], ord(';')])
        perm = torch.randperm(len(facts)).tolist(); s = [t for i in perm for t in facts[i]]; c = torch.randint(0, nch, (1,)).item(); k1 = ks[2 * c]
        s += [ord('>'), k1[0], k1[1], ord('='), DIG[vs[c]]]; X.append(s); m = [0] * len(s); m[-1] = 1; M.append(m)
    return torch.tensor(X), torch.tensor(M)
# reversal: fixed fact table, same for all seeds of a run
NF = 200; pk = key(2 * NF); FACT = [(pk[2 * i], pk[2 * i + 1]) for i in range(NF)]
def rev_train(B):   # forward statements only; loss on the 2-letter object
    X, M = [], []
    for _ in range(B):
        s, m = [], []
        for i in torch.randint(0, NF, (6,)).tolist():
            (a, b), (c, d) = FACT[i]; s += [a, b, ord('>'), c, d, ord(';')]; m += [0, 0, 0, 1, 1, 0]
        X.append(s); M.append(m)
    return torch.tensor(X), torch.tensor(M)
def new_model():
    if MODEL == 'L5': return M5.L5Model(d=128, n_layers=4, heads=4)
    return M0.TModel(d=128, n_layers=4, h=4, T=1024)
m = new_model().to(DEV)
def fwd(x):
    return m(x, fastmem.hippo_batch(x))[0] if MODEL == 'L5' else m(x)[0]
def train_batch():
    if TASK == 'kv': return kv_batch(32, int(torch.randint(4, 9, (1,))))
    if TASK == 'hop2': return hop2_batch(32, int(torch.randint(1, 5, (1,))))
    if TASK == 'kvupd': return kvupd_batch(32, int(torch.randint(4, 9, (1,))))
    return rev_train(32)
LR0 = float(os.environ.get('LR', 0)) or (2e-3 if TASK != 'reverse' else 3e-3)   # LR flag: default off = original rates
opt = torch.optim.AdamW(m.parameters(), LR0, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, opt.param_groups[0]['lr'], total_steps=STEPS, pct_start=0.1)
t0 = time.time(); print(MODEL, TASK, 'params', M0.nparams(m), flush=True)
CKD = os.environ.get('LCK', ''); CKF = os.path.join(CKD, f'ck_{MODEL}_{TASK}_s{SEED}.pt') if CKD else ''; it0 = 0   # LCK flag (default off): checkpoint/resume every 100 steps
if CKF and os.path.exists(CKF):
    c_ = torch.load(CKF, map_location=DEV); m.load_state_dict(c_['m']); opt.load_state_dict(c_['o']); sch.load_state_dict(c_['s']); it0 = c_['it']; torch.manual_seed(SEED * 100003 + it0); print('RESUMED at', it0, flush=True)
for it in range(it0, STEPS):
    m.train(); X, Mk = train_batch(); X, Mk = X.to(DEV), Mk.to(DEV); lg = fwd(X[:, :-1]); tgt = X[:, 1:]; mk = Mk[:, 1:].float()
    loss = (F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1), reduction='none') * mk.reshape(-1)).sum() / mk.sum().clamp(min=1)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    if CKF and (it + 1) % 100 == 0: torch.save({'m': m.state_dict(), 'o': opt.state_dict(), 's': sch.state_dict(), 'it': it + 1}, CKF + '.tmp'); os.replace(CKF + '.tmp', CKF)
    if (it + 1) % max(1, STEPS // 6) == 0: print(it + 1, 'loss', round(loss.item(), 4), f'{time.time()-t0:.0f}s', flush=True)
m.eval(); R = {'model': MODEL, 'task': TASK, 'seed': SEED, 'steps': STEPS, 'params': M0.nparams(m), 'acc': {}}
@torch.no_grad()
def acc(X, Mk):
    X, Mk = X.to(DEV), Mk.to(DEV); lg = fwd(X[:, :-1]); pred = lg.argmax(-1); ok = (pred == X[:, 1:]).float(); mk = Mk[:, 1:].float()
    return ((ok * mk).sum(1) == mk.sum(1)).float().mean().item()   # whole answer correct
if TASK == 'kv':
    for n in (4, 8, 16, 32, 64, 128): R['acc'][f'N{n}'] = round(sum(acc(*kv_batch(16, n)) for _ in range(4)) / 4, 4)
elif TASK == 'kvupd':
    for n in (4, 8, 16, 32, 64, 128): R['acc'][f'N{n}'] = round(sum(acc(*kvupd_batch(16, n)) for _ in range(4)) / 4, 4)
elif TASK == 'hop2':
    for n in (1, 2, 4, 8, 16, 32): R['acc'][f'chains{n}'] = round(sum(acc(*hop2_batch(16, n)) for _ in range(4)) / 4, 4)
else:
    f, r = [], []
    for i in range(NF):
        (a, b), (c, d) = FACT[i]; f.append([a, b, ord('>'), c, d]); r.append([c, d, ord('>'), a, b])
    mk = torch.tensor([[0, 0, 0, 1, 1]] * NF)
    R['acc']['forward'] = round(acc(torch.tensor(f), mk), 4); R['acc']['reverse'] = round(acc(torch.tensor(r), mk), 4)
print(json.dumps(R), flush=True); OUT = os.environ.get('LOUT', 'results'); os.makedirs(OUT, exist_ok=True)
json.dump(R, open(os.path.join(OUT, f'lim_{MODEL}_{TASK}_s{SEED}.json'), 'w'))
