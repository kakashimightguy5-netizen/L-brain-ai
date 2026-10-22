"""Three-task sequence A -> B -> C with statistics-matched dreaming (vault note 138). Does forgetting stay solved when tasks pile up?
Tasks (in-context key->value recall, 4 pairs, whole-answer accuracy, chance 10%):
  A: keys letters a-m, values digits.   B: keys letters n-z, values letters a-j.   C: keys digit pairs, values letters k-t.
Conditions: none (plain sequential) | dream (while learning task k, KL to the frozen model from the end of task k-1 on dream sequences sampled from the
byte histogram of ALL earlier tasks; only 256 numbers per task are kept, no episodes).
Usage: python cont3.py MODEL SEED COND [STEPS_PER_TASK]   MODEL L5|T. MEASURED on a fixed 1024-episode eval per task."""
import os, sys, json, time, copy, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]; os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem
MODEL, SEED, COND = sys.argv[1], int(sys.argv[2]), sys.argv[3]; ST = int(sys.argv[4]) if len(sys.argv) > 4 else 2500
LAM = float(os.environ.get('LAM', 25)); torch.manual_seed(SEED); torch.set_num_threads(int(os.environ.get('THREADS', 1)))
LET = [ord(c) for c in 'abcdefghijklmnopqrstuvwxyz']; DIG = [ord(c) for c in '0123456789']
KEYS = {'A': LET[:13], 'B': LET[13:], 'C': DIG}; VALS = {'A': DIG, 'B': LET[:10], 'C': LET[10:20]}
def batch(task, B, n=4):
    X, M = [], []; ks = KEYS[task]; K = len(ks)
    for _ in range(B):
        idx = torch.randperm(K * K)[:n].tolist(); keys = [(ks[i // K], ks[i % K]) for i in idx]
        vals = [VALS[task][v] for v in torch.randint(0, 10, (n,)).tolist()]; s = []
        for (a, b), v in zip(keys, vals): s += [a, b, ord(':'), v, ord(';')]
        q = int(torch.randint(0, n, (1,))); a, b = keys[q]; s += [ord('?'), a, b, ord('='), vals[q]]; X.append(s); m = [0] * len(s); m[-1] = 1; M.append(m)
    return torch.tensor(X), torch.tensor(M)
m = M5.L5Model(d=128, n_layers=4, heads=4) if MODEL == 'L5' else M0.TModel(d=128, n_layers=4, h=4, T=1024)
def fwd(mm, x): return mm(x, fastmem.hippo_batch(x))[0] if MODEL == 'L5' else mm(x)[0]
@torch.no_grad()
def acc(mm, task, n=16):
    mm.eval(); tot = 0.0
    with torch.random.fork_rng():
        torch.manual_seed(424242 + ord(task))
        for _ in range(n):
            X, Mk = batch(task, 64); lg = fwd(mm, X[:, :-1]); ok = (lg.argmax(-1) == X[:, 1:]).float(); mk = Mk[:, 1:].float(); tot += ((ok * mk).sum(1) == mk.sum(1)).float().mean().item()
    return round(tot / n, 4)
BIG = os.environ.get('DREAM', 'unigram') == 'bigram'   # DREAM flag: unigram (256 numbers/task) | bigram (256x256 transition counts/task, still no episodes)
def hist(task, n=64):
    if BIG:
        c = torch.zeros(256, 256)
        for _ in range(n):
            X, _ = batch(task, 32); a, b = X[:, :-1].flatten(), X[:, 1:].flatten(); c.index_put_((a, b), torch.ones_like(a, dtype=torch.float), accumulate=True)
        return c / c.sum()
    c = torch.zeros(256)
    for _ in range(n): X, _ = batch(task, 32); c += torch.bincount(X.flatten(), minlength=256).float()
    return c / c.sum()
def sample_dream(prior, B=32, L=24):
    if not BIG: return torch.multinomial(prior, B * L, replacement=True).view(B, L)
    rows = prior.sum(1); x = [torch.multinomial(rows, B, replacement=True)]
    T_ = prior / prior.sum(1, keepdim=True).clamp(min=1e-12); T_[rows <= 0] = 1.0 / 256
    for _ in range(L - 1): x.append(torch.multinomial(T_[x[-1]], 1)[:, 0])
    return torch.stack(x, 1)
def train(task, teacher=None, prior=None, lr=2e-3):
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=ST, pct_start=0.1)
    for it in range(ST):
        m.train(); X, Mk = batch(task, 32); lg = fwd(m, X[:, :-1]); mk = Mk[:, 1:].float()
        loss = (F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1), reduction='none') * mk.reshape(-1)).sum() / mk.sum().clamp(min=1)
        if teacher is not None:
            P = sample_dream(prior)
            with torch.no_grad(): tl = fwd(teacher, P[:, :-1])
            loss = loss + LAM * F.kl_div(fwd(m, P[:, :-1]).log_softmax(-1).reshape(-1, 256), tl.softmax(-1).reshape(-1, 256), reduction='batchmean')
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if (it + 1) % (ST // 4) == 0: print(task, it + 1, round(loss.item(), 4), flush=True)
R = {'model': MODEL, 'seed': SEED, 'cond': COND, 'dream': os.environ.get('DREAM', 'unigram'), 'lam': LAM, 'steps_per_task': ST}; t0 = time.time(); hists = []
CKP = f"/home/claude/cont3_ck/{MODEL}_s{SEED}_{COND}_{os.environ.get('DREAM','unigram')}{os.environ.get('FTAG','')}.pt"; os.makedirs('/home/claude/cont3_ck', exist_ok=True); k0 = 0
if os.path.exists(CKP): c_ = torch.load(CKP); m.load_state_dict(c_['m']); hists = c_['hists']; R = c_['R']; k0 = c_['k']; torch.manual_seed(SEED * 7 + k0); print('RESUMED after task', k0, flush=True)   # phase checkpoint (cloud container resets)
for k, task in enumerate('ABC'):
    if k < k0: continue
    if COND == 'dream' and k > 0:
        teacher = copy.deepcopy(m).eval(); [p.requires_grad_(False) for p in teacher.parameters()]; wts = torch.tensor([float(os.environ.get('RECENT', 1.0)) ** i for i in range(len(hists))]); wts = wts / wts.sum(); prior = (torch.stack(hists) * wts.view(-1, *([1] * hists[0].dim()))).sum(0) + 1e-6   # RECENT>1 weights recent tasks more in dreams; prior = prior / prior.sum()
        train(task, teacher, prior)
    else: train(task)
    hists.append(hist(task)); R[f'after_{task}'] = {t: acc(m, t) for t in 'ABC'[:k + 1]}; print(R[f'after_{task}'], f'{time.time()-t0:.0f}s', flush=True); torch.save({'m': m.state_dict(), 'hists': hists, 'R': R, 'k': k + 1}, CKP)
R['secs'] = round(time.time() - t0); print(json.dumps(R), flush=True)
OUT = os.environ.get('LOUT', '/home/claude/cont3'); os.makedirs(OUT, exist_ok=True); json.dump(R, open(os.path.join(OUT, f'cont3_{MODEL}_s{SEED}_{COND}{"_bigram" if BIG else ""}{os.environ.get("FTAG", "")}.json'), 'w'))
