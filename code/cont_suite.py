"""Continual-learning suite (vault note 130, experiment B1): does a model forget task A while learning task B, and does hippocampus-style replay help?
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
t0 = time.time(); m = new_model(); R = {'model': MODEL, 'seed': SEED, 'stepsA': SA, 'stepsB': SB, 'params': M0.nparams(m)}
train(m, SA, 'A'); R['A_after_A'] = round(sum(acc(m, 'A') for _ in range(4)) / 4, 4)
# fixed-length replay buffer: 64 episodes of A, all with 4 pairs so tensors stack
g = torch.Generator().manual_seed(SEED); BX, BM = batch('A', 64, 4, 4); buf = (BX, BM)
for cond, rf in (('none', 0.0), ('replay', 0.25)):
    mm = copy.deepcopy(m); torch.manual_seed(SEED + 7); train(mm, SB, 'B', buf if rf else None, rf)
    R[cond] = {'A_after_B': round(sum(acc(mm, 'A') for _ in range(4)) / 4, 4), 'B_after_B': round(sum(acc(mm, 'B') for _ in range(4)) / 4, 4)}
R['secs'] = round(time.time() - t0); print(json.dumps(R), flush=True)
OUT = os.environ.get('LOUT', 'results'); os.makedirs(OUT, exist_ok=True); json.dump(R, open(os.path.join(OUT, f'cont_{MODEL}_s{SEED}.json'), 'w'))
