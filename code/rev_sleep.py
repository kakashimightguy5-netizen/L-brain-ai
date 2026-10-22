"""Brain principle test (vault note 134): SLEEP REVERSE REPLAY against the reversal curse.
Biology: during sleep/rest the hippocampus replays recent episodes, often in REVERSE order (Foster & Wilson 2006, Diba & Buzsaki 2007),
and cortex consolidates from that replay. Here: a model that memorised facts 'AB>CD;' forward-only (lim_suite reverse task, 8000 steps,
checkpoint from LCK dir) enters a sleep phase. It GENERATES its own memories (free sampling from a random first letter; no access to the
fact table), parses 'xy>zw;' episodes, and trains on them replayed forward AND reversed ('zw>xy;'). Zero external data during sleep.
Usage: python rev_sleep.py MODEL SEED [SLEEP_STEPS]   (needs LCK/ck_MODEL_reverse_sSEED.pt). MEASURED forward/reverse accuracy on all 200 facts.
Flags: REVONLY=1 trains only the reversed replay (ablation); POOL refresh every 200 steps."""
import os, sys, re, json, time, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]; os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem
MODEL, SEED = sys.argv[1], int(sys.argv[2]); SS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
torch.manual_seed(SEED); torch.set_num_threads(int(os.environ.get('THREADS', 1)))
L = lambda s: [ord(c) for c in s]; LET = L('abcdefghijklmnopqrstuvwxyz')
def key(n):
    idx = torch.randperm(676)[:n]; return [(LET[i // 26], LET[i % 26]) for i in idx.tolist()]
NF = 200; pk = key(2 * NF); FACT = [(pk[2 * i], pk[2 * i + 1]) for i in range(NF)]   # identical construction/RNG order to lim_suite.py
TRUE = {bytes(a): bytes(b) for a, b in FACT}
m = M5.L5Model(d=128, n_layers=4, heads=4) if MODEL == 'L5' else M0.TModel(d=128, n_layers=4, h=4, T=1024)
CK = os.path.join(os.environ.get('LCK', '/home/claude/lim8k'), f'ck_{MODEL}_reverse_s{SEED}.pt'); m.load_state_dict(torch.load(CK, map_location='cpu')['m'])
def fwd(x): return m(x, fastmem.hippo_batch(x))[0] if MODEL == 'L5' else m(x)[0]
@torch.no_grad()
def evaluate():
    m.eval(); f, r = [], []
    for (a, b), (c, d) in FACT: f.append([a, b, ord('>'), c, d]); r.append([c, d, ord('>'), a, b])
    out = {}
    for name, X in (('forward', torch.tensor(f)), ('reverse', torch.tensor(r))):
        pred = fwd(X[:, :-1]).argmax(-1); out[name] = round(((pred[:, 2:] == X[:, 3:]).all(1)).float().mean().item(), 4)
    return out
@torch.no_grad()
def dream(thr=float(os.environ.get('FAMTHR', 0.5))):
    """cued spontaneous reactivation: every 2-letter cue 'xy>' is replayed; the model completes it greedily, and only completions it is
    FAMILIAR with (joint prob > thr: a familiarity signal, like perirhinal 'I know this') are kept as memories. No fact table used."""
    m.eval(); cues = torch.tensor([[a, b, ord('>')] for a in LET for b in LET]); x = cues; conf = torch.ones(len(cues))
    for _ in range(2):
        p = fwd(x)[:, -1].softmax(-1); c = p.argmax(-1); conf = conf * p.gather(1, c[:, None])[:, 0]; x = torch.cat([x, c[:, None]], 1)
    eps = []
    for row, cf in zip(x.tolist(), conf.tolist()):
        if cf > thr and all(97 <= t <= 122 for t in row[3:5]): eps.append((bytes(row[:2]).decode(), bytes(row[3:5]).decode()))
    return eps
def replay_batch(eps, B=32):
    X, M = [], []
    for _ in range(B):
        s, mk = [], []
        for j in torch.randint(0, len(eps), (6,)).tolist():
            a, c = eps[j]; rev = os.environ.get('REVONLY') == '1' or torch.rand(1).item() < 0.5
            subj, obj = (c, a) if rev else (a, c); s += L(subj) + [ord('>')] + L(obj) + [ord(';')]; mk += [0, 0, 0, 1, 1, 0]
        X.append(s); M.append(mk)
    return torch.tensor(X), torch.tensor(M)
import copy
teacher = copy.deepcopy(m).eval(); ANC = float(os.environ.get('ANCHOR', 0))   # ANCHOR flag (default off): KD to the pre-sleep self on random bytes (note 133) to stop memory drift
for p_ in teacher.parameters(): p_.requires_grad_(False)
R = {'model': MODEL, 'anchor': ANC, 'seed': SEED, 'sleep_steps': SS, 'before': evaluate(), 'log': []}; print(R['before'], flush=True)
opt = torch.optim.AdamW(m.parameters(), float(os.environ.get('LR', 1e-3)), weight_decay=0.01); t0 = time.time(); eps = []
for it in range(SS):
    if it % 200 == 0:
        new_eps = dream()
        if os.environ.get('CONSIST') == '1':   # CONSIST flag: keep only memories that also came back in the previous dream (repeated reactivation -> consolidation)
            prev = set(globals().get('_prev_eps', [])); globals()['_prev_eps'] = new_eps; eps = [e for e in new_eps if e in prev] if prev else new_eps
        else: eps = new_eps
        ok = sum(TRUE.get(a.encode('latin1')) == c.encode('latin1') for a, c in eps)   # diagnostic only, never used for training
        cov = len({a for a, c in eps if TRUE.get(a.encode('latin1')) == c.encode('latin1')})
        R['log'].append({'it': it, 'episodes': len(eps), 'correct': ok, 'facts_covered': cov, **evaluate()}); print(R['log'][-1], f'{time.time()-t0:.0f}s', flush=True)
        if not eps: break
    m.train(); X, Mk = replay_batch(eps); lg = fwd(X[:, :-1]); mk = Mk[:, 1:].float()
    loss = (F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1), reduction='none') * mk.reshape(-1)).sum() / mk.sum().clamp(min=1)
    if ANC > 0:
        P = torch.randint(0, 256, (32, 24)); sl = (m(P, fastmem.hippo_batch(P))[0] if MODEL == 'L5' else m(P)[0])
        with torch.no_grad(): tl = teacher(P, fastmem.hippo_batch(P))[0] if MODEL == 'L5' else teacher(P)[0]
        loss = loss + ANC * F.kl_div(sl.log_softmax(-1).reshape(-1, 256), tl.softmax(-1).reshape(-1, 256), reduction='batchmean')
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
R['after'] = evaluate(); R['secs'] = round(time.time() - t0); print(json.dumps(R), flush=True)
OUT = os.environ.get('LOUT', '/home/claude/revsleep'); os.makedirs(OUT, exist_ok=True); json.dump(R, open(os.path.join(OUT, f'revsleep_{MODEL}_s{SEED}{os.environ.get("FTAG", "")}.json'), 'w'))
