"""L-1 vs transformer on the larger, file-disjoint CPython corpus (38 MB train / 2.1 MB val).
Usage: python train_L1.py NAME MODEL SEED STEPS [lam] [fuzzy] [block]
  MODEL in {L1, T}; lam = state spike energy budget weight; fuzzy 0/1; block = decoupled block size (0 = end-to-end)"""
import sys, os, json, time, math, random, re, torch, torch.nn as nn, torch.nn.functional as F
_wf = os.environ.get('WAITFOR', '')   # run one GPU job at a time: wait until these result files exist
while _wf and not all(os.path.exists(os.path.join(os.environ.get('LOUT', '.'), w)) for w in _wf.split(',')): time.sleep(30)
import l_model as M0, l1_model as M1
import os
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.set_num_threads(int(os.environ.get('THREADS', 1)))
DATA = os.environ.get('LDATA', '/home/claude/data')
name, kind, seed, STEPS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lam = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0; fuzzy = int(sys.argv[6]) if len(sys.argv) > 6 else 1; block = int(sys.argv[7]) if len(sys.argv) > 7 else 0
torch.manual_seed(seed); random.seed(seed); T, BS = int(os.environ.get('SEQ', 256)), int(os.environ.get('BS', 16))
DM, NL = int(os.environ.get('DM', 160)), int(os.environ.get('NL', 4))
ld = lambda p: torch.tensor(list(open(p, 'rb').read()), dtype=torch.long)
tr, va = ld(os.path.join(DATA, 'py_big_train.txt')), ld(os.path.join(DATA, 'py_big_val.txt'))
def batch(d, g=None):
    ix = torch.randint(0, len(d) - T - 1, (BS,), generator=g); return torch.stack([d[i:i + T + 1] for i in ix]).to(DEV)
if kind == 'T': m = M0.TModel(d=int(os.environ.get('TDM', 152)), n_layers=int(os.environ.get('TNL', 4)), T=T)
elif kind == 'L4':
    import l4_model as M4; m = M4.L4Model(d=int(os.environ.get('DM', 144)), n_layers=NL)
elif kind == 'L5':
    import l5_model as M5; m = M5.L5Model(d=int(os.environ.get('DM', 144)), n_layers=NL, conv=int(os.environ.get('CONV', 4)), heads=int(os.environ.get('HEADS', 4)), rhit=bool(int(os.environ.get('RHIT', 0))), tsign=bool(int(os.environ.get('TSIGN', 0))), colpos=bool(int(os.environ.get('COLPOS', 0))), mixcopy=bool(int(os.environ.get('MIXCOPY', 0))), bracket=bool(int(os.environ.get('BRACKET', 0))), copyfeat=bool(int(os.environ.get('COPYFEAT', 0))))
elif kind == 'L6':   # L-6 = L-5 + all fixes: 8 memory blocks, finer spikes (-8..8), layout place cells, calibrated copy
    os.environ.setdefault('SPIKEK', '8'); import l5_model as M5
    m = M5.L5Model(d=int(os.environ.get('DM', 144)), n_layers=NL, conv=4, heads=int(os.environ.get('HEADS', 8)), colpos=True, mixcopy=True)
elif kind == 'L1T': m = M1.L1TModel(d=DM, n_layers=NL, homeo=bool(int(os.environ.get('HOMEO', 0))))
elif kind == 'L1TG': m = M1.L1TModel(d=DM, n_layers=NL, gated=True, homeo=bool(int(os.environ.get('HOMEO', 0))))
else: m = M1.L1Model(d=DM, n_layers=NL, fuzzy=bool(fuzzy), homeo=bool(int(os.environ.get('HOMEO', 0))))
if int(os.environ.get('SPIKE_GROUPS', 1)) > 1:             # flag (H4)
    for _l in m.layers: _l.spike_groups = int(os.environ['SPIKE_GROUPS'])
if float(os.environ.get('LOWRANK', 0)) > 0:
    import hwflags; _b, _a = hwflags.apply_lowrank(m, float(os.environ['LOWRANK']), hwflags.ALL_NAMES if int(os.environ.get('LOWRANK_ALL', 0)) else None); print('LOWRANK dense MACs/token', _b, '->', _a, flush=True)
if int(os.environ.get('SHARE_LAYERS', 0)):
    import hwflags; print('SHARE_LAYERS saved params', hwflags.apply_share_layers(m), flush=True)
if int(os.environ.get('TIE_HEAD', 0)):
    import hwflags; print('TIE_HEAD saved params', hwflags.apply_tie_head(m), flush=True)
if float(os.environ.get('DENSE_SPIKE', 0)) > 0:     # make dense projections event-driven (spike-coded inputs)
    import hwflags; _w = hwflags.apply_dense_spikes(m, float(os.environ['DENSE_SPIKE'])); print('DENSE_SPIKE wrapped', len(_w), 'matrices', flush=True)
if os.environ.get('QAT', 'none') != 'none':                 # flag (H6)
    import hwflags; hwflags.apply_qat(m, os.environ['QAT'], int(os.environ.get('QAT_B', 32)))
m = m.to(DEV)
heads = nn.ModuleList([nn.Sequential(M0.RMSNorm(DM), nn.Linear(DM, 256)) for _ in range(NL - 1)]).to(DEV) if block else None
params = list(m.parameters()) + (list(heads.parameters()) if heads else [])
def run(x, train=False):
    inp = x[:, :-1]
    if kind == 'T': return m(inp)[0], []
    if kind in ('L4', 'L5', 'L6') and int(os.environ.get('FASTMEM', 1)):
        import fastmem; rec = fastmem.hippo_batch(inp)
    else: rec = M0.Hippocampus.batch(inp.cpu()).to(DEV)
    return m(inp, rec, block=block or None)
LR = float(os.environ.get('LR', 2e-3)); opt = torch.optim.AdamW(params, LR, weight_decay=0.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
R = {'device': DEV, 'homeo': int(os.environ.get('HOMEO', 0)), 'name': name, 'model': kind, 'seed': seed, 'steps': STEPS, 'lam': lam, 'fuzzy': fuzzy, 'block': block, 'params': M0.nparams(m)}
t0 = time.time(); start = 0
RES = os.path.join(os.environ.get('LOUT', '.'), f'{name}.resume.pt')
if os.environ.get('RESUME') and os.path.exists(RES):
    ck = torch.load(RES, map_location=DEV); m.load_state_dict(ck['m']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sch']); start = ck['it'] + 1
    torch.set_rng_state(ck['rng'].cpu()); print('resumed at step', start, flush=True)
for it in range(start, STEPS):
    if os.environ.get('RESUME') and it % 100 == 0 and it > start: torch.save({'m': m.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'it': it - 1, 'rng': torch.get_rng_state()}, RES)
    m.train(); xb = batch(tr); ACC = int(os.environ.get('ACC', 1)); opt.zero_grad()
    for x in xb.chunk(ACC):            # gradient accumulation: same effective batch, less memory
        y = x[:, 1:].reshape(-1); lg, states = run(x, True); loss = F.cross_entropy(lg.reshape(-1, 256), y)
        if block:
            for i in range(len(states) - 1):
                if (i + 1) % block == 0: loss = loss + F.cross_entropy(heads[i](states[i]).reshape(-1, 256), y)   # local goal at each block's top
        if kind != 'T' and lam > 0: loss = loss + lam * m.rate_cost()
        if hasattr(m, '_dense_spk'): loss = loss + float(os.environ.get('DENSE_LAM', 0.01)) * sum(w.rate for w in m._dense_spk) / len(m._dense_spk)
        (loss / ACC).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
    if it % 250 == 0 and it > 0 and os.environ.get('CKPT'): torch.save(m.state_dict(), os.path.join(os.environ.get('LOUT', '.'), f'{name}.ckpt.pt'))
    if os.path.exists(os.path.join(os.environ.get('LOUT', '.'), f'STOP_{name}')): print('stopped by request', flush=True); sys.exit(3)
    if it % 50 == 0: print(name, it, round(loss.item(), 3), f'{time.time() - t0:.0f}s', (f'gpu_mem_GB {torch.cuda.max_memory_allocated() / 1e9:.2f}' if DEV == 'cuda' else ''), flush=True)
R['train_minutes'] = round((time.time() - t0) / 60, 1); R['sec_per_step'] = round((time.time() - t0) / STEPS, 3); m.eval()
torch.save(m.state_dict(), os.path.join(os.environ.get('LOUT', '.'), f'{name}.pt'))
with torch.no_grad():
    g = torch.Generator().manual_seed(123); tot = 0
    for _ in range(40): x = batch(va, g); tot += F.cross_entropy(run(x)[0].reshape(-1, 256), x[:, 1:].reshape(-1)).item()
    R['val_bits_per_byte'] = round(tot / 40 / math.log(2), 4)
    if kind != 'T': R['active_neurons_per_layer [state, ffn]'] = [[round(v, 3) for v in l.last_rate] for l in m.layers]
    rng = random.Random(7); vb = va.tolist()
    def perturb(s, how):
        if how == 'exact': return list(s), [True] * len(s)
        if how == 'typos5%':
            s2 = list(s); ch = [False] * len(s)
            for i in range(len(s)):
                if rng.random() < 0.05 and 97 <= s[i] <= 122: s2[i] = rng.randrange(97, 123); ch[i] = s2[i] != s[i]
            return s2, [not c for c in ch]
        if how == 'renamed':   # consistent renaming of identifiers (like refactoring a variable name)
            txt = bytes(s).decode('latin-1'); names = {}
            def sub(mo):
                w = mo.group(0)
                if w in ('def', 'self', 'return', 'import', 'from', 'class', 'for', 'in', 'if', 'else', 'not', 'and', 'None', 'True', 'False'): return w
                if w not in names: names[w] = ''.join(chr(rng.randrange(97, 123)) for _ in range(len(w)))
                return names[w]
            t2 = re.sub(r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b', sub, txt); b2 = list(t2.encode('latin-1'))
            return b2, [x == y for x, y in zip(b2, s)]
    def score(n, how, reps=4):
        accs = []
        for _ in range(reps):
            st = rng.randrange(0, len(vb) - n); s = vb[st:st + n]; s2, keep = perturb(s, how)
            x = torch.tensor(s + s2)[None].to(DEV)
            if kind == 'T' and x.size(1) > T: return 'cannot (context limit 256)'
            pred = run(torch.cat([x, x[:, -1:]], 1))[0].argmax(-1)[0, n - 1:n - 1 + len(s2)]
            ok = [(int(p) == b) for p, b, k in zip(pred.tolist(), s2, keep) if k]
            accs.append(sum(ok) / max(len(ok), 1))
        return round(sum(accs) / len(accs), 4)
    R['copy'] = {}
    for how in ['exact', 'typos5%', 'renamed']:
        for n in ([120] if kind == 'T' else [120, 512, 2048]):
            R['copy'][f'{how} n={n}'] = score(n, how); print(name, how, n, R['copy'][f'{how} n={n}'], flush=True)
    if kind in ('L1T', 'L1TG'):
        m.use_track = False
        for how in ['exact', 'typos5%', 'renamed']: R['copy'][f'{how} n=512 (trajectory OFF)'] = score(512, how)
        m.use_track = True
    if kind == 'L1' and fuzzy:      # ablation at test time: same trained model, fuzzy memory switched off
        m.use_fuzzy = False
        for how in ['typos5%', 'renamed']: R['copy'][f'{how} n=512 (fuzzy OFF)'] = score(512, how)
        m.use_fuzzy = True
print(json.dumps(R, indent=1)); json.dump(R, open(os.path.join(os.environ.get('LOUT', '.'), f'results_L1_{name}.json'), 'w'), indent=1)
