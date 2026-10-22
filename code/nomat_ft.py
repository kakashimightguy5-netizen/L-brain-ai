"""(nomat_ft: adds SPARSE_K env = sparse ternary wiring student) Quality-preserving compression: SVD-init a low-rank student from the TRAINED baseline, then recover quality by fine-tuning with
knowledge distillation (teacher = the baseline itself). Usage: python compress_ft.py FRAC STEPS [all6|dense4] [TAG]
Trains only on py_big_train; evaluates on the fixed 6x1024 eval windows (same as fix_eval). MEASURED bits/byte."""
import os, sys, math, json, time, copy, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem, hwflags
torch.manual_seed(int(os.environ.get('SEED', 0))); torch.set_num_threads(int(os.environ.get('THREADS', 4))); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
frac, STEPS = float(sys.argv[1]), int(sys.argv[2]); which = sys.argv[3] if len(sys.argv) > 3 else 'all6'; tag = sys.argv[4] if len(sys.argv) > 4 else f'cft_{which}_{frac}'
DATA = os.environ.get('LDATA', '/home/claude/data'); ld = lambda p: torch.tensor(list(open(p, 'rb').read()), dtype=torch.long)
tr, va = ld(os.path.join(DATA, 'py_big_train.txt')), ld(os.path.join(DATA, 'py_big_val.txt')); half = len(va) // 2
X = torch.stack([va[half + i * 1024: half + i * 1024 + 1025] for i in range(6)]).to(DEV)
def new(): return M5.L5Model(d=128, n_layers=4, heads=4)
teacher = new(); teacher.load_state_dict(torch.load(next(p for p in (os.path.join(os.environ.get('LOUT', '.'), 'LEdge_s0.pt'), os.path.join(HERE, 'LEdge_s0.pt')) if os.path.exists(p)), map_location='cpu')); teacher.to(DEV).eval()
for p in teacher.parameters(): p.requires_grad_(False)
st = copy.deepcopy(teacher).cpu(); names = () if frac >= 1 else (hwflags.ALL_NAMES if which == 'all6' else hwflags.DENSE_NAMES)
for parent in list(st.modules()):
    for nm in names:
        lin = getattr(parent, nm, None)
        if isinstance(lin, nn.Linear):
            o, i = lin.weight.shape; r = max(4, round(frac * min(o, i))); U, S, Vh = torch.linalg.svd(lin.weight.data, full_matrices=False)
            A = nn.Linear(i, r, bias=False); B = nn.Linear(r, o, bias=lin.bias is not None)
            A.weight.data = Vh[:r] * S[:r, None].sqrt(); B.weight.data = U[:, :r] * S[:r].sqrt()
            if lin.bias is not None: B.bias.data = lin.bias.data.clone()
            setattr(parent, nm, nn.Sequential(A, B))
import nomat
SK = int(os.environ.get('SPARSE_K', 0))
if SK:
    _b, _a = nomat.apply_sparse_tern(st, SK); print('NOMAT K', SK, 'dense MACs/token', _b, '-> adds/token', _a, f'({_a/_b:.3f}x)', flush=True)
QAT = os.environ.get('QAT', 'none')
QMAP = os.environ.get('QAT_MAP', '')
if QMAP:
    print('QAT_MAP', hwflags.apply_qat_map(st, dict(kv.split(':') for kv in QMAP.split(',')), 32, tuple(x for x in os.environ.get('QAT_SKIP', '').split(',') if x)), flush=True)
elif QAT != 'none': print('QAT', QAT, 'on', hwflags.apply_qat(st, QAT, 32, tuple(x for x in os.environ.get('QAT_SKIP', '').split(',') if x)), 'matrices', flush=True)
st.to(DEV)
if int(os.environ.get('UNFREEZE', 0)):
    for p_ in st.parameters(): p_.requires_grad_(True)   # deepcopy of the frozen teacher leaves un-factorised params frozen (bug in runs before UNFREEZE existed)
HID = float(os.environ.get('HID', 0)); RAMP = float(os.environ.get('ALPHA_RAMP', 0)); tf, sf = [], []
if HID > 0:
    for l in teacher.layers: l.register_forward_hook(lambda m_, i_, o_: tf.append(o_[0]))
    for l in st.layers: l.register_forward_hook(lambda m_, i_, o_: sf.append(o_[0]))
def bpb(m):
    m.eval()
    with torch.no_grad():
        inp, y = X[:, :-1], X[:, 1:]; lp = m(inp, fastmem.hippo_batch(inp))[0].log_softmax(-1); return -lp.gather(-1, y[..., None]).mean().item() / math.log(2)
base = bpb(teacher); b0 = bpb(st); print(tag, 'params', M0.nparams(st), 'teacher', round(base, 4), 'svd-init', round(b0, 4), flush=True)
LRN = float(os.environ.get('LRN', 5e-4)); opt = torch.optim.AdamW(st.parameters(), LRN, weight_decay=0.0); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LRN, total_steps=STEPS, pct_start=0.1)
BS, T = 16, 256; t0 = time.time(); log = []
def batch():
    ix = torch.randint(0, len(tr) - T - 1, (BS,)); return torch.stack([tr[i:i + T + 1] for i in ix]).to(DEV)
for it in range(STEPS):
    st.train(); x = batch(); inp, y = x[:, :-1], x[:, 1:]; rec = fastmem.hippo_batch(inp)
    tf.clear(); sf.clear()
    if RAMP > 0:
        for m_ in st.modules():
            if isinstance(m_, hwflags.FakeQuant): m_.alpha = min(1.0, it / max(1, RAMP * STEPS))
    with torch.no_grad(): tl = teacher(inp, rec)[0]
    sl = st(inp, rec)[0]; KT = float(os.environ.get('KDT', 1)); kd = F.kl_div((sl / KT).log_softmax(-1).reshape(-1, 256), (tl / KT).softmax(-1).reshape(-1, 256), reduction='batchmean') * KT * KT
    loss = 0.5 * kd + 0.5 * F.cross_entropy(sl.reshape(-1, 256), y.reshape(-1))
    if HID > 0: loss = loss + HID * sum(F.mse_loss(a, b.detach()) / (b.detach().pow(2).mean() + 1e-6) for a, b in zip(sf, tf)) / len(tf)
    opt.zero_grad()
    if not loss.requires_grad:
        print('DEBUG no grad: sl', sl.requires_grad, 'trainable params', sum(p.numel() for p in st.parameters() if p.requires_grad), 'of', sum(p.numel() for p in st.parameters()), 'training', st.training, 'grad_enabled', torch.is_grad_enabled(), flush=True); sys.exit(3)
    loss.backward(); nn.utils.clip_grad_norm_(st.parameters(), 1.0); opt.step(); sch.step()
    if (it + 1) % max(1, STEPS // 6) == 0 or it == STEPS - 1:
        b = bpb(st); log.append((it + 1, b)); print(tag, 'step', it + 1, 'bpb', round(b, 4), f'{(b/base-1)*100:+.2f}%', f'{time.time()-t0:.0f}s', flush=True)
OUT = os.environ.get('LOUT', 'results'); os.makedirs(OUT, exist_ok=True); torch.save(st.state_dict(), os.path.join(OUT, tag + '.pt')); json.dump({'tag': tag, 'frac': frac, 'which': which, 'steps': STEPS, 'params': M0.nparams(st), 'teacher': base, 'svd_init': b0, 'log': log, 'final': log[-1][1], 'device': DEV, 'qat': QAT}, open(os.path.join(OUT, f'{tag}.json'), 'w'))
