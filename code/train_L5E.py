"""Effort gate FIX ('post-hoc thalamus'): the old gate was trained together with the model and its deep supervision
damaged full-depth quality (1.347 -> 1.385). Fix: train the model normally FIRST, then
  stage A: FREEZE the trained model and train only small exit read-outs + halt heads (full-depth path untouched by construction)
  stage B: short, low-LR adaptation with practice exits + DISTILLATION to the frozen original (teaches deeper layers to live
           with skimmed tokens while anchoring the full-depth output to what it was).
Usage: python train_L5E.py NAME SEED STEPS_A STEPS_B  (base: L5_1M_s0.pt)"""
import sys, os, json, time, math, copy, torch, torch.nn as nn, torch.nn.functional as F
_O = os.environ.get('LOUT', '.'); _wf = os.environ.get('WAITFOR', '')
while _wf and not all(os.path.exists(os.path.join(_O, w)) for w in _wf.split(',')): time.sleep(30)
import l_model as M0, l4_model as M4, l5_model as M5, fastmem
torch.set_num_threads(int(os.environ.get('THREADS', 1))); DEV = 'cuda' if torch.cuda.is_available() and os.environ.get('L5E_GPU', '0') == '1' else 'cpu'; name, seed, SA, SB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); torch.manual_seed(seed)
T, BS = 256, 16
ld = lambda p: torch.tensor(list(open(p, 'rb').read()), dtype=torch.long)
DATA = os.environ.get('LDATA', '/home/claude/data'); OUTD = os.environ.get('LOUT', '.')
tr, va = ld(os.path.join(DATA, 'py_big_train.txt')), ld(os.path.join(DATA, 'py_big_val.txt'))
def batch(d, g=None): ix = torch.randint(0, len(d) - T - 1, (BS,), generator=g); return torch.stack([d[i:i + T + 1] for i in ix]).to(DEV)
class L5E2(M4.L4E2Model):
    def __init__(s, d=144, n_layers=4, heads=4):
        super().__init__(d, n_layers, heads); s.fw = nn.ModuleList([M5.DeltaMemory(d, heads, conv=4) for _ in range(n_layers)])
m = L5E2(); base_sd = torch.load(os.path.join(OUTD, os.environ['BASE']) if 'BASE' in os.environ and not os.path.exists(os.environ['BASE']) else os.environ.get('BASE', 'L5_1M_s0.pt'), map_location='cpu')
missing = m.load_state_dict(base_sd, strict=False); assert not missing.unexpected_keys, missing.unexpected_keys
with torch.no_grad():
    for i in range(len(m.enorm)): m.enorm[i].load_state_dict(m.norm.state_dict()); m.etrust[i].load_state_dict(m.trust.state_dict())
m = m.to(DEV); teacher = copy.deepcopy(m).eval()
for p in teacher.parameters(): p.requires_grad_(False)
new_params = [p for n, p in m.named_parameters() if n.startswith(('enorm', 'halt', 'etrust'))]
def evaluate(R, tag):
    m.eval()
    with torch.no_grad():
        for thr in [None, 0.99, 0.95, 0.9, 0.8]:
            g = torch.Generator().manual_seed(123); tot, used = 0, 0
            for _ in range(30):
                x = batch(va, g); inp = x[:, :-1]; rec = fastmem.hippo_batch(inp)
                if thr is None: lg = m(inp, rec)[0][-1]; u = 1.0
                else: lg, u = m(inp, rec, thr=thr)
                tot += F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1)).item(); used += u
            R[f'{tag} | ' + ('full depth' if thr is None else f'gate {thr}')] = {'bits_per_byte': round(tot / 30 / math.log(2), 4), 'compute_used': round(used / 30, 3)}
    print(json.dumps(R, indent=1), flush=True)
def train(steps, params, lr, stage):
    opt = torch.optim.AdamW(params, lr, weight_decay=0.0); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1); t0 = time.time()
    for it in range(steps):
        m.train(); x = batch(tr); inp, y = x[:, :-1], x[:, 1:]; rec = fastmem.hippo_batch(inp)
        outs, halts = m(inp, rec, practice_p=0.5 if stage == 'B' else 0.0); K = len(outs)
        fa = outs[-1].argmax(-1).detach()
        exit_loss = sum(F.cross_entropy(o.reshape(-1, 256), y.reshape(-1)) for o in outs[:-1]) / (K - 1)
        halt_loss = sum(F.binary_cross_entropy_with_logits(h, (o.argmax(-1) == fa).float()) for o, h in zip(outs[:-1], halts)) / (K - 1)
        loss = exit_loss + 0.3 * halt_loss
        if stage == 'B':
            with torch.no_grad(): tl = teacher(inp, rec)[0][-1]
            final = F.cross_entropy(outs[-1].reshape(-1, 256), y.reshape(-1))
            distill = F.kl_div(outs[-1].log_softmax(-1), tl.log_softmax(-1), log_target=True, reduction='batchmean') / T
            loss = 2.0 * final + 2.0 * distill + 0.5 * exit_loss + 0.3 * halt_loss + 0.03 * m.rate_cost()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        if it % 250 == 0: print(name, stage, it, round(loss.item(), 3), f'{time.time() - t0:.0f}s', flush=True)
R = {'name': name, 'base': os.environ.get('BASE', 'L5_1M_s0.pt')}
evaluate(R, '0 before')                                     # exits untrained: gate rows meaningless, full depth = base model
for p in m.parameters(): p.requires_grad_(False)
for p in new_params: p.requires_grad_(True)
train(SA, new_params, 3e-3, 'A'); evaluate(R, 'A frozen model, heads trained')
if SB > 0:
    for p in m.parameters(): p.requires_grad_(True)
    train(SB, list(m.parameters()), 3e-4, 'B'); evaluate(R, 'B short adaptation + distillation')
json.dump(R, open(os.path.join(OUTD, f'results_L5E_{name}.json'), 'w'), indent=1); torch.save({k: v.cpu() for k, v in m.state_dict().items()}, os.path.join(OUTD, f'{name}.pt'))
