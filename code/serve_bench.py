"""Part 3 - fit the effort policy on CALIBRATION, evaluate on EVAL, measure GPU energy per effort level.
Model: L-5 1M + post-hoc effort gate (train_L5E.py checkpoint, L4E2 skim model). Usage: python serve_bench.py CKPT TAG
  CAL  = first half of py_big_val (threshold grid search; effort levels chosen here, never on EVAL)
  EVAL = second half (quality, easy/hard bytes, work, flags) - B parallel streams (decisions are per stream)
  energy: batch-1 decode over EVAL bytes, 5 repeats per level (MEASURED, GPU board only, NVML)
  SPARK energy: MODELLED MACs x ESTIMATED 45 nm per-op energy (Horowitz ISSCC 2014; configs/sustainability.yaml)
Effort levels: low / balanced / high = cheapest CAL setting whose bits/byte is within +5 % / +2 % / +0.5 % of full depth.
Writes LOUT/serve_<TAG>.json"""
import os, sys, time, json, math, torch, numpy as np
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import torch.nn as nn, l4_model as M4, l5_model as M5, runtime
ck, tag = sys.argv[1], sys.argv[2]; dev = 'cuda' if torch.cuda.is_available() else 'cpu'
QUICK = os.environ.get('QUICK', '0') == '1'; REP = 2 if QUICK else 5
class L5E2(M4.L4E2Model):
    def __init__(s, d=144, n_layers=4, heads=4):
        super().__init__(d, n_layers, heads); s.fw = nn.ModuleList([M5.DeltaMemory(d, heads, conv=4) for _ in range(n_layers)])
m = L5E2().to(dev).eval(); m.load_state_dict(torch.load(ck if os.path.exists(ck) else os.path.join(OUT, ck), map_location=dev))
data = list(open(os.path.join(os.environ.get('LDATA', '/home/claude/data'), 'py_big_val.txt'), 'rb').read()); half = len(data) // 2
B, W = (4, 256) if QUICK else (8, 1024)
E_OP = 0.23e-12 + 1.25e-12            # ESTIMATED J per MAC incl. one 8-bit weight read (45 nm, Horowitz 2014) - see configs/sustainability.yaml
def run(start, pol, cap=None, B=B, W=W):
    st = runtime.EffortStepper(m, B, 'x', {'x': pol}, cap); seqs = [data[start + b * W: start + b * W + W + 1] for b in range(B)]
    lp_true, top1 = np.zeros((W, B)), np.zeros((W, B), bool)
    for t in range(W):
        lp = st.step([s[t] for s in seqs]); y = torch.tensor([s[t + 1] for s in seqs], device=lp.device)
        lp_true[t] = lp.gather(-1, y[:, None])[:, 0].cpu().numpy(); top1[t] = (lp.argmax(-1) == y).cpu().numpy()
    lg = {k: np.stack([l[k].numpy() for l in st.log]) for k in st.log[0]}
    D = st.D; spark = (lg['layers_full'] * 8 * D * D + (st.nL - lg['layers_full']) * 6 * D * D + lg['fired'] * D + D * 256)
    return dict(bits=-lp_true / math.log(2), top1=top1, layers=lg['layers_full'], work=lg['work'] / st.full_work(), spark_macs=spark,
                instinct=lg['instinct'], flag=lg['flag'], fired=lg['fired'], full_work=st.full_work())
R = {'checkpoint': ck, 'device': torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', 'streams': B, 'bytes_per_stream': W}
FULL = dict(tau_instinct=2.0, tau_halt=2.0)
cal_full = run(0, FULL); bf = cal_full['bits'].mean(); grid = []
SMALL = os.environ.get('GRID', '') == 'small'
for th in ((0.95,) if SMALL else (0.995, 0.99, 0.98, 0.95, 0.9, 0.8)):
    for ti in ((2.0, 0.95) if SMALL else (2.0, 0.995, 0.98, 0.95, 0.9)):
        for rb in ((False,) if SMALL else (False, True)):
            p = dict(tau_instinct=ti, tau_halt=th, recall_bypass=rb); r = run(0, p)
            grid.append(dict(policy=p, bits=float(r['bits'].mean()), work=float(r['work'].mean()))); print('cal', grid[-1], flush=True)
R['cal_full_bits'] = float(bf); R['cal_grid'] = grid
levels = {'full': FULL}
for name, tol in (('high', 0.005), ('balanced', 0.02), ('low', 0.05)):
    ok = [g for g in grid if g['bits'] <= bf * (1 + tol)]; levels[name] = min(ok, key=lambda g: g['work'])['policy'] if ok else FULL
R['levels_fitted_on_CAL'] = levels
# ---------------- EVAL ----------------
ev_full = run(half, FULL); hard_cut = np.quantile(ev_full['bits'], 0.8); easy_cut = np.quantile(ev_full['bits'], 0.5)
hard, easy = ev_full['bits'] >= hard_cut, ev_full['bits'] <= easy_cut
res = {}
cfgs = [(k, v, None) for k, v in levels.items()] + [('balanced + energy cap 0.85 of full work', levels['balanced'], 0.85), ('full + energy cap 0.75 of full work', FULL, 0.75)]
for name, pol, cap in cfgs:
    r = run(half, pol, cap)
    res[name] = {'policy': pol, 'cap_layers': cap, 'bits_per_byte': round(float(r['bits'].mean()), 4), 'top1_acc': round(float(r['top1'].mean()), 4),
                 'bits_easy_bytes': round(float(r['bits'][easy].mean()), 4), 'bits_hard_bytes': round(float(r['bits'][hard].mean()), 4),
                 'work_vs_full_GPU_MACs (MODELLED)': round(float(r['work'].mean()), 4),
                 'work_easy': round(float(r['work'][easy].mean()), 4), 'work_hard': round(float(r['work'][hard].mean()), 4),
                 'layers_full_mean': round(float(r['layers'].mean()), 3), 'instinct_share': round(float(r['instinct'].mean()), 4),
                 'flag_share': round(float(r['flag'].mean()), 4), 'flag_precision (flagged bytes that are wrong)': round(float((~r['top1'][r['flag']]).mean()), 4) if r['flag'].any() else None,
                 'SPARK_MACs_per_byte (MODELLED, event-driven)': round(float(r['spark_macs'].mean())),
                 'SPARK_nJ_per_byte (ESTIMATED)': round(float(r['spark_macs'].mean() * E_OP * 1e9), 2),
                 'SPARK_nJ_easy': round(float(r['spark_macs'][easy].mean() * E_OP * 1e9), 2), 'SPARK_nJ_hard': round(float(r['spark_macs'][hard].mean() * E_OP * 1e9), 2)}
    print('eval', name, res[name], flush=True)
R['eval'] = res; R['easy_def'] = 'bytes whose full-depth loss is in the lowest 50 % (EVAL)'; R['hard_def'] = 'highest 20 %'
json.dump(R, open(os.path.join(OUT, f'serve_{tag}.json'), 'w'), indent=1)
# ---------------- measured GPU energy, batch 1 ----------------
if dev == 'cuda':
    import energy as EN
    idle, _ = EN.idle_watts(); R['idle_W'] = idle; NB = 512 if QUICK else 2048
    for name, pol, cap in cfgs:
        st = [None]
        def f():
            st[0] = runtime.EffortStepper(m, 1, 'x', {'x': pol}, cap); seq = data[half: half + NB]
            for t in range(NB): st[0].step([seq[t]])
        f(); summ, _ = EN.repeat(f, REP, NB, idle, torch.cuda.synchronize); res[name]['GPU_measured_batch1'] = summ
        print('energy', name, summ, flush=True)
    R['eval'] = res
json.dump(R, open(os.path.join(OUT, f'serve_{tag}.json'), 'w'), indent=1); print('done')
