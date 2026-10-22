"""Energy fix A/B (GPU job): one L5 training step with the GPU scan kernel ON (cuscan, after the self-test fix) vs OFF (log-depth scan), plus a matched transformer step.
Interleaved A/B/A/B so GPU contention and clock drift hit both equally. Reports ms/step (median), CUDA kernel launches per step, NVML energy per token (indicative if other jobs share the GPU).
Writes ../results/scan_ab.json. No power/clock settings are changed. env: AB_REPS (default 5), AB_STEPS (default 10), AB_B/AB_T (default 16/256)."""
import os, sys, json, time, statistics as st, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]; os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; REPS = int(os.environ.get('AB_REPS', 5)); STEPS = int(os.environ.get('AB_STEPS', 10)); B = int(os.environ.get('AB_B', 16)); T = int(os.environ.get('AB_T', 256))
torch.manual_seed(0); x = torch.randint(0, 256, (B, T)); rec = fastmem.hippo_batch(x); x = x.to(DEV); rec = rec.to(DEV)
L = M5.L5Model(d=128, n_layers=4, heads=4).to(DEV); TT = M0.TModel(d=128, n_layers=4, h=4, T=max(T, 256)).to(DEV)
optL = torch.optim.AdamW(L.parameters(), 1e-3); optT = torch.optim.AdamW(TT.parameters(), 1e-3)
def stepL():
    lg = L(x[:, :-1], rec[:, :, :-1])[0]; loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1)); optL.zero_grad(); loss.backward(); optL.step()
def stepT():
    lg = TT(x[:, :-1])[0]; loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1)); optT.zero_grad(); loss.backward(); optT.step()
sync = torch.cuda.synchronize if DEV == 'cuda' else (lambda: None)
def timed(fn):
    for _ in range(2): fn()
    sync(); t = time.perf_counter()
    for _ in range(STEPS): fn()
    sync(); return (time.perf_counter() - t) / STEPS * 1000
def nkernels(fn):
    if DEV != 'cuda': return None
    from torch.profiler import profile, ProfilerActivity
    fn(); sync()
    with profile(activities=[ProfilerActivity.CUDA]) as p: fn(); sync()
    return sum(1 for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA)
R = {'device': DEV, 'batch': [B, T], 'reps': REPS, 'steps': STEPS, 'note': 'timings share the GPU with any other running job' if DEV == 'cuda' else 'cpu dry run'}
modes = {'scan_kernel_ON': 'cuscan', 'scan_log_depth_OFF': 'hs'}; ms = {k: [] for k in modes}; ms['transformer'] = []
for _ in range(REPS):
    for k, v in modes.items(): os.environ['SCAN_MODE'] = v; ms[k].append(timed(stepL))
    ms['transformer'].append(timed(stepT))
for k in ms: R[k + '_ms_per_step_median'] = round(st.median(ms[k]), 2); R[k + '_ms_all'] = [round(v, 1) for v in ms[k]]
for k, v in modes.items(): os.environ['SCAN_MODE'] = v; R[k + '_kernels_per_step'] = nkernels(stepL)
R['transformer_kernels_per_step'] = nkernels(stepT)
try:
    import cuscan; R['cuscan_VERIFIED'] = cuscan.VERIFIED
except Exception as e: R['cuscan_VERIFIED'] = 'n/a ' + repr(e)[:80]
if DEV == 'cuda':
    try:
        import energy as EN
        idle, _ = EN.idle_watts(6, 2); R['idle_W'] = round(idle, 2)
        for k, v in modes.items():
            os.environ['SCAN_MODE'] = v; fn = lambda: [stepL() for _ in range(STEPS)]; fn(); s, _ = EN.repeat(fn, 3, STEPS * B * (T - 1), idle, sync); R[k + '_energy'] = s
        fn = lambda: [stepT() for _ in range(STEPS)]; fn(); s, _ = EN.repeat(fn, 3, STEPS * B * (T - 1), idle, sync); R['transformer_energy'] = s
    except Exception as e: R['energy_error'] = repr(e)[:200]
os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True); json.dump(R, open(os.path.join(HERE, '..', 'results', 'scan_ab.json'), 'w'), indent=1); print(json.dumps(R), flush=True)
