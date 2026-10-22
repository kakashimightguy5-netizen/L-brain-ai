"""Phase 0/1 measurement harness (GPU). One run measures, for L (current default = L-6 config) and the transformer baseline:
  train: ms/step, tokens/s, peak VRAM, CUDA kernel launches per step (profiler), per-module time breakdown
  infer: full-window forward throughput at batch 1 / 8 / 64 (L has no token-by-token decode path yet -> reported as such)
  energy: NVML total-energy counter AND integrated power samples, above-idle, per token
  firing: per-layer fraction of non-zero spikes (state + FFN) at batch 1 and batch 32, plus the UNION across the batch
  hippocampus/trajectory share of step time
Writes results/bench_<tag>.json. Usage: python l_bench.py TAG"""
import os, sys, time, json, threading, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('LOUT', '.'); tag = sys.argv[1] if len(sys.argv) > 1 else 'run'
wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import l_model as M0, l5_model as M5, fastmem
dev = 'cuda'; T = 256; DM = int(os.environ.get('DM', 352)); NL = 6
data = torch.tensor(list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_val.txt'), 'rb').read()[:2000000]), dtype=torch.long)
g = torch.Generator().manual_seed(7)
def batch(B): return torch.stack([data[i:i + T + 1] for i in torch.randint(0, len(data) - T - 1, (B,), generator=g)]).to(dev)
def build(kind):
    torch.manual_seed(0)
    if kind == 'T': return M0.TModel(d=DM, n_layers=NL, T=T).to(dev)
    return M5.L5Model(d=DM, n_layers=NL, heads=8, conv=4, colpos=True, mixcopy=True).to(dev)
def fwd(m, kind, x):
    inp = x[:, :-1]
    return m(inp)[0] if kind == 'T' else m(inp, fastmem.hippo_batch(inp))[0]
def step_fn(m, kind, opt):
    def f(x):
        opt.zero_grad(set_to_none=True); lg = fwd(m, kind, x)
        loss = F.cross_entropy(lg.float().reshape(-1, 256), x[:, 1:].reshape(-1)) + (0.01 * m.rate_cost() if kind != 'T' else 0)
        loss.backward(); opt.step(); return loss
    return f
# ---------------- NVML energy ----------------
try:
    import pynvml
except ImportError:
    import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nvidia-ml-py']); import pynvml
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def energy_counter():
    try: return pynvml.nvmlDeviceGetTotalEnergyConsumption(H) / 1000.0   # J
    except Exception: return None
class PowerSampler:
    def __init__(s): s.samples = []; s.run = False
    def _loop(s):
        while s.run: s.samples.append((time.time(), pynvml.nvmlDeviceGetPowerUsage(H) / 1000.0)); time.sleep(0.01)
    def __enter__(s): s.run = True; s.t = threading.Thread(target=s._loop, daemon=True); s.t.start(); return s
    def __exit__(s, *a):
        s.run = False; s.t.join()
    def joules(s):
        sm = s.samples; return sum((sm[i + 1][0] - sm[i][0]) * (sm[i][1] + sm[i + 1][1]) / 2 for i in range(len(sm) - 1))
def idle_watts(sec=5):
    torch.cuda.synchronize(); time.sleep(1)
    with PowerSampler() as p: time.sleep(sec)
    return p.joules() / sec
def measure_energy(fn, n, tokens_per_call):
    torch.cuda.synchronize(); e0 = energy_counter(); t0 = time.time()
    with PowerSampler() as p:
        for _ in range(n): fn()
        torch.cuda.synchronize()
    dt = time.time() - t0; e1 = energy_counter(); J = p.joules(); toks = n * tokens_per_call
    return {'seconds': round(dt, 2), 'tokens': toks, 'J_power_samples': round(J, 2), 'J_counter': round(e1 - e0, 2) if e0 is not None else None,
            'mJ_per_token_total': round(J / toks * 1000, 4), 'mJ_per_token_above_idle': round((J - IDLE * dt) / toks * 1000, 4), 'avg_W': round(J / dt, 1)}
R = {'gpu': torch.cuda.get_device_name(0), 'tag': tag, 'd': DM, 'layers': NL, 'note': 'all numbers measured on this machine in this run'}
IDLE = idle_watts(); R['idle_W'] = round(IDLE, 2)
for kind in ('L', 'T'):
    m = build(kind); opt = torch.optim.AdamW(m.parameters(), 2e-3); f = step_fn(m, kind, opt); r = {'params': M0.nparams(m)}
    B = 32; xs = [batch(B) for _ in range(25)]
    for x in xs[:5]: f(x)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for x in xs[5:]: f(x)
    torch.cuda.synchronize(); ms = (time.time() - t0) / 20 * 1000
    r['train'] = {'batch': B, 'ms_per_step': round(ms, 1), 'tokens_per_s': round(B * T / ms * 1000), 'peak_VRAM_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof: f(xs[0]); torch.cuda.synchronize()
    ka = prof.key_averages(); r['train']['kernel_launches_per_step'] = int(sum(e.count for e in ka if e.key in ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC')))
    r['train']['energy'] = measure_energy(lambda: f(xs[1]), 20, B * T)
    if kind == 'L':   # per-module forward breakdown (synchronised hooks; forward only)
        times = {}
        def hook(mod, key):
            def pre(*a):
                torch.cuda.synchronize(); mod._t = time.time()
            def post(*a):
                torch.cuda.synchronize(); times[key] = times.get(key, 0) + (time.time() - mod._t) * 1000
            mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)
        for i in range(NL): hook(m.fw[i], 'fast-weight memory'); hook(m.layers[i], 'membrane+spikes+FFN layer')
        x = xs[2]; inp = x[:, :-1]
        with torch.no_grad():
            fwd(m, kind, x); times.clear(); torch.cuda.synchronize(); t0 = time.time()
            rec = fastmem.hippo_batch(inp); torch.cuda.synchronize(); th = (time.time() - t0) * 1000
            t1 = time.time(); tr = fastmem.traj_batch(inp); torch.cuda.synchronize(); tt = (time.time() - t1) * 1000
            t2 = time.time(); m(inp, rec); torch.cuda.synchronize(); tot = (time.time() - t2) * 1000
        r['forward_breakdown_ms'] = {k: round(v, 1) for k, v in times.items()}
        r['forward_breakdown_ms'].update({'hippocampus index': round(th, 1), 'trajectory replay': round(tt, 1), 'model forward total (incl. replay)': round(tot, 1)})
        # firing rates incl. union across the batch
        m.eval(); fr = {}
        for Bf in (1, 32):
            caps = []
            x = batch(Bf); inp = x[:, :-1]; rates = []
            orig = M0.spikes
            store = []
            def spy(xx, lo, hi):
                c = orig(xx, lo, hi); store.append(c.detach()); return c
            M0.spikes = spy
            with torch.no_grad(): m(inp, fastmem.hippo_batch(inp))
            M0.spikes = orig
            per = []
            for li in range(0, len(store), 2):
                st, ff = store[li], store[li + 1]
                per.append({'state_active': round((st != 0).float().mean().item(), 4), 'ffn_active': round((ff != 0).float().mean().item(), 4),
                            'ffn_union_over_batch_per_token': round((ff != 0).any(0).float().mean().item(), 4) if Bf > 1 else None})
            fr[f'batch {Bf}'] = per
        r['firing'] = fr; m.train()
    # inference throughput (full-window forward, no grad) at batch 1/8/64
    m.eval(); inf = {}
    for Bi in (1, 8, 64):
        x = batch(Bi)
        with torch.no_grad():
            for _ in range(3): fwd(m, kind, x)
            e = measure_energy(lambda: fwd(m, kind, x), 10 if Bi < 64 else 5, Bi * T)
        inf[f'batch {Bi}'] = {'tokens_per_s': round(e['tokens'] / e['seconds']), 'mJ_per_token_above_idle': e['mJ_per_token_above_idle'], 'mJ_per_token_total': e['mJ_per_token_total']}
    r['inference_full_window'] = inf
    if kind == 'L':   # real autoregressive path: token-by-token Stepper (constant work per byte)
        import l6_step; dec = {}
        for Bi in (1, 8, 64):
            x = batch(Bi)[:, :257].tolist(); st = l6_step.Stepper(m, B=Bi)
            for t in range(16): st.step([row[t] for row in x])
            cnt = [16]
            def run_some(n=48):
                for _ in range(n):
                    t = cnt[0]; st.step([row[t] for row in x]); cnt[0] += 1
            e = measure_energy(lambda: run_some(), 4, Bi * 48)
            dec[f'batch {Bi}'] = {'tokens_per_s': round(e['tokens'] / e['seconds']), 'ms_per_step': round(e['seconds'] / (4 * 48) * 1000, 2), 'mJ_per_token_above_idle': e['mJ_per_token_above_idle']}
        r['inference_token_by_token'] = dec
    R[kind] = r; print(kind, json.dumps(r, indent=1), flush=True)
    del m, opt; torch.cuda.empty_cache()
R['wall_plug'] = 'For a wall-plug check: read the meter (W) at idle for 1 min, then during each phase above; energy = (W_load - W_idle) x seconds.'
json.dump(R, open(os.path.join(OUT, f'bench_{tag}.json'), 'w'), indent=1); print(json.dumps(R, indent=1))
