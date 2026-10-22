"""Energy fix F1 A/B (GPU job): torch.compile (Triton) on L5's delta-memory and spiking blocks vs the eager baseline. One subprocess per variant (clean caches), identical init + data.
Variants: eager | c_cuscan | c_hs | c_cuscan_bf16 | c_hs_bf16.   c_* = compiled blocks + SPIKE_STE=1 (compile-friendly straight-through spikes, bit-identical forward/gradient to the original);
cuscan = verified CuPy scan kernel (kept outside the compiled graph), hs = log-depth scan fused by the compiler. Equivalence: logits and one-step gradients vs eager.
Run: python -B compile_ab.py            (all variants, ~10-15 min incl. compile)   |   python -B compile_ab.py <variant>   (one variant)
Writes ../results/compile_ab.json. No power/clock settings changed. GPU timings/energy are indicative while other jobs share the GPU."""
import os, sys, json, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results'); os.makedirs(RES, exist_ok=True)
V = {'eager': dict(ste=0, scan='cuscan', bf16=0, comp=0), 'c_cuscan': dict(ste=1, scan='cuscan', bf16=0, comp=1), 'c_hs': dict(ste=1, scan='hs', bf16=0, comp=1),
     'c_cuscan_bf16': dict(ste=1, scan='cuscan', bf16=1, comp=1), 'c_hs_bf16': dict(ste=1, scan='hs', bf16=1, comp=1)}
if len(sys.argv) == 1:
    out = {}
    for name in V:
        t = time.time(); p = subprocess.run([sys.executable, '-B', os.path.abspath(__file__), name], capture_output=True, text=True, cwd=HERE)
        f = os.path.join(RES, f'compile_ab_{name}.json')
        out[name] = json.load(open(f)) if os.path.exists(f) else {'error': (p.stderr or p.stdout)[-600:]}
        print(name, 'done in', round(time.time() - t), 's:', json.dumps({k: out[name].get(k) for k in ('ms_per_step_median', 'kernels_per_step', 'mJ_per_token', 'max_logit_diff_vs_eager', 'grad_cosine_vs_eager', 'compile_s', 'error')}), flush=True)
        json.dump(out, open(os.path.join(RES, 'compile_ab.json'), 'w'), indent=1)
    sys.exit(0)
name = sys.argv[1]; C = V[name]
os.environ['SPIKEK'] = '8'; os.environ['LSCAN_DISABLE'] = '1'; os.environ['SPIKE_STE'] = str(C['ste']); os.environ['SCAN_MODE'] = C['scan']
sys.path[:0] = [HERE]
import statistics as st, torch, torch.nn.functional as F
import l5_model as M5, fastmem
import torch._dynamo as dyn; dyn.config.cache_size_limit = 64
if C['scan'] == 'cuscan':
    import cuscan; cuscan.scan = dyn.disable(cuscan.scan)
DEV = 'cuda'; B, T = 16, 256; STEPS = 10; REPS = 5
torch.manual_seed(0); X = torch.randint(0, 256, (B, T)); REC = fastmem.hippo_batch(X).to(DEV); X = X.to(DEV)
torch.manual_seed(1); m = M5.L5Model(d=128, n_layers=4, heads=4).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3)
R = {'variant': name, 'config': C, 'device': torch.cuda.get_device_name(0), 'batch': [B, T], 'note': 'shares the GPU with other running jobs'}
if C['comp']:
    for l in list(m.layers) + list(m.fw): l.compile()
def fwd():
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(C['bf16'])): lg = m(X[:, :-1], REC[:, :, :-1])[0]
    return lg.float()
def step():
    lg = fwd(); loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
# equivalence at identical init (before any update)
t0 = time.time(); lg = fwd(); loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1)); m.zero_grad(); loss.backward(); torch.cuda.synchronize(); R['compile_s'] = round(time.time() - t0, 1)
g = torch.cat([p.grad.flatten().float() for p in m.parameters() if p.grad is not None]).cpu(); lgc = lg.detach().cpu(); R['init_loss'] = round(loss.item(), 5)
ref = os.path.join(RES, '_ref_eager.pt')
if name == 'eager': torch.save({'lg': lgc, 'g': g}, ref)
elif os.path.exists(ref):
    r = torch.load(ref); R['max_logit_diff_vs_eager'] = float((lgc - r['lg']).abs().max()); R['grad_cosine_vs_eager'] = float(torch.dot(g, r['g']) / (g.norm() * r['g'].norm()))
for _ in range(3): step()
import torch as _t
def timed():
    _t.cuda.synchronize(); t = time.perf_counter()
    for _ in range(STEPS): step()
    _t.cuda.synchronize(); return (time.perf_counter() - t) / STEPS * 1000
ms = [timed() for _ in range(REPS)]; R['ms_per_step_median'] = round(st.median(ms), 2); R['ms_all'] = [round(v, 1) for v in ms]
from torch.profiler import profile, ProfilerActivity
step(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p: step(); torch.cuda.synchronize()
R['kernels_per_step'] = sum(1 for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA)
try:
    import energy as EN; idle, _ = EN.idle_watts(6, 2); fn = lambda: [step() for _ in range(STEPS)]; fn(); s, _ = EN.repeat(fn, 3, STEPS * B * (T - 1), idle, torch.cuda.synchronize)
    R['mJ_per_token'] = s['mJ_per_token_mean']; R['mJ_per_token_std'] = s['mJ_per_token_std']; R['avg_W'] = s['avg_W_mean']; R['throttle'] = s['throttle_reasons']
except Exception as e: R['energy_error'] = repr(e)[:200]
R['last_loss'] = round(float(F.cross_entropy(fwd().detach().reshape(-1, 256), X[:, 1:].reshape(-1))), 4)
json.dump(R, open(os.path.join(RES, f'compile_ab_{name}.json'), 'w'), indent=1); print(json.dumps(R), flush=True)
