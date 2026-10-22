"""Fair transformer baseline for the energy comparison: SAME optimisations given to the transformer (torch.compile, bf16).
Same B,T,token count, same energy.py method as compile_ab.py. TModel d=152 (param-matched control). One subprocess per variant.
Run: python -B tf_ab.py   -> ../results/tf_ab.json.  Close other GPU jobs first for clean numbers."""
import os, sys, json, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results'); os.makedirs(RES, exist_ok=True)
V = {'t_eager': dict(comp=0, bf16=0), 't_comp': dict(comp=1, bf16=0), 't_comp_bf16': dict(comp=1, bf16=1), 't_eager_bf16': dict(comp=0, bf16=1)}
if len(sys.argv) == 1:
    out = {}
    for n in V:
        t = time.time(); p = subprocess.run([sys.executable, '-B', os.path.abspath(__file__), n], capture_output=True, text=True, cwd=HERE)
        f = os.path.join(RES, f'tf_ab_{n}.json'); out[n] = json.load(open(f)) if os.path.exists(f) else {'error': (p.stderr or p.stdout)[-600:]}
        print(n, 'done in', round(time.time() - t), 's:', json.dumps({k: out[n].get(k) for k in ('ms_per_step_median', 'kernels_per_step', 'mJ_per_token', 'avg_W', 'error')}), flush=True)
        json.dump(out, open(os.path.join(RES, 'tf_ab.json'), 'w'), indent=1)
    sys.exit(0)
name = sys.argv[1]; C = V[name]; sys.path[:0] = [HERE]
import statistics as st, torch, torch.nn.functional as F
import l_model as LM
DEV = 'cuda'; B, T = 16, 256; STEPS = 10; REPS = 5
torch.manual_seed(0); X = torch.randint(0, 256, (B, T)).to(DEV)
torch.manual_seed(1); m = LM.TModel(d=152, n_layers=4, h=4, T=T).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3)
R = {'variant': name, 'config': C, 'params': LM.nparams(m), 'device': torch.cuda.get_device_name(0)}
f = torch.compile(m) if C['comp'] else m
def step():
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(C['bf16'])): lg = f(X[:, :-1])[0].float()
    loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
t0 = time.time(); step(); torch.cuda.synchronize(); R['compile_s'] = round(time.time() - t0, 1)
for _ in range(3): step()
def timed():
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(STEPS): step()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / STEPS * 1000
ms = [timed() for _ in range(REPS)]; R['ms_per_step_median'] = round(st.median(ms), 2)
from torch.profiler import profile, ProfilerActivity
step(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p: step(); torch.cuda.synchronize()
R['kernels_per_step'] = sum(1 for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA)
try:
    import energy as EN; idle, _ = EN.idle_watts(6, 2); fn = lambda: [step() for _ in range(STEPS)]; fn(); s, _ = EN.repeat(fn, 3, STEPS * B * (T - 1), idle, torch.cuda.synchronize)
    R['mJ_per_token'] = s['mJ_per_token_mean']; R['mJ_per_token_std'] = s['mJ_per_token_std']; R['avg_W'] = s['avg_W_mean']
except Exception as e: R['energy_error'] = repr(e)[:200]
json.dump(R, open(os.path.join(RES, f'tf_ab_{name}.json'), 'w'), indent=1); print(json.dumps(R), flush=True)
