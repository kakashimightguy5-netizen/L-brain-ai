"""Energy fix: where does L5's GPU time go? (GPU job) One fp32 training step, scan kernel ON. Reports (1) forward CUDA-event time per module family
(delta memory x4, spiking blocks x4, hippocampus/embeddings/head), (2) total forward / backward / optimizer, (3) top CUDA kernels by total GPU time and call count.
Writes ../results/profile_modules.json. No power/clock settings changed. Shares the GPU with other running jobs (timings indicative)."""
import os, sys, json, time, collections, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]; os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, fastmem
assert torch.cuda.is_available(); DEV = 'cuda'; B, T = 16, 256
torch.manual_seed(0); X = torch.randint(0, 256, (B, T)); REC = fastmem.hippo_batch(X).to(DEV); X = X.to(DEV)
m = M5.L5Model(d=128, n_layers=4, heads=4).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3)
fam = collections.defaultdict(float); ev = []
def hook_pre(name):
    def f(mod, inp): a = torch.cuda.Event(enable_timing=True); a.record(); mod._t0 = a
    return f
def hook_post(name):
    def f(mod, inp, out): b = torch.cuda.Event(enable_timing=True); b.record(); ev.append((name, mod._t0, b))
    return f
for i, l in enumerate(m.layers): l.register_forward_pre_hook(hook_pre('spiking_block')); l.register_forward_hook(hook_post('spiking_block'))
for i, l in enumerate(m.fw): l.register_forward_pre_hook(hook_pre('delta_memory')); l.register_forward_hook(hook_post('delta_memory'))
def step():
    ev.clear(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True); c = torch.cuda.Event(enable_timing=True); d = torch.cuda.Event(enable_timing=True)
    a.record(); lg = m(X[:, :-1], REC[:, :, :-1])[0]; loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1)); b.record()
    opt.zero_grad(); loss.backward(); c.record(); opt.step(); d.record(); torch.cuda.synchronize()
    return a.elapsed_time(b), b.elapsed_time(c), c.elapsed_time(d)
for _ in range(3): step()
R = {'device': torch.cuda.get_device_name(0), 'batch': [B, T], 'note': 'shares the GPU with other running jobs'}
fw, bw, op = zip(*[step() for _ in range(5)]); R['forward_ms'] = round(sorted(fw)[2], 2); R['backward_ms'] = round(sorted(bw)[2], 2); R['optimizer_ms'] = round(sorted(op)[2], 2)
step(); tot = collections.defaultdict(float)
for name, a, b in ev: tot[name] += a.elapsed_time(b)
R['forward_ms_by_module_family'] = {k: round(v, 2) for k, v in tot.items()}
R['forward_ms_other_(embeddings,hippocampus_boost,head,loss)'] = round(R['forward_ms'] - sum(tot.values()), 2)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p: step(); torch.cuda.synchronize()
ks = collections.defaultdict(lambda: [0.0, 0])
for e in p.events():
    if e.device_type == torch.autograd.DeviceType.CUDA: ks[e.name][0] += e.cuda_time_total if hasattr(e, 'cuda_time_total') else e.device_time_total; ks[e.name][1] += 1
top = sorted(ks.items(), key=lambda kv: -kv[1][0])[:20]; allt = sum(v[0] for v in ks.values())
R['cuda_kernel_total_ms'] = round(allt / 1000, 2); R['kernel_launches'] = sum(v[1] for v in ks.values())
R['top_kernels'] = [{'name': n[:90], 'ms': round(v[0] / 1000, 2), 'calls': v[1], 'pct': round(100 * v[0] / allt, 1)} for n, v in top]
os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True); json.dump(R, open(os.path.join(HERE, '..', 'results', 'profile_modules.json'), 'w'), indent=1); print(json.dumps(R), flush=True)
