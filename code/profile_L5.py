"""GPU profile of one training step: transformer vs L-5 (old code path vs new code path) + per-component breakdown.
Waits until the GPU is free (WAITFOR file) so timings are clean. Writes results/profile_L5.json."""
import os, sys, time, json, torch, torch.nn.functional as F
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
import l_model as M0, l1_model as M1, l4_model as M4, l5_model as M5, fastmem
dev = os.environ.get('PDEV', 'cuda'); sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None); torch.manual_seed(0); B, T = 16, 256
X = torch.randint(0, 256, (B, T + 1), device=dev)
d = open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_val.txt'), 'rb').read()[:200000]
real = torch.tensor(list(d), dtype=torch.long)
ix = torch.randint(0, len(real) - T - 1, (B,)); X = torch.stack([real[i:i + T + 1] for i in ix]).to(dev)
R = {'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', 'batch': [B, T]}
def timeit(fn, n=10):
    for _ in range(3): fn()
    sync(); t = time.time()
    for _ in range(n): fn()
    sync(); return round((time.time() - t) / n * 1000, 1)
def step(m, kind):
    inp, y = X[:, :-1], X[:, 1:]
    def f():
        if kind == 'T': lg = m(inp)[0]
        else:
            rec = fastmem.hippo_batch(inp) if M4.FASTMEM else M0.Hippocampus.batch(inp.cpu()).to(dev)
            lg = m(inp, rec)[0]
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        if kind != 'T': loss = loss + 0.01 * m.rate_cost()
        loss.backward()
    return f
mT = M0.TModel(d=352, n_layers=6, T=T).to(dev); R['T_ms_per_microbatch'] = timeit(step(mT, 'T')); del mT
m5 = M5.L5Model(d=352, n_layers=6, heads=8).to(dev)
M5.DELTA_V, M4.FASTMEM = 1, 0; R['L5_old_ms_per_microbatch'] = timeit(step(m5, 'L'))
M5.DELTA_V, M4.FASTMEM = 2, 0; R['L5_newdelta_only_ms'] = timeit(step(m5, 'L'))
M5.DELTA_V, M4.FASTMEM = 2, 1; R['L5_new_ms_per_microbatch'] = timeit(step(m5, 'L'))
# component breakdown (fwd+bwd), new path
z = torch.randn(B, T, 352, device=dev, requires_grad=True)
R['one_delta_memory_ms'] = timeit(lambda: m5.fw[0](z).sum().backward())
M5.DELTA_V = 1; R['one_delta_memory_old_ms'] = timeit(lambda: m5.fw[0](z).sum().backward()); M5.DELTA_V = 2
R['one_spiking_layer_ms'] = timeit(lambda: m5.layers[0](z)[0].sum().backward())
inp = X[:, :-1]
R['hippo_python_cpu_ms'] = timeit(lambda: M0.Hippocampus.batch(inp.cpu()), 5)
R['hippo_gpu_ms'] = timeit(lambda: fastmem.hippo_batch(inp))
R['traj_python_cpu_ms'] = timeit(lambda: M1.Trajectory.batch(inp.cpu()), 5)
R['traj_gpu_ms'] = timeit(lambda: fastmem.traj_batch(inp))
json.dump(R, open(os.path.join(OUT, os.environ.get('PNAME', 'profile_L5') + '.json'), 'w'), indent=1)
try:
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == 'cuda' else [])
    with profile(activities=acts) as prof: step(m5, 'L')(); sync()
    key = lambda e: getattr(e, 'device_time_total', getattr(e, 'cuda_time_total', 0))
    R['top_ops [name, gpu_ms, cpu_ms, calls]'] = [[e.key, round(key(e) / 1000, 1), round(e.cpu_time_total / 1000, 1), e.count]
                                                   for e in sorted(prof.key_averages(), key=lambda e: -(key(e) or e.cpu_time_total))[:25]]
except Exception as ex: R['profiler_error'] = repr(ex)
print(json.dumps(R, indent=1)); json.dump(R, open(os.path.join(OUT, os.environ.get('PNAME', 'profile_L5') + '.json'), 'w'), indent=1)
