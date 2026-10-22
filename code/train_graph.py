"""Training-energy levers for L vs the transformer (MEASURED, NVML): eager fp32 | bf16 autocast | whole-step CUDA graph (| graph+bf16).
Why: old GPU profile shows the L training step is elementwise/launch-bound (3290 kernel launches), not matmul-bound.
Batch 16 x 256, d352, 6 layers (same configs as eff_bench). Each variant: 3 warm repeats of 10 steps. A variant that fails to capture is reported, not hidden.
Writes LOUT/train_graph.json. Usage: python train_graph.py"""
import os, sys, time, json, math, torch
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import torch.nn.functional as F
import l_model as M0, l5_model as M5, fastmem, energy as EN
dev = 'cuda'; sync = torch.cuda.synchronize; B, Tn, NS, REP = 16, 256, 10, 3
data = torch.tensor(list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_val.txt'), 'rb').read()[:400000]), dtype=torch.long)
def batch():
    ix = torch.randint(0, len(data) - Tn - 1, (B,)); return torch.stack([data[i:i + Tn + 1] for i in ix]).to(dev)
mk = {'L': lambda: M5.L5Model(d=352, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True), 'T': lambda: M0.TModel(d=352, n_layers=6, h=4, T=Tn)}
R = {'gpu': EN.gpu_info(), 'torch': torch.__version__, 'batch': [B, Tn], 'labels': 'MEASURED'}; R['idle_W'], _ = EN.idle_watts(); IDLE = R['idle_W']
def path(kind, m, x, bf):
    inp = x[:, :-1]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf):
        lg = m(inp, fastmem.hippo_batch(inp))[0] if kind == 'L' else m(inp)[0]
    return F.cross_entropy(lg.float().reshape(-1, 256), x[:, 1:].reshape(-1))
def run(kind, bf, graph):
    torch.manual_seed(0); m = mk[kind]().to(dev).train(); opt = torch.optim.AdamW(m.parameters(), 1e-4, capturable=graph)
    xs = [batch() for _ in range(NS)]; losses = []
    if not graph:
        def f():
            for x in xs: loss = path(kind, m, x, bf); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item() if len(losses) < 3 else 0)
    else:
        sx = xs[0].clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): opt.zero_grad(set_to_none=True); path(kind, m, sx, bf).backward(); opt.step()
        sync(); g = torch.cuda.CUDAGraph(); opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(g): sl = path(kind, m, sx, bf); sl.backward(); opt.step()
        def f():
            for x in xs: sx.copy_(x); g.replay(); losses.append(sl.item() if len(losses) < 3 else 0)
    f(); sync(); losses.clear(); torch.cuda.reset_peak_memory_stats()
    summ, _ = EN.repeat(f, REP, NS * B * Tn, IDLE, sync)
    out = {**summ, 'ms_per_step': round(summ['seconds_mean'] / NS * 1000, 1), 'peak_VRAM_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2), 'first_losses': [round(l, 4) for l in losses[:3]]}
    del m, opt; torch.cuda.empty_cache(); return out
for kind in ('L', 'T'):
    for name, bf, gr in [('eager_fp32', False, False), ('bf16', True, False), ('cudagraph_fp32', False, True), ('cudagraph_bf16', True, True)]:
        key = f'{kind}_{name}'
        try: R[key] = run(kind, bf, gr)
        except Exception as e: R[key] = {'failed': f'{type(e).__name__}: {str(e)[:300]}'}; torch.cuda.empty_cache()
        print(key, {k: R[key].get(k) for k in ('ms_per_step', 'mJ_per_token_mean', 'mJ_per_token_above_idle_mean', 'peak_VRAM_GB', 'failed')}, flush=True)
        json.dump(R, open(os.path.join(OUT, 'train_graph.json'), 'w'), indent=1)
print('done', flush=True)
