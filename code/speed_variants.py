"""SPEED FIX VERIFICATION (a test, not a benchmark). Same model, same start weights, same batches, 4 ways of running a
training step on the GPU. Reports ms/step and how far each variant's losses drift from the plain version.
  plain   : current code
  graph   : whole forward+backward captured once as a CUDA graph and replayed (removes ~3,300 CPU kernel launches/step)
  bf16    : bfloat16 autocast for the matrix multiplies (Ampere tensor cores), fp32 elsewhere
  graph+bf16
Also: per-component forward time (where the time goes). Writes results/speed_variants.json"""
import os, sys, time, json, copy, torch, torch.nn.functional as F
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import l_model as M0, l5_model as M5, fastmem
dev = 'cuda'; torch.manual_seed(0); B, T, N = 32, 256, int(os.environ.get('NSTEPS', 30))
d = torch.tensor(list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_train.txt'), 'rb').read()[:3000000]), dtype=torch.long)
g = torch.Generator().manual_seed(1); batches = [torch.stack([d[i:i + T + 1] for i in torch.randint(0, len(d) - T - 1, (B,), generator=g)]) for _ in range(N + 5)]
DM = int(os.environ.get('DM', 352))
base = M5.L5Model(d=DM, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True).to(dev); init = copy.deepcopy(base.state_dict())
R = {'gpu': torch.cuda.get_device_name(0), 'params': M0.nparams(base), 'batch': [B, T]}
def make():
    m = M5.L5Model(d=DM, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True).to(dev); m.load_state_dict(init)
    return m, torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.05)
def loss_of(m, x, bf16):
    inp, y = x[:, :-1], x[:, 1:]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        lg = m(inp, fastmem.hippo_batch(inp))[0]
    return F.cross_entropy(lg.float().reshape(-1, 256), y.reshape(-1)) + 0.01 * m.rate_cost()
def run(graph, bf16):
    m, opt = make(); m.train(); losses = []
    if not graph:
        for i, x in enumerate(batches):
            if i == 5: torch.cuda.synchronize(); t0 = time.time()
            x = x.to(dev, non_blocking=True); opt.zero_grad(set_to_none=True); l = loss_of(m, x, bf16); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); losses.append(l.detach())
        torch.cuda.synchronize(); return (time.time() - t0) / N * 1000, torch.stack(losses).cpu()
    sx = batches[0].to(dev).clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                                     # warm-up on a side stream (required before capture)
        for _ in range(3):
            opt.zero_grad(set_to_none=True); l = loss_of(m, sx, bf16); l.backward()
    torch.cuda.current_stream().wait_stream(s)
    m.load_state_dict(init); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.05)   # undo warm-up effects
    opt.zero_grad(set_to_none=True); gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): sl = loss_of(m, sx, bf16); sl.backward()
    losses = []
    for i, x in enumerate(batches):
        if i == 5: torch.cuda.synchronize(); t0 = time.time()
        sx.copy_(x.to(dev, non_blocking=True)); gr.replay()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); losses.append(sl.detach().clone())
    torch.cuda.synchronize(); return (time.time() - t0) / N * 1000, torch.stack(losses).cpu()
ref = None
for name, gph, bf in [('plain', False, False), ('bf16', False, True), ('graph', True, False), ('graph+bf16', True, True)]:
    try:
        ms, L = run(gph, bf)
        if ref is None: ref = L
        R[name] = {'ms_per_step': round(ms, 1), 'max_loss_diff_vs_plain': round((L - ref).abs().max().item(), 4), 'final_loss': round(L[-1].item(), 4)}
    except Exception as e:
        R[name] = {'error': repr(e)[:400]}
    print(name, R[name], flush=True); torch.cuda.empty_cache()
# where the time goes (forward only, per component, synchronised timing)
m, _ = make(); x = batches[0].to(dev); inp = x[:, :-1]; times = {}
def hook_pair(mod, key):
    def pre(*a): torch.cuda.synchronize(); mod._t = time.time()
    def post(*a): torch.cuda.synchronize(); times[key] = times.get(key, 0) + (time.time() - mod._t) * 1000
    mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)
for i in range(6): hook_pair(m.fw[i], 'delta memory (x6)'); hook_pair(m.layers[i], 'spiking layer (x6)')
with torch.no_grad():
    for _ in range(3): m(inp, fastmem.hippo_batch(inp))
    times.clear(); torch.cuda.synchronize(); t0 = time.time(); m(inp, fastmem.hippo_batch(inp)); torch.cuda.synchronize(); tot = (time.time() - t0) * 1000
R['forward breakdown ms'] = {k: round(v, 1) for k, v in times.items()}; R['forward breakdown ms']['total'] = round(tot, 1)
print(json.dumps(R, indent=1)); json.dump(R, open(os.path.join(OUT, 'speed_variants.json'), 'w'), indent=1)
