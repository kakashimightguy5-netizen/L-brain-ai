"""Phase 2 measurement: each optimisation flag ON vs the reference path, same weights/batches.
Reports ms/step, peak VRAM, and max |loss - reference loss| over the steps (exact flags must give 0)."""
import os, sys, time, json, copy, torch, torch.nn.functional as F
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import l_model as M0, l5_model as M5, fastmem
dev = 'cuda'; T, B, N, DM = 256, 32, 20, int(os.environ.get('DM', 352))
data = torch.tensor(list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_train.txt'), 'rb').read()[:3000000]), dtype=torch.long)
g = torch.Generator().manual_seed(3); xs = [torch.stack([data[i:i + T + 1] for i in torch.randint(0, len(data) - T - 1, (B,), generator=g)]).to(dev) for _ in range(N + 3)]
torch.manual_seed(0); init = M5.L5Model(d=DM, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True).state_dict()
def run(mask=False, scan='cuscan', bf16=False):
    M0.SpikeCount.MASK = mask; os.environ['SCAN_MODE'] = scan
    torch.manual_seed(0); m = M5.L5Model(d=DM, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True).to(dev); m.load_state_dict(init)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.05); L = []
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for i, x in enumerate(xs):
        if i == 3: torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True); inp = x[:, :-1]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16): lg = m(inp, fastmem.hippo_batch(inp))[0]
        loss = F.cross_entropy(lg.float().reshape(-1, 256), x[:, 1:].reshape(-1)) + 0.01 * m.rate_cost(); loss.backward(); opt.step(); L.append(loss.item())
    torch.cuda.synchronize(); ms = (time.time() - t0) / N * 1000
    return {'ms_per_step': round(ms, 1), 'peak_VRAM_GB': round(torch.cuda.max_memory_allocated() / 1e9, 3), 'losses': L}
R = {}; ref = run(); R['reference'] = ref
R['reference (repeat, measures GPU non-determinism)'] = run()
for name, kw in [('SPIKE_MASK=1', dict(mask=True)), ('SCAN_MODE=hs', dict(scan='hs')), ('bf16 autocast', dict(bf16=True)),
                 ('SPIKE_MASK=1 + bf16', dict(mask=True, bf16=True))]:
    try: R[name] = run(**kw)
    except Exception as e: R[name] = {'error': repr(e)[:300]}
REFL = list(ref['losses'])
for k, v in R.items():
    if 'losses' in v: v['max_loss_diff_vs_reference'] = round(max(abs(a - b) for a, b in zip(v['losses'], REFL)), 5); v['final_loss'] = round(v['losses'][-1], 4); del v['losses']
    print(k, v, flush=True)
json.dump(R, open(os.path.join(OUT, 'flag_sweep.json'), 'w'), indent=1)
