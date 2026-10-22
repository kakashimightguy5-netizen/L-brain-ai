"""Energy fix quality gate (GPU job): does compile / bf16 change MODEL QUALITY? Uses the trained baseline LEdge_s0.pt (d=128, 4 layers) and the fixed 6x1024 held-out windows (second half of py_big_val, same as compress_ft/fix_eval).
Per variant (own subprocess): (1) bits/byte of the trained model under that numeric path (inference drift only); (2) 600 CE fine-tune steps (AdamW 5e-4, batch 16 x 256, SAME batches for every variant), bits/byte again; ms/step.
Variants: eager (original fp32) | c_fp32 (compiled blocks, SPIKE_STE=1) | c_bf16 (compiled blocks + bf16 autocast). Scan = verified CuPy kernel.
Gate: |bpb - eager| / eager <= 0.5%. Run: python -B quality_ab.py   (all)  |  python -B quality_ab.py <variant>. Writes ../results/quality_ab.json. No power/clock settings changed."""
import os, sys, json, subprocess, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results'); os.makedirs(RES, exist_ok=True)
V = {'eager': dict(ste=0, bf16=0, comp=0), 'c_fp32': dict(ste=1, bf16=0, comp=1), 'c_bf16': dict(ste=1, bf16=1, comp=1)}
if len(sys.argv) == 1:
    out = {}
    for n in V:
        t = time.time(); p = subprocess.run([sys.executable, '-B', os.path.abspath(__file__), n], capture_output=True, text=True, cwd=HERE); f = os.path.join(RES, f'quality_ab_{n}.json')
        out[n] = json.load(open(f)) if os.path.exists(f) else {'error': (p.stderr or p.stdout)[-600:]}
        e = out.get('eager', {}); r = out[n]
        for k in ('bpb_before', 'bpb_after'):
            if k in r and k in e: r[k + '_rel_vs_eager_pct'] = round((r[k] - e[k]) / e[k] * 100, 3)
        print(n, 'done in', round(time.time() - t), 's:', json.dumps({k: r.get(k) for k in ('bpb_before', 'bpb_before_rel_vs_eager_pct', 'bpb_after', 'bpb_after_rel_vs_eager_pct', 'ms_per_step', 'error')}), flush=True)
        json.dump(out, open(os.path.join(RES, 'quality_ab.json'), 'w'), indent=1)
    sys.exit(0)
name = sys.argv[1]; C = V[name]
os.environ['SPIKEK'] = '8'; os.environ['LSCAN_DISABLE'] = '1'; os.environ['SPIKE_STE'] = str(C['ste']); os.environ['SCAN_MODE'] = 'cuscan'; sys.path[:0] = [HERE]
import torch, torch.nn.functional as F
import l5_model as M5, fastmem, cuscan, torch._dynamo as dyn
dyn.config.cache_size_limit = 64; cuscan.scan = dyn.disable(cuscan.scan)
DEV = 'cuda'; DATA = os.environ.get('LDATA', os.path.join(HERE, '..', 'data')); ld = lambda p: torch.tensor(list(open(p, 'rb').read()), dtype=torch.long)
tr, va = ld(os.path.join(DATA, 'py_big_train.txt')), ld(os.path.join(DATA, 'py_big_val.txt')); half = len(va) // 2
XE = torch.stack([va[half + i * 1024: half + i * 1024 + 1025] for i in range(6)]).to(DEV)
ck = next(p for p in (os.path.join(RES, 'LEdge_s0.pt'), os.path.join(HERE, 'LEdge_s0.pt')) if os.path.exists(p))
m = M5.L5Model(d=128, n_layers=4, heads=4); m.load_state_dict(torch.load(ck, map_location='cpu')); m.to(DEV)
if C['comp']:
    for l in list(m.layers) + list(m.fw): l.compile()
def bpb():
    m.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(C['bf16'])):
        inp, y = XE[:, :-1], XE[:, 1:]; lp = m(inp, fastmem.hippo_batch(inp).to(DEV))[0].float().log_softmax(-1); return -lp.gather(-1, y[..., None]).mean().item() / math.log(2)
R = {'variant': name, 'config': C, 'checkpoint': os.path.basename(ck), 'eval_windows': '6x1024 second half of py_big_val'}
R['bpb_before'] = round(bpb(), 5); print(name, 'bpb before', R['bpb_before'], flush=True)
opt = torch.optim.AdamW(m.parameters(), 5e-4, weight_decay=0.0); g = torch.Generator().manual_seed(123); STEPS = 600; ts = []
for it in range(STEPS):
    m.train(); ix = torch.randint(0, len(tr) - 257, (16,), generator=g); x = torch.stack([tr[i:i + 257] for i in ix]).to(DEV); inp, y = x[:, :-1], x[:, 1:]
    t = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(C['bf16'])): lg = m(inp, fastmem.hippo_batch(inp).to(DEV))[0]
    loss = F.cross_entropy(lg.float().reshape(-1, 256), y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    if (it + 1) % 100 == 0: print(name, it + 1, 'loss', round(loss.item(), 4), flush=True)
R['bpb_after'] = round(bpb(), 5); ts = sorted(ts[20:]); R['ms_per_step'] = round(ts[len(ts) // 2] * 1000, 1); R['steps'] = STEPS
json.dump(R, open(os.path.join(RES, f'quality_ab_{name}.json'), 'w'), indent=1); print(json.dumps(R), flush=True)
