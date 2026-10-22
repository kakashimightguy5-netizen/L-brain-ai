"""Delta-memory speed test on GPU (test, not benchmark): chunk size 16/32/64 give identical math; which is fastest?"""
import os, time, json, torch, torch.nn.functional as F
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
import l5_model as M5
dev = 'cuda'; torch.manual_seed(0); R = {}
x = torch.randn(32, 256, 352, device=dev, requires_grad=True)
ref = None
for C in (64, 32, 16):
    torch.manual_seed(0); m = M5.DeltaMemory(352, 8, C=C).to(dev)
    for bf in (False, True):
        def f():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf): y = m(x)
            y.float().pow(2).mean().backward(); return y
        for _ in range(3): y = f()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10): y = f()
        torch.cuda.synchronize(); ms = (time.time() - t) / 10 * 1000
        if ref is None: ref = y.detach().float()
        R[f'C={C} bf16={bf}'] = {'ms_fwd_bwd': round(ms, 1), 'max_diff_vs_C64_fp32': round((y.detach().float() - ref).abs().max().item(), 6)}
        print(R, flush=True)
json.dump(R, open(os.path.join(OUT, 'delta_speed.json'), 'w'), indent=1)
