"""Energy fix F1 feasibility check (GPU job): can torch.compile (Triton) fuse L's elementwise chain on this laptop?
Compares eager vs compiled for RMSNorm + spike-count chain (fwd+bwd): numerical error, CUDA kernel launches, ms/step. Writes ../results/compile_check.json. No model weights, no power settings changed."""
import os, sys, json, time, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); R = {'torch': torch.__version__, 'cuda': torch.cuda.is_available()}
try:
    import triton; R['triton'] = triton.__version__
except Exception as e: R['triton'] = 'missing: ' + repr(e)[:120]
def chain(x, w, th, K: int = 8):
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    t = th.abs().clamp(min=0.05); s = torch.clamp(torch.round(n / t), -K, K) * t   # forward value of spike count (no STE here: timing/kernel-count test only)
    g = torch.relu(s) / t
    return (s + g * t).float()
def run(fn, x, w, th, n=50):
    for _ in range(5): y = fn(x, w, th); y.sum().backward()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): y = fn(x, w, th); y.sum().backward()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1000
def kernels(fn, x, w, th):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        y = fn(x, w, th); y.sum().backward(); torch.cuda.synchronize()
    return sum(1 for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA)
if torch.cuda.is_available():
    dev = 'cuda'; x = torch.randn(16, 256, 128, device=dev, requires_grad=True); w = torch.ones(128, device=dev, requires_grad=True); th = torch.full((128,), 0.5, device=dev, requires_grad=True)
    R['eager_ms'] = run(chain, x, w, th); R['eager_kernels'] = kernels(chain, x, w, th)
    try:
        cc = torch.compile(chain); y0 = chain(x, w, th); y1 = cc(x, w, th); R['max_abs_err'] = (y0 - y1).abs().max().item()
        R['compiled_ms'] = run(cc, x, w, th); R['compiled_kernels'] = kernels(cc, x, w, th); R['compile_ok'] = True
    except Exception as e: R['compile_ok'] = False; R['compile_error'] = repr(e)[:400]
os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
json.dump(R, open(os.path.join(HERE, '..', 'results', 'compile_check.json'), 'w'), indent=1); print(json.dumps(R), flush=True)
