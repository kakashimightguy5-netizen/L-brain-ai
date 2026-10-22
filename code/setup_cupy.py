"""Installs CuPy into this Python environment (all on D:), then verifies and benchmarks the GPU scan kernel."""
import subprocess, sys, json, os, time
r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x'], capture_output=True, text=True); print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
import torch, l_model as M0, cuscan
R = {'cupy_available': cuscan.AVAILABLE, 'error': getattr(cuscan, '_err', None)}
if cuscan.AVAILABLE:
    a = (torch.rand(32, 256, 384, device='cuda') * 2 - 1).requires_grad_(); b = torch.randn(32, 256, 384, device='cuda').requires_grad_(); h0 = torch.zeros(32, 384, device='cuda')
    y, _ = cuscan.scan(a, b, h0, M0.LLayer.scan_hs); R['verified'] = cuscan.VERIFIED
    for name, f in [('log_depth_scan', lambda: M0.LLayer.scan_hs(a, b, h0)[0]), ('gpu_kernel', lambda: cuscan.scan(a, b, h0, M0.LLayer.scan_hs)[0])]:
        for _ in range(3): f().sum().backward()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20): f().sum().backward()
        torch.cuda.synchronize(); R[f'{name}_ms_fwd_bwd_32x256x384'] = round((time.time() - t) / 20 * 1e3, 2)
print(json.dumps(R)); json.dump(R, open(os.path.join(os.environ.get('LOUT', '.'), 'cupy_setup.json'), 'w'), indent=1)
