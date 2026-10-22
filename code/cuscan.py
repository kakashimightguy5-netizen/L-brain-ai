"""GPU kernel for L's membrane scan h_t = a_t*h_{t-1} + b_t (one thread per neuron, exact sequential recurrence).
Compiled at runtime by CuPy/NVRTC (no Visual Studio needed). Verified against the reference scan on first use;
if anything is off, it switches itself off and L falls back to the log-depth scan."""
import torch
try:
    import cupy as cp
    _src = r'''
extern "C" __global__ void fwd(const float* a, const float* b, const float* h0, float* h, long B, long T, long D){
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i >= B*D) return;
  long n = i / D, d = i % D; float s = h0[i];
  for (long t = 0; t < T; t++){ long k = (n*T + t)*D + d; s = a[k]*s + b[k]; h[k] = s; }
}
extern "C" __global__ void bwd(const float* a, const float* h, const float* h0, const float* g, float* ga, float* gb, float* g0, long B, long T, long D){
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i >= B*D) return;
  long n = i / D, d = i % D; float G = 0.f;
  for (long t = T-1; t >= 0; t--){ long k = (n*T + t)*D + d; G += g[k]; float hp = (t > 0) ? h[k-D] : h0[i];
    ga[k] = G*hp; gb[k] = G; G *= a[k]; }
  g0[i] = G;
}'''
    _mod = cp.RawModule(code=_src); _f, _b = _mod.get_function('fwd'), _mod.get_function('bwd'); AVAILABLE = True
except Exception as e:
    AVAILABLE = False; _err = repr(e)
def _cp(t): return cp.from_dlpack(t.detach())
def _launch(fn, n, args):
    # launch on PyTorch's current CUDA stream -> correctly ordered with torch ops, no device-wide synchronisation needed
    dev = args[0].device.index
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(dev).cuda_stream):
        fn(((n + 255) // 256,), (256,), tuple(_cp(x) if torch.is_tensor(x) else x for x in args))
class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, h0):
        a, b, h0 = a.contiguous().float(), b.contiguous().float(), h0.contiguous().float(); B, T, D = b.shape
        h = torch.empty_like(b); _launch(_f, B * D, [a, b, h0, h, cp.int64(B), cp.int64(T), cp.int64(D)])
        ctx.save_for_backward(a, h, h0); return h
    @staticmethod
    def backward(ctx, g):
        a, h, h0 = ctx.saved_tensors; B, T, D = a.shape; g = g.contiguous().float()
        ga, gb, g0 = torch.empty_like(a), torch.empty_like(a), torch.empty_like(h0)
        _launch(_b, B * D, [a, h, h0, g, ga, gb, g0, cp.int64(B), cp.int64(T), cp.int64(D)])
        return ga, gb, g0
VERIFIED = None
def scan(a, b, h0, reference):
    global VERIFIED
    if VERIFIED is None:
        try:
            with torch.enable_grad():   # FIX: self-test must run with autograd on even when the caller is inside torch.no_grad() (teacher pass); it used to fail and silently switch the GPU kernel OFF
                ta = (torch.rand(3, 77, 19, device=b.device) * 2 - 1).requires_grad_(); tb = torch.randn(3, 77, 19, device=b.device).requires_grad_(); t0 = torch.randn(3, 19, device=b.device)
                y = _Scan.apply(ta, tb, t0); w = torch.randn_like(y); (y * w).sum().backward(); g1 = (ta.grad.clone(), tb.grad.clone()); ta.grad = None; tb.grad = None
                y2 = reference(ta, tb, t0)[0]; (y2 * w).sum().backward()
                err = max((y - y2).abs().max().item(), (g1[0] - ta.grad).abs().max().item(), (g1[1] - tb.grad).abs().max().item())
                VERIFIED = err < 1e-3; print(f'cuscan self-test max error {err:.2e} -> {"ON" if VERIFIED else "OFF"}', flush=True)
        except Exception as e:
            VERIFIED = False; print('cuscan self-test failed -> OFF:', repr(e)[:200], flush=True)
    if not VERIFIED: return reference(a, b, h0)
    h = _Scan.apply(a, b, h0); return h, h[:, -1]
