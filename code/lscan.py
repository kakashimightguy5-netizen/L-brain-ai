"""Compiled (C++/OpenMP) scan for L's membrane update h_t = a_t*h_{t-1} + b_t, with exact hand-written backward."""
import os, torch
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext'), exist_ok=True)
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> fwd(torch::Tensor a, torch::Tensor b, torch::Tensor h0) {
  a = a.contiguous(); b = b.contiguous(); h0 = h0.contiguous();
  auto B = a.size(0), T = a.size(1), D = a.size(2);
  auto h = torch::empty_like(b);
  float *pa = a.data_ptr<float>(), *pb = b.data_ptr<float>(), *ph = h.data_ptr<float>(), *p0 = h0.data_ptr<float>();
  #pragma omp parallel for collapse(2)
  for (long i = 0; i < B; i++) for (long d = 0; d < D; d++) {
    float s = p0[i*D+d];
    for (long t = 0; t < T; t++) { long k = (i*T+t)*D+d; s = pa[k]*s + pb[k]; ph[k] = s; }
  }
  return {h};
}
std::vector<torch::Tensor> bwd(torch::Tensor a, torch::Tensor h, torch::Tensor h0, torch::Tensor g) {
  a = a.contiguous(); h = h.contiguous(); g = g.contiguous(); h0 = h0.contiguous();
  auto B = a.size(0), T = a.size(1), D = a.size(2);
  auto ga = torch::empty_like(a), gb = torch::empty_like(a), g0 = torch::empty_like(h0);
  float *pa = a.data_ptr<float>(), *ph = h.data_ptr<float>(), *pg = g.data_ptr<float>(), *p0 = h0.data_ptr<float>();
  float *pga = ga.data_ptr<float>(), *pgb = gb.data_ptr<float>(), *pg0 = g0.data_ptr<float>();
  #pragma omp parallel for collapse(2)
  for (long i = 0; i < B; i++) for (long d = 0; d < D; d++) {
    float G = 0.f;
    for (long t = T-1; t >= 0; t--) {
      long k = (i*T+t)*D+d;
      G = pg[k] + G;                      // total grad wrt h_t
      float hp = (t > 0) ? ph[k-D] : p0[i*D+d];
      pga[k] = G * hp; pgb[k] = G;
      G = G * pa[k];                      // pass to h_{t-1}
    }
    pg0[i*D+d] = G;
  }
  return {ga, gb, g0};
}
'''
cpp = 'std::vector<torch::Tensor> fwd(torch::Tensor a, torch::Tensor b, torch::Tensor h0);\nstd::vector<torch::Tensor> bwd(torch::Tensor a, torch::Tensor h, torch::Tensor h0, torch::Tensor g);'
try:
  if os.environ.get('LSCAN_DISABLE') == '1': raise RuntimeError('disabled')
  _ext = load_inline('lscan_ext', cpp_sources=[cpp + '\n' + src.replace('#include <torch/extension.h>', '')] if False else [src], functions=['fwd', 'bwd'],
                   extra_cflags=['-O3', '-fopenmp'], extra_ldflags=['-fopenmp'], build_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext'), verbose=False); AVAILABLE = True
except Exception as e:
  _ext = None; AVAILABLE = False; print('lscan: compiled scan unavailable, using torch chunked scan:', type(e).__name__)
class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, h0):
        h, = _ext.fwd(a, b, h0); ctx.save_for_backward(a, h, h0); return h
    @staticmethod
    def backward(ctx, g):
        a, h, h0 = ctx.saved_tensors; ga, gb, g0 = _ext.bwd(a, h, h0, g.contiguous()); return ga, gb, g0
def scan(a, b, h0): return Scan.apply(a.float(), b.float(), h0.float())
