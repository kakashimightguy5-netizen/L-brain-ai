"""Fused GPU kernel for L's error-correcting (delta-rule) fast-weight memory.
  S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T ,   o_t = S_t q_t
One CUDA block per (batch, head); thread i owns ROW i of the dk x dk memory matrix and keeps it in registers, so the
whole recurrence runs on-chip with no tiny-matrix library calls. Backward = exact reverse-mode of the same recurrence
(states re-computed per 64-token chunk from checkpoints, so memory stays small).
Compiled at runtime by CuPy/NVRTC (no Visual Studio). Self-verified against the PyTorch chunk form on first use;
any mismatch -> switches itself off and L uses the PyTorch path."""
import torch
_SRC = r'''
#define MAXDK 64
extern "C" __global__ void dfwd(const float* q, const float* k, const float* v, const float* a, const float* b,
                                float* o, float* chk, int T, int dk, int C){
  int bh = blockIdx.x, i = threadIdx.x; int NC = (T + C - 1) / C;
  __shared__ float sk[MAXDK], sq[MAXDK];
  float S[MAXDK]; for (int j = 0; j < dk; j++) S[j] = 0.f;
  for (int t = 0; t < T; t++){
    if (t % C == 0){ float* c = chk + (((long)bh * NC + t / C) * dk + i) * dk; for (int j = 0; j < dk; j++) c[j] = S[j]; }
    long base = ((long)bh * T + t) * dk;
    sk[i] = k[base + i]; sq[i] = q[base + i]; __syncthreads();
    float at = a[(long)bh * T + t], bt = b[(long)bh * T + t];
    float skd = 0.f; for (int j = 0; j < dk; j++) skd += S[j] * sk[j];
    float u = bt * (v[base + i] - at * skd); float od = 0.f;
    for (int j = 0; j < dk; j++){ S[j] = at * S[j] + u * sk[j]; od += S[j] * sq[j]; }
    o[base + i] = od; __syncthreads();
  }
}
extern "C" __global__ void dbwd(const float* q, const float* k, const float* v, const float* a, const float* b,
                                const float* go, const float* chk, float* scr, float* gq, float* gk, float* gv,
                                float* ga, float* gb, int T, int dk, int C){
  int bh = blockIdx.x, i = threadIdx.x; int NC = (T + C - 1) / C;
  __shared__ float sk[MAXDK], sq[MAXDK], sg[MAXDK], aq[MAXDK], ak[MAXDK], red[2];
  float G[MAXDK]; for (int j = 0; j < dk; j++) G[j] = 0.f;
  float* myscr = scr + (long)bh * (C + 1) * dk * dk;
  for (int c = NC - 1; c >= 0; c--){
    int t0 = c * C, n = min(T, t0 + C) - t0;
    float S[MAXDK]; const float* cp = chk + (((long)bh * NC + c) * dk + i) * dk;
    for (int j = 0; j < dk; j++){ S[j] = cp[j]; myscr[(0 * dk + i) * dk + j] = S[j]; }
    for (int s = 0; s < n; s++){                      // recompute the states inside this chunk
      int t = t0 + s; long base = ((long)bh * T + t) * dk;
      sk[i] = k[base + i]; __syncthreads();
      float at = a[(long)bh * T + t], bt = b[(long)bh * T + t];
      float skd = 0.f; for (int j = 0; j < dk; j++) skd += S[j] * sk[j];
      float u = bt * (v[base + i] - at * skd);
      for (int j = 0; j < dk; j++){ S[j] = at * S[j] + u * sk[j]; myscr[((long)(s + 1) * dk + i) * dk + j] = S[j]; }
      __syncthreads();
    }
    for (int s = n - 1; s >= 0; s--){                 // reverse-mode through the chunk
      int t = t0 + s; long base = ((long)bh * T + t) * dk;
      sk[i] = k[base + i]; sq[i] = q[base + i]; sg[i] = go[base + i]; aq[i] = 0.f; ak[i] = 0.f;
      if (i == 0){ red[0] = 0.f; red[1] = 0.f; }
      __syncthreads();
      float at = a[(long)bh * T + t], bt = b[(long)bh * T + t], vi = v[base + i], gi = sg[i];
      const float* St = myscr + ((long)(s + 1) * dk + i) * dk; const float* Sp = myscr + ((long)s * dk + i) * dk;
      for (int j = 0; j < dk; j++) G[j] += gi * sq[j];
      float Gk = 0.f, Spk = 0.f, GS = 0.f;
      for (int j = 0; j < dk; j++){ Gk += G[j] * sk[j]; Spk += Sp[j] * sk[j]; GS += G[j] * Sp[j]; }
      for (int j = 0; j < dk; j++){
        atomicAdd(&aq[j], St[j] * gi);
        atomicAdd(&ak[j], bt * G[j] * vi - at * bt * (Sp[j] * Gk + G[j] * Spk));
      }
      gv[base + i] = bt * Gk;
      atomicAdd(&red[0], Gk * (vi - at * Spk)); atomicAdd(&red[1], GS - bt * Gk * Spk);
      __syncthreads();
      gq[base + i] = aq[i]; gk[base + i] = ak[i];
      if (i == 0){ gb[(long)bh * T + t] = red[0]; ga[(long)bh * T + t] = red[1]; }
      for (int j = 0; j < dk; j++) G[j] = at * (G[j] - bt * Gk * sk[j]);
      __syncthreads();
    }
  }
}'''
try:
    import cupy as cp
    _mod = cp.RawModule(code=_SRC); _f, _b = _mod.get_function('dfwd'), _mod.get_function('dbwd'); AVAILABLE = True
except Exception as e:
    AVAILABLE = False; _err = repr(e)
C = 64
def _cp(t): return cp.from_dlpack(t.detach())
def _launch(fn, BH, dk, args):
    dev = args[0].device.index
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(dev).cuda_stream):
        fn((BH,), (dk,), tuple(_cp(x) if torch.is_tensor(x) else x for x in args))
class _Delta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, a, b):
        B, H, T, dk = q.shape; BH = B * H; NC = (T + C - 1) // C
        q, k, v, a, b = [x.contiguous().float() for x in (q, k, v, a, b)]
        o = torch.empty_like(q); chk = torch.empty(BH, NC, dk, dk, device=q.device)
        _launch(_f, BH, dk, [q, k, v, a, b, o, chk, cp.int32(T), cp.int32(dk), cp.int32(C)])
        ctx.save_for_backward(q, k, v, a, b, chk); return o
    @staticmethod
    def backward(ctx, go):
        q, k, v, a, b, chk = ctx.saved_tensors; B, H, T, dk = q.shape; BH = B * H
        go = go.contiguous().float(); scr = torch.empty(BH, C + 1, dk, dk, device=q.device)
        gq, gk, gv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v); ga, gb = torch.empty_like(a), torch.empty_like(b)
        _launch(_b, BH, dk, [q, k, v, a, b, go, chk, scr, gq, gk, gv, ga, gb, cp.int32(T), cp.int32(dk), cp.int32(C)])
        return gq, gk, gv, ga, gb
VERIFIED = None
def delta(q, k, v, la, b, reference):
    """q,k,v (B,H,T,dk); la log-decay (B,H,T); b write strength (B,H,T). Returns o (B,H,T,dk) or falls back to reference."""
    global VERIFIED
    if VERIFIED is None:
        try:
            g = torch.Generator(device=q.device).manual_seed(0); B, H, T, dk = 2, 3, 150, min(q.size(-1), 64)
            mk = lambda *s: torch.randn(*s, device=q.device, generator=g)
            tq, tv = mk(B, H, T, dk).requires_grad_(), mk(B, H, T, dk).requires_grad_()
            tk = torch.nn.functional.normalize(mk(B, H, T, dk), dim=-1).requires_grad_()
            tla = torch.nn.functional.logsigmoid(mk(B, H, T) + 3).requires_grad_(); tb = torch.sigmoid(mk(B, H, T)).requires_grad_()
            w = mk(B, H, T, dk); ins = [tq, tk, tv, tla, tb]
            y1 = _Delta.apply(tq, tk, tv, tla.exp(), tb); g1 = torch.autograd.grad((y1 * w).sum(), ins)
            y2 = reference(tq, tk, tv, tla, tb, torch.zeros(B, H, dk, dk, device=q.device))[0]; g2 = torch.autograd.grad((y2 * w).sum(), ins)
            err = max([(y1 - y2).abs().max().item()] + [(x - y).abs().max().item() / (y.abs().max().item() + 1e-6) for x, y in zip(g1, g2)])
            VERIFIED = err < 2e-3; print(f'deltakern self-test max (relative) error {err:.2e} -> {"ON" if VERIFIED else "OFF"}', flush=True)
        except Exception as e:
            VERIFIED = False; print('deltakern self-test failed -> OFF:', repr(e)[:300], flush=True)
    if not VERIFIED or q.size(-1) > 64: return reference(q, k, v, la, b, q.new_zeros(q.size(0), q.size(1), q.size(-1), q.size(-1)))[0]
    return _Delta.apply(q, k, v, la.exp(), b)

def emulate(q, k, v, a, b, go):
    """CPU emulation of the exact kernel algorithm (rows vectorised), to verify the backward math without a GPU."""
    BH, T, dk = q.shape; NC = (T + C - 1) // C; S = q.new_zeros(BH, dk, dk); chk = []; o = torch.empty_like(q)
    for t in range(T):
        if t % C == 0: chk.append(S.clone())
        at, bt = a[:, t, None, None], b[:, t, None]
        u = bt * (v[:, t] - a[:, t, None] * (S @ k[:, t, :, None])[..., 0])
        S = at * S + u[:, :, None] * k[:, t, None, :]; o[:, t] = (S @ q[:, t, :, None])[..., 0]
    G = q.new_zeros(BH, dk, dk); gq, gk, gv = [torch.empty_like(q) for _ in range(3)]; ga, gb = torch.empty_like(a), torch.empty_like(b)
    for c in range(NC - 1, -1, -1):
        t0 = c * C; n = min(T, t0 + C) - t0; S = chk[c].clone(); st = [S.clone()]
        for s in range(n):
            t = t0 + s; at = a[:, t, None, None]; u = b[:, t, None] * (v[:, t] - a[:, t, None] * (S @ k[:, t, :, None])[..., 0])
            S = at * S + u[:, :, None] * k[:, t, None, :]; st.append(S.clone())
        for s in range(n - 1, -1, -1):
            t = t0 + s; at, bt = a[:, t, None], b[:, t, None]; St, Sp = st[s + 1], st[s]; kt, qt, gt, vt = k[:, t], q[:, t], go[:, t], v[:, t]
            G = G + gt[:, :, None] * qt[:, None, :]
            Gk = (G @ kt[:, :, None])[..., 0]; Spk = (Sp @ kt[:, :, None])[..., 0]; GS = (G * Sp).sum((1, 2))
            gq[:, t] = (St * gt[:, :, None]).sum(1)
            gk[:, t] = (bt[:, :, None] * G * vt[:, :, None] - (at * bt)[:, :, None] * (Sp * Gk[:, :, None] + G * Spk[:, :, None])).sum(1)
            gv[:, t] = bt * Gk; gb[:, t] = (Gk * (vt - at * Spk)).sum(1); ga[:, t] = GS - bt[:, 0] * (Gk * Spk).sum(1)
            G = at[:, :, None] * (G - bt[:, :, None] * Gk[:, :, None] * kt[:, None, :])
    return o, (gq, gk, gv, ga, gb)

if __name__ == '__main__':
    import torch.nn.functional as F, l5_model as M5
    torch.manual_seed(0); B, H, T, dk = 2, 3, 150, 7
    q, v = torch.randn(B, H, T, dk, dtype=torch.float64), torch.randn(B, H, T, dk, dtype=torch.float64)
    k = F.normalize(torch.randn(B, H, T, dk, dtype=torch.float64), dim=-1)
    a = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64) + 3); b = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    go = torch.randn(B, H, T, dk, dtype=torch.float64)
    ins = [x.clone().requires_grad_() for x in (q, k, v, a, b)]
    ref = M5.delta_loop(ins[0], ins[1], ins[2], ins[3].log(), ins[4], torch.zeros(B, H, dk, dk, dtype=torch.float64))[0]
    gref = torch.autograd.grad((ref * go).sum(), ins)
    fl = lambda x: x.reshape(B * H, *x.shape[2:])
    o, g = emulate(fl(q), fl(k), fl(v), fl(a), fl(b), fl(go))
    print('forward max err', (o - fl(ref)).abs().max().item())
    for nm, x, y in zip(['q', 'k', 'v', 'a', 'b'], g, gref): print('grad', nm, 'max err', (x - fl(y)).abs().max().item())
