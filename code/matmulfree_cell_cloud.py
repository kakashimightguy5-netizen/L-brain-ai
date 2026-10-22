"""MatMul-Free Spiking Recurrent Cell (runnable PyTorch, not just pseudocode).  Every hardware claim below is labelled MODELLED unless a test measures it.

DESIGN RULE: in the forward pass every tensor that crosses a block boundary is a SPIKE (0/1), a ternary line (-1/0/+1) or a small integer, so inference needs only
ADD, SUBTRACT, COMPARE, SHIFT and table/register reads: no floating-point multiplier anywhere. Training uses float latents + straight-through / surrogate gradients (that part DOES use ordinary floating-point math, on the GPU/CPU, offline).

(A) TernaryLinear (BitLinear-style). Latent float weights are quantised to {-1,0,+1} (absmean rule, straight-through gradient). Because the INPUT is a binary spike vector, y = sum of the weight-columns of the ACTIVE inputs:
      y[o] = sum_{i : spike_i = 1} W[o,i],  W in {-1,0,+1}      -> accumulate only; weight 0 = no wire, spike 0 = no work (event-driven zero-skipping).
    Honest note: this removes MULTIPLIERS, not the matrix-vector STRUCTURE: it is still a (ternary, sparse) matrix-vector product with the same index pattern, and weights must still be fetched. Savings: no multiplier area/energy, ~1.58 bits per weight (MODELLED), skipped work for silent spikes and zero weights.
    The scale factor of BitLinear is dropped: it is absorbed into the integer thresholds, so no rescaling multiply is needed.
(B) Continuous state = diagonal linear (SSM-style) leaky membrane, in fixed point:  vi <- vi - floor(vi * 2^-k_c) + S*u.  The decay is a BIT-SHIFT with a per-channel shift k_c in 1..6 (multi-timescale memory),
    so the 'multiply by a' of an SSM becomes one shift + one subtract. O(d_hid) state, O(1) work per step regardless of how long the history is.
(C) Spikes: s = H(vi - theta*S) (a comparator). Training uses a fast-sigmoid surrogate gradient. Soft reset (vi -= S*theta*s) = a conditional subtract.
(D) Fast weights, INTEGER and local: F (d_v x d_k) holds small integers in [-FMAX, FMAX] (e.g. 5-bit saturating counters).
      read   r = F q          q = binary query spike vector  -> sum of the F columns selected by q (accumulate only)
      write  F += (v - p) k^T k = binary key spike vector, v = ternary target, p = ternary prediction thresholded from F k  -> (v-p) in {-2..2}: add/subtract a small integer into the columns where k fires.
    This is a perceptron/delta rule: pre-synaptic = key spike, post-synaptic error = target - prediction (local to the synapse; Hebbian with an error term). It is NOT timing-exact STDP (no spike-time window); STDP could be added with eligibility counters.
    Forgetting = overwrite by later errors + saturation. Memory per sequence is O(d_hid + d_v*d_k), independent of length T (no KV cache, no O(T^2) attention).
Training limits (honest): BPTT through the loop (and through F) needs memory ~ O(T*d_v*d_k); the fast-weight write is sequential in time (cannot be scanned like the diagonal slow state)."""
import os, torch, torch.nn as nn, torch.nn.functional as F_
KWTA = int(__import__('os').environ.get('KWTA', 0))
HOMEO = float(__import__('os').environ.get('HOMEO', 0))   # target firing rate per hidden neuron (0 = off)
TAPLIST = [int(v) for v in __import__('os').environ.get('TAPLIST', ','.join(str(j) for j in range(64))).split(',')]   # e.g. 0,1,2,3,5,8,12,16: log-spaced 'time cells' (scale-invariant temporal memory)

class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.save_for_backward(x); ctx.alpha = alpha; return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, g): (x,) = ctx.saved_tensors; return g / (1 + ctx.alpha * x.abs()) ** 2, None
heav = lambda x, alpha=2.0: SpikeFn.apply(x, alpha)                          # comparator in hardware
tern = lambda x, th: heav(x - th) - heav(-x - th)                            # two comparators -> ternary line {-1,0,+1}
def ste_floor(x): return x + (torch.floor(x) - x).detach()                    # shift/truncate in hardware; identity gradient

class TernaryLinear(nn.Module):
    def __init__(self, d_in, d_out, scale=1.0):
        super().__init__(); self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.7); self.scale = scale   # scale = static 1/sqrt(fan-in) constant: folds into the integer thresholds on hardware (no multiplier)
    def wq(self):
        if os.environ.get('TERN', '1') == '0': return self.weight / (self.weight.abs().mean() + 1e-8)   # ABLATION flag (default off): float weights, to separate the cost of ternarisation from the cost of spiking
        # float latent -> {-1,0,+1}, straight-through
        g = self.weight.abs().mean() + 1e-8; w = self.weight / g; return w + (torch.round(torch.clamp(w, -1, 1)) - w).detach()
    def forward(self, x): return F_.linear(x, self.wq()) * self.scale                      # TRAINING/emulation path: values identical to the accumulate-only path below
    @torch.no_grad()
    def event_forward(self, x_bin):                                           # INFERENCE path: no multiplications, only gathers + adds of ternary entries
        W = self.wq().round(); y = torch.zeros(x_bin.size(0), W.size(0))
        for b in range(x_bin.size(0)): y[b] = W[:, x_bin[b].nonzero(as_tuple=True)[0]].sum(1)
        return y * self.scale

class MatMulFreeSpikingCell(nn.Module):
    S = 64.0                                                                  # fixed-point scale of the membrane (6 fractional bits)
    def __init__(self, d_in, d_hid=64, d_out=None, d_k=24, d_v=24, fmax=15, norm=False, taps=1):
        super().__init__(); self.norm = norm; self.taps = taps;   # TAPS flag (default 1 = off): causal shift register of the last `taps` input frames (flip-flops on FPGA) feeding the input layer
        d_out = d_out or d_hid; self.d_hid, self.d_k, self.d_v, self.d_out, self.fmax = d_hid, d_k, d_v, d_out, fmax
        sc = (lambda fan: (fan * 0.3 * 0.67) ** -0.5) if norm else (lambda fan: 1.0)   # NORM flag (default off): variance-preserving static scale for ~30% spike rate, ~67% non-zero ternary weights
        self.inp = TernaryLinear(d_in * taps, d_hid, sc(d_in * taps)); self.key = TernaryLinear(d_hid, d_k, sc(d_hid)); self.qry = TernaryLinear(d_hid, d_k, sc(d_hid)); self.val = TernaryLinear(d_hid, d_v, sc(d_hid))
        self.out = TernaryLinear(d_hid + 2 * d_v, d_out, sc(d_hid + 2 * d_v))
        self.register_buffer('shift', torch.linspace(1, 6, d_hid).round())
        self.register_buffer('hbias', torch.zeros(d_hid))   # HOMEO flag: intrinsic-plasticity threshold offset (not trained by gradient)    # decay = 1 - 2^-shift, a bit-shift per channel
        f = (lambda a, b: b) if norm else (lambda a, b: a)   # (unnormalised init, normalised init)
        self.theta = nn.Parameter(torch.full((d_hid,), f(1.5, 1.0))); self.th_k = nn.Parameter(torch.full((d_k,), f(0.5, 0.3))); self.th_q = nn.Parameter(torch.full((d_k,), f(0.5, 0.3)))
        self.th_v = nn.Parameter(torch.full((d_v,), f(0.5, 0.3))); self.th_p = nn.Parameter(torch.full((d_v,), f(1.0, 0.5))); self.th_r = nn.Parameter(torch.full((d_v,), f(1.0, 0.5))); self.th_o = nn.Parameter(torch.full((d_out,), f(0.5, 0.0)))
    def init_state(self, B, dev=None): return torch.zeros(B, self.d_hid, device=dev), torch.zeros(B, self.d_v, self.d_k, device=dev)
    def forward(self, x, state=None, fast=True):
        """x: [B,T,d_in] BINARY spikes -> (spikes [B,T,d_out], pre-threshold [B,T,d_out], new state, stats)."""
        if self.taps > 1: x = torch.cat([F_.pad(x, (0, 0, j, 0))[:, :x.size(1)] for j in TAPLIST[:self.taps]], -1)   # TAPLIST env: which past frames are tapped (default 0..taps-1, contiguous)   # NOTE: window restarts at the chunk start; streaming with taps>1 needs the last taps-1 frames carried in the state
        B, T, _ = x.shape; vi, Fw = state if state is not None else self.init_state(B, x.device); S = self.S; dec = 2.0 ** (-self.shift); outs, pres = [], []; act = 0.0; nsp = 0.0
        hb = self.hbias.clone(); hcount = torch.zeros_like(hb)
        for t in range(T):
            vi = vi - ste_floor(vi * dec) + S * self.inp(x[:, t])               # (B) shift-decay + ternary accumulate of input spikes
            s = heav((vi - S * (self.theta + hb)) / S); hcount = hcount + s.detach().mean(0)
            if KWTA > 0 and KWTA < s.size(-1):   # KWTA flag (default 0 = off): lateral inhibition, only the k most-depolarised neurons may fire this step (competition)
                mg = (vi - S * (self.theta + hb)).detach(); kth = mg.topk(KWTA, -1).values[..., -1:]; s = s - (s * (mg < kth).to(s.dtype)).detach() if __import__('os').environ.get('KWTA_STE') == '1' else s * (mg >= kth).to(s.dtype)   # KWTA_STE: losers silenced in forward, still receive gradient (avoids dead losers)
            vi = vi - (torch.round(S * self.theta.detach()).clamp(min=1) if self.norm else S * self.theta.detach().round().clamp(min=1)) * s   # (C) comparator + conditional subtract (reset)
            if fast:
                k = heav(self.key(s) - self.th_k); q = heav(self.qry(s) - self.th_q); v = tern(self.val(s), self.th_v)          # binary key/query lines, ternary target
                rr = torch.bmm(Fw, q.unsqueeze(-1)).squeeze(-1)                  # read: sum of F columns selected by q
                p = tern(torch.bmm(Fw, k.unsqueeze(-1)).squeeze(-1), self.th_p); err = v - p                                       # prediction & error, integers in {-2..2}
                Fw = torch.clamp(Fw + torch.bmm(err.unsqueeze(-1), k.unsqueeze(1)), -self.fmax, self.fmax)                        # (D) local write: add/subtract err into columns where k fires
                r = torch.cat([heav(rr - self.th_r), heav(-rr - self.th_r)], -1)  # signed readout as two binary lines
            else: r = torch.zeros(B, 2 * self.d_v, device=x.device)
            z = torch.cat([s, r], -1); pre = self.out(z) - self.th_o; o = heav(pre); outs.append(o); pres.append(pre)
            nsp = nsp + (s.detach().sum() + r.detach().sum()) / T                # spikes into ternary layers (events that cause accumulates)
        stats = {'hid_rate': float(nsp.item() / (B * (self.d_hid + 2 * self.d_v))), 'F_int': bool(torch.equal(Fw, Fw.round())), 'F_absmax': float(Fw.abs().max().item())}
        if HOMEO > 0 and self.training:   # intrinsic plasticity: neurons firing above the target rate raise their threshold, silent ones lower it
            with torch.no_grad(): self.hbias += float(__import__('os').environ.get('HOMEO_ETA', 0.05)) * (hcount / T - HOMEO)
        return torch.stack(outs, 1), torch.stack(pres, 1), (vi, Fw), stats

def modelled_ops(cell, rate, sparsity_w=None):
    """MODELLED per-step operation counts. Dense float MACs vs event-driven adds: adds = (active input spikes) x d_out x (nonzero-weight fraction)."""
    L = [(cell.inp, None), (cell.key, None), (cell.qry, None), (cell.val, None), (cell.out, None)]
    dense = sum(l.weight.numel() for l, _ in L); nz = sum((l.wq() != 0).float().mean().item() * l.weight.numel() for l, _ in L)
    return {'dense_float_MACs_per_step': dense, 'event_adds_per_step_at_rate': round(nz * rate), 'nonzero_weight_frac': round(nz / dense, 3)}

if __name__ == '__main__':
    torch.manual_seed(0); cell = MatMulFreeSpikingCell(16, 64, 16); x = (torch.rand(4, 40, 16) > 0.7).float()
    y, pre, (vi, Fw), st = cell(x); assert y.shape == (4, 40, 16) and set(y.unique().tolist()) <= {0.0, 1.0}; print('binary spike outputs ok; stats', st)
    assert torch.equal(Fw, Fw.round()) and (cell.norm or torch.equal(vi, vi.round())); print('fast weights (and, with norm off, the fixed-point membrane) are exact integers: ok')
    lin = cell.key; sb = (torch.rand(5, 64) > 0.8).float()
    assert torch.equal(lin(sb), lin.event_forward(sb)); print('accumulate-only (no multiply) path == dense emulation for ternary layer on spikes: ok')
    pre.sum().backward(); n_ok = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in cell.parameters()); print('params with nonzero grad:', int(n_ok), 'of', len(list(cell.parameters())))
    with torch.no_grad():
        y_full, _, s_full, _ = cell(x); y1, _, s1, _ = cell(x[:, :15]); y2, _, s2, _ = cell(x[:, 15:], s1); assert torch.equal(torch.cat([y1, y2], 1), y_full); print('chunked streaming == one-shot: ok')
        _, _, (v2, F2), _ = cell(torch.rand(4, 400, 16).gt(0.7).float()); assert F2.shape == Fw.shape; print('state size independent of T: ok', tuple(F2.shape))
    print('MODELLED ops', modelled_ops(cell, st['hid_rate']))
    import sys
    if 'notrain' in sys.argv: raise SystemExit
    # micro-training: delayed copy (output at t = input at t-3); with fast weights on vs off. MEASURED, tiny, CPU.
    def run(fast, steps=500, seed=0):
        torch.manual_seed(seed); c = MatMulFreeSpikingCell(8, 64, 8); opt = torch.optim.Adam(c.parameters(), 1e-2)
        for it in range(steps):
            xb = (torch.rand(64, 24, 8) > 0.5).float(); tgt = torch.cat([torch.zeros(64, 3, 8), xb[:, :-3]], 1)
            _, pre, _, _ = c(xb, fast=fast); loss = F_.binary_cross_entropy_with_logits(pre[:, 3:], tgt[:, 3:]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            xb = (torch.rand(256, 24, 8) > 0.5).float(); tgt = torch.cat([torch.zeros(256, 3, 8), xb[:, :-3]], 1); o, _, _, _ = c(xb, fast=fast); return round((o[:, 3:] == tgt[:, 3:]).float().mean().item(), 4), round(loss.item(), 4)
    for fast in (True, False): print('delayed-copy bit accuracy (chance ~0.5), fast weights', 'ON ' if fast else 'OFF', run(fast))
