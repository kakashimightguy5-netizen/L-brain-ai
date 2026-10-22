"""Spiking Linear Recurrent Cell: runnable PyTorch (not just pseudocode).
A) continuous diagonal linear state (SSM-style leaky membrane): v_t = a*v_{t-1} + W_in x_t, a in (0,1) per channel, so the slow state is O(d) per step.
B) binary spikes s_t = H(v_t - theta) with a surrogate gradient (fast-sigmoid), soft reset v -= s*theta.
C) fast weights: a per-sequence matrix F (d_v x d_k) written DURING the forward pass by a local delta/Hebbian rule from binary spike keys:
       F_t = lam * F_{t-1} + eta * (v_t - F_{t-1} k_t) k_t^T / (k_t.k_t + 1)       (normalised delta rule; k_t is a binary spike vector)
   and read by a binary spike query: r_t = F_{t-1} q_t. Pre-synaptic = spike key, post-synaptic target = value: a local, Hebbian-with-error rule.
Memory per sequence is O(d_hid + d_v*d_k), independent of sequence length T (no KV cache, no O(T^2) attention).
Honest limits: (1) this loop is sequential in time (the diagonal slow state could use a parallel scan, the fast-weight write cannot be scanned as simply);
(2) the write is delta-rule 'STDP-like' on rates/spikes in a step, not timing-exact STDP; (3) BPTT through F costs memory ~ O(T*d_v*d_k) in training unless truncated."""
import math, torch, torch.nn as nn

class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x); ctx.alpha = alpha; return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors; return g / (1 + ctx.alpha * x.abs()) ** 2, None   # fast-sigmoid surrogate
spike = lambda x, alpha=10.0: SpikeFn.apply(x, alpha)

class SpikingLinearRecurrentCell(nn.Module):
    def __init__(self, d_in, d_hid=128, d_k=32, d_v=32, lam=0.99, eta=0.5):
        super().__init__()
        self.d_hid, self.d_k, self.d_v = d_hid, d_k, d_v
        self.inp = nn.Linear(d_in, d_hid)
        self.decay_logit = nn.Parameter(torch.linspace(0.0, 4.0, d_hid))        # a = sigmoid in (0.5, 0.98): multi-timescale memory
        self.theta = nn.Parameter(torch.ones(d_hid))                             # firing threshold per channel
        self.key, self.qry, self.val = nn.Linear(d_hid, d_k), nn.Linear(d_hid, d_k), nn.Linear(d_hid, d_v)
        self.out = nn.Linear(d_hid + d_v, d_hid); self.theta_out = nn.Parameter(torch.zeros(d_hid))
        self.lam, self.eta = lam, eta
    def init_state(self, B, device=None):
        z = lambda *s: torch.zeros(*s, device=device)
        return z(B, self.d_hid), z(B, self.d_v, self.d_k)                        # (membrane v, fast-weight matrix F)
    def forward(self, x, state=None, fast=True):
        """x: [B,T,d_in] -> spikes [B,T,d_hid], new state, stats. fast=False disables the fast-weight path (equivalence/ablation flag)."""
        B, T, _ = x.shape; v, F_ = state if state is not None else self.init_state(B, x.device); a = torch.sigmoid(self.decay_logit); u = self.inp(x); outs = []; rate = 0.0; wnorm = 0.0
        for t in range(T):
            v = a * v + u[:, t]                                                  # A) linear state update
            s = spike(v - self.theta); v = v - s * self.theta                    # B) binary spikes + soft reset
            if fast:
                k = spike(self.key(s)); q = spike(self.qry(s)); val = self.val(s)  # binary key/query spikes, analog value
                r = torch.bmm(F_, q.unsqueeze(-1)).squeeze(-1)                   # read old memory (causal)
                err = val - torch.bmm(F_, k.unsqueeze(-1)).squeeze(-1)           # prediction error of what the memory would return for this key
                F_ = self.lam * F_ + self.eta * torch.bmm(err.unsqueeze(-1), k.unsqueeze(1)) / (k.pow(2).sum(-1, keepdim=True).unsqueeze(-1) + 1.0)   # C) local write
            else: r = torch.zeros(B, self.d_v, device=x.device)
            o = spike(self.out(torch.cat([s, r], -1)) - self.theta_out); outs.append(o); rate = rate + o.mean().detach() / T
        return torch.stack(outs, 1), (v, F_), {'out_rate': float(rate), 'F_abs': float(F_.abs().mean().detach())}

if __name__ == '__main__':
    torch.manual_seed(0); cell = SpikingLinearRecurrentCell(16, 64, 16, 16); x = torch.randn(4, 50, 16)
    y, (v, F_), st = cell(x); assert y.shape == (4, 50, 64) and set(y.unique().tolist()) <= {0.0, 1.0}; print('shapes ok, binary outputs ok, stats', st)
    y.sum().backward(); gn = {n: p.grad.abs().mean().item() for n, p in cell.named_parameters() if p.grad is not None}; print('grads reach', len(gn), 'of', len(list(cell.parameters())), 'params; inp grad', round(gn['inp.weight'], 6))
    # state size does not grow with T
    _, (v2, F2), _ = cell(torch.randn(4, 500, 16)); assert F2.shape == F_.shape and v2.shape == v.shape; print('state size independent of T: ok', tuple(F2.shape))
    # streaming == one-shot (equivalence): process in two chunks carrying the state
    with torch.no_grad():
        y_full, s_full, _ = cell(x); y1, s1, _ = cell(x[:, :20]); y2, s2, _ = cell(x[:, 20:], s1); assert torch.equal(torch.cat([y1, y2], 1), y_full); print('chunked streaming equals one-shot: ok')
    # fast weights really change during the forward pass, and a written (key,value) is readable back
    with torch.no_grad():
        c = SpikingLinearRecurrentCell(8, 32, 16, 8, lam=1.0, eta=1.0); F0 = torch.zeros(1, 8, 16); k = (torch.rand(1, 16) > 0.5).float(); val = torch.randn(1, 8)
        F1 = F0 + 1.0 * torch.bmm((val - torch.bmm(F0, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1), k.unsqueeze(1)) / (k.pow(2).sum() + 1.0)
        rec = torch.bmm(F1, k.unsqueeze(-1)).squeeze(-1); print('one write recovers value fraction:', round((rec.norm() / val.norm()).item(), 3), '(=|k|^2/(|k|^2+1) < 1 by design; repeated writes -> 1)')
        for _ in range(8): F1 = F1 + torch.bmm((val - torch.bmm(F1, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1), k.unsqueeze(1)) / (k.pow(2).sum() + 1.0)
        print('after 9 writes cosine to value:', round(torch.cosine_similarity(torch.bmm(F1, k.unsqueeze(-1)).squeeze(-1), val).item(), 4))
    _, _, s_on = cell(x); _, _, s_off = cell(x, fast=False); print('fast on F_abs', round(s_on['F_abs'], 4), 'vs off', round(s_off['F_abs'], 4))
