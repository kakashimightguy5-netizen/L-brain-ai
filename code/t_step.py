"""Fair transformer decode baseline: token-by-token with a PRE-ALLOCATED KV cache (the standard fast way to generate).
Cost per token grows with context (attention reads the whole cache); memory grows linearly. Equivalence-tested."""
import torch, torch.nn.functional as F

class TStepper:
    def __init__(s, m, B=1):
        s.m, s.B, s.t = m, B, 0; cap = m.pos.size(0); b0 = m.blocks[0]; D = m.emb.weight.size(1); p = m.emb.weight
        shape = (B, b0.h, cap, D // b0.h)
        s.K = [torch.zeros(shape, device=p.device, dtype=p.dtype) for _ in m.blocks]; s.V = [torch.zeros(shape, device=p.device, dtype=p.dtype) for _ in m.blocks]
    @torch.no_grad()
    def step(s, bytes_):
        m = s.m; x = torch.tensor(bytes_, device=m.emb.weight.device); t = s.t
        if t >= m.pos.size(0): raise RuntimeError('context limit of the transformer reached')
        z = (m.emb(x) + m.pos[t])[:, None]
        for i, b in enumerate(m.blocks):
            B, _, D = z.shape; q, k, v = b.qkv(b.n1(z)).view(B, 1, 3, b.h, D // b.h).permute(2, 0, 3, 1, 4)
            s.K[i][:, :, t:t + 1] = k; s.V[i][:, :, t:t + 1] = v
            a = F.scaled_dot_product_attention(q, s.K[i][:, :, :t + 1], s.V[i][:, :, :t + 1])   # reads every cached position
            z = z + b.proj(a.transpose(1, 2).reshape(B, 1, D)); z = z + b.mlp(b.n2(z))
        s.t += 1; return m.head(m.n(z))[:, 0].log_softmax(-1)
    @torch.no_grad()
    def prefill(s, X):
        """X: (B, P) prefix bytes. Fills the KV cache in one parallel pass (standard prompt processing)."""
        m = s.m; B, P = X.shape; z = m.emb(X) + m.pos[:P]
        for i, b in enumerate(m.blocks):
            D = z.size(-1); q, k, v = b.qkv(b.n1(z)).view(B, P, 3, b.h, D // b.h).permute(2, 0, 3, 1, 4)
            s.K[i][:, :, :P] = k; s.V[i][:, :, :P] = v
            z = z + b.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, P, D)); z = z + b.mlp(b.n2(z))
        s.t = P

class TGraphStepper(TStepper):
    """Fair counterpart of L's GraphStepper: the transformer decode step captured as ONE CUDA graph and replayed.
    Shapes must be static, so attention runs over the whole pre-allocated cache with a mask (cache capacity = context)."""
    def __init__(s, m, B=1):
        super().__init__(m, B); dev = m.emb.weight.device; cap = m.pos.size(0)
        s.x = torch.zeros(B, dtype=torch.long, device=dev); s.tt = torch.zeros((), dtype=torch.long, device=dev); s.ar = torch.arange(cap, device=dev); s.graph = None
    @torch.no_grad()
    def _step(s):
        m = s.m; t = s.tt; z = (m.emb(s.x) + m.pos.index_select(0, t.view(1)))[:, None]
        mask = torch.where(s.ar <= t, 0.0, float('-inf')).to(z.dtype)[None, None, None]
        for i, b in enumerate(m.blocks):
            B, _, D = z.shape; q, k, v = b.qkv(b.n1(z)).view(B, 1, 3, b.h, D // b.h).permute(2, 0, 3, 1, 4)
            s.K[i].index_copy_(2, t.view(1), k); s.V[i].index_copy_(2, t.view(1), v)
            a = F.scaled_dot_product_attention(q, s.K[i], s.V[i], attn_mask=mask)
            z = z + b.proj(a.transpose(1, 2).reshape(B, 1, D)); z = z + b.mlp(b.n2(z))
        return m.head(m.n(z))[:, 0].log_softmax(-1)
    @torch.no_grad()
    def step(s, bytes_):
        s.x.copy_(torch.tensor(bytes_)); s.tt.fill_(s.t)
        if s.graph is None:
            Ks = [k.clone() for k in s.K]; Vs = [v.clone() for v in s.V]
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2): s._step()
            torch.cuda.current_stream().wait_stream(st)
            for k, k0 in zip(s.K, Ks): k.copy_(k0)
            for v, v0 in zip(s.V, Vs): v.copy_(v0)
            s.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(s.graph): s.out = s._step()
        s.graph.replay(); s.t += 1; return s.out.clone()
