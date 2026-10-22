"""Token-by-token (recurrent) inference for L-5/L-6: constant work and constant memory per byte, no re-reading of the past.
This is the path a hardware implementation runs. It must equal the full-window forward of the same model (tests/).
State per stream: hippocampus dicts, trajectory pointer, layout counters, per layer (conv buffer, fast-weight matrices S,
membrane h). Batch of independent streams supported (B >= 1)."""
import torch, torch.nn.functional as F
import l_model as M0, fastmem

class Stepper:
    def __init__(s, m, B=1):
        s.m, s.B = m, B; p = next(m.parameters()); s.dev, s.dt = p.device, p.dtype
        s.seq = [[] for _ in range(B)]
        s.hidx = [[{} for _ in M0.Hippocampus.ORDERS] for _ in range(B)]
        s.tidx = [{6: {}, 12: {}} for _ in range(B)]; s.ptr = [None] * B; s.last_pred = [None] * B
        s.depth = [0] * B; s.col = [0] * B; s.ind = [0] * B; s.pind = [0] * B; s.leading = [True] * B
        s.convbuf, s.S, s.h = [], [], []
        for fw, l in zip(m.fw, m.layers):
            inner = fw.h * fw.dk
            s.convbuf.append(torch.zeros(B, fw.K - 1, 3 * inner, device=s.dev, dtype=s.dt) if fw.conv is not None else None)
            s.S.append(torch.zeros(B, fw.h, fw.dk, fw.dk, device=s.dev, dtype=s.dt)); s.h.append(torch.zeros(B, l.theta.numel(), device=s.dev, dtype=s.dt))
    # ---------- O(1) memories ----------
    def _recall(s, i, t):
        seq = s.seq[i]; out = []
        for oi, k in enumerate(M0.Hippocampus.ORDERS):
            r = 256
            if t + 1 >= k:
                key = tuple(seq[t + 1 - k:t + 1]); d = s.hidx[i][oi]
                if key in d: r = seq[d[key] + 1]
                d[key] = t
            out.append(r)
        hit = 2 if s.last_pred[i] is None else int(s.last_pred[i] == seq[t]); found = None
        for k in (12, 6):
            if t + 1 >= k:
                p = s.tidx[i][k].get(tuple(seq[t + 1 - k:t + 1]))
                if p is not None and found is None: found = p
        for k in (6, 12):
            if t + 1 >= k: s.tidx[i][k][tuple(seq[t + 1 - k:t + 1])] = t
        if found is not None: s.ptr[i] = found
        elif s.ptr[i] is not None: s.ptr[i] += 1
        if s.ptr[i] is not None and s.ptr[i] + 1 <= t: tr = seq[s.ptr[i] + 1]; s.last_pred[i] = tr
        else: tr = 256; s.last_pred[i] = None
        b = seq[t]; s.depth[i] += (b in (40, 91, 123)) - (b in (41, 93, 125))      # raw counter; clamp only on read (== fastmem.bracket_depth)
        # layout place cells (same as fastmem.layout)
        if b == 10: s.pind[i] = s.ind[i]; s.col[i] = 0; s.ind[i] = 0; s.leading[i] = True; col, ind, pind = 0, 0, s.pind[i]
        else:
            s.col[i] += 1
            if s.leading[i] and b == 32: s.ind[i] += 1
            elif b != 32: s.leading[i] = False
            col, ind, pind = s.col[i], s.ind[i], s.pind[i]
        lay = (min(col, 64), min(ind, 32), min(pind, 32))
        if getattr(s.m, 'bracket', False): lay = lay + (min(max(s.depth[i], 0), 16),)
        return out, tr, hit, lay
    @torch.no_grad()
    def step(s, bytes_):
        """bytes_: list of B ints. Returns log-probabilities (B, 256) for the NEXT byte."""
        recs, trs, hits, lays = [], [], [], []
        for i, b in enumerate(bytes_):
            s.seq[i].append(int(b)); r, tr, hit, lay = s._recall(i, len(s.seq[i]) - 1); recs.append(r); trs.append(tr); hits.append(hit); lays.append(lay)
        ints = torch.tensor([[int(b) for b in bytes_]] + [list(c) for c in zip(*recs)] + [trs, hits] + [list(c) for c in zip(*lays)], dtype=torch.long)
        return s.neural(ints.to(s.dev, non_blocking=True))
    @torch.no_grad()
    def neural(s, ints):
        """ints (9, B) long on device: byte, 3 recalls, trajectory, hit, col, indent, prev-indent. Updates states IN PLACE."""
        m = s.m; x = ints[0]; rec = ints[1:4]; tr = ints[4]; hit = ints[5]
        he = m.hemb(hit); g = torch.sigmoid(m.tgate(he))
        z = m.emb(x) + sum(e(r) for e, r in zip(m.remb, rec)) + g * m.temb(tr) + he
        if getattr(m, 'colpos', False): z = z + m.colemb(ints[6]) + m.indemb(ints[7]) + m.pindemb(ints[8])
        if getattr(m, 'bracket', False): z = z + m.bracketemb(ints[9])
        if getattr(m, 'copyfeat', False): z = z + m.cfemb(fastmem.copy_feat(rec, tr))
        for li, (fw, l) in enumerate(zip(m.fw, m.layers)):
            xn = fw.n(z); zz = fw.qkv(xn)
            if fw.conv is not None:
                buf = torch.cat([s.convbuf[li], zz[:, None]], 1); s.convbuf[li].copy_(buf[:, 1:])
                zz = (buf * fw.conv.weight[:, 0, :].T[None]).sum(1)
            zz = F.silu(zz); q, k, v = zz.view(s.B, 3, fw.h, fw.dk).unbind(1)
            k = F.normalize(k, dim=-1); q = F.normalize(q, dim=-1)
            a = F.logsigmoid(fw.ga(xn)).exp()[..., None, None]; bb = torch.sigmoid(fw.gb(xn))[..., None]
            Sa = a * s.S[li]; s.S[li].copy_(Sa + (bb * (v - (Sa @ k[..., None])[..., 0]))[..., None] * k[..., None, :])
            o = (s.S[li] @ q[..., None])[..., 0]; z = z + fw.o(o.reshape(s.B, -1))
            u, zg = l.inp(l.n1(z)).chunk(2, -1); aa = l.gate(zg); b2 = (1 - aa.abs()) * u
            s.h[li].copy_(aa * s.h[li] + b2); h = s.h[li]; th = l.theta.abs().clamp(min=0.05)
            gain = h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() if l.homeo else 1.0
            cnt = M0.spikes(h / (gain * th), -l.K, l.K); z = z + l.out(cnt * th)
            gg = l.f1(l.n2(z)); th2 = l.theta2.abs().clamp(min=0.05); cnt2 = M0.spikes(F.relu(gg) / th2, 0, l.K)
            z = z + l.f2(cnt2 * th2)
        hz = m.norm(z); logits = m.head(hz)
        if getattr(m, 'mixcopy', False):
            mem = list(rec) + [tr]; t_ = m.trust(hz); t_ = torch.cat([t_[..., :-1], t_[..., -1:] + torch.log(g.clamp(min=1e-6))], -1)
            avail = torch.stack([(r < 256) for r in mem], -1)
            lw = torch.cat([torch.zeros_like(t_[..., :1]), t_.masked_fill(~avail, float('-inf'))], -1).log_softmax(-1)
            lp = logits.log_softmax(-1) + lw[..., :1]
            for kk, r in enumerate(mem):
                add = torch.full_like(lp, float('-inf')).scatter(-1, r.clamp(max=255).unsqueeze(-1), lw[..., kk + 1:kk + 2]); lp = torch.logaddexp(lp, add)
            return lp
        w = F.softplus(m.trust(hz))
        if getattr(m, 'tsign', False): w = w - F.softplus(m.veto(hz))
        w = torch.cat([w[..., :-1], w[..., -1:] * g], -1); boost = torch.zeros_like(logits)
        for kk, r in enumerate(list(rec) + [tr]):
            boost = boost.scatter_add(-1, r.clamp(max=255).unsqueeze(-1), (w[..., kk] * (r < 256).to(w.dtype) * 8.0).unsqueeze(-1))
        return (logits + boost).log_softmax(-1)

class GraphStepper(Stepper):
    """Same as Stepper, but the neural part is captured ONCE as a CUDA graph and replayed every byte
    (removes ~400 kernel launches of Python overhead per byte). Memory part (hash dicts) stays on the CPU."""
    def __init__(s, m, B=1):
        super().__init__(m, B); s.ints = torch.zeros(10 if getattr(m, 'bracket', False) else 9, B, dtype=torch.long, device=s.dev); s.graph = None
    @torch.no_grad()
    def neural(s, ints):
        s.ints.copy_(ints)
        if s.graph is None:
            saved = [t.clone() for t in s.convbuf + s.S + s.h if t is not None]
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2): Stepper.neural(s, s.ints)
            torch.cuda.current_stream().wait_stream(st)
            for t, v in zip([t for t in s.convbuf + s.S + s.h if t is not None], saved): t.copy_(v)   # undo warm-up updates
            s.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(s.graph): s.out = Stepper.neural(s, s.ints)
            for t, v in zip([t for t in s.convbuf + s.S + s.h if t is not None], saved): t.copy_(v)   # undo capture-time update
        s.graph.replay(); return s.out.clone()
