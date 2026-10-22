"""GPU-native (vectorised) hippocampus + trajectory replay. Same results as the python dict versions in l_model/l1_model,
but computed with sorts and cumulative maxima on the GPU instead of per-byte python loops on the CPU.
Idea: 'most recent earlier position with the same last-k bytes' = previous element after a STABLE sort by key."""
import torch

def _prev_same(X, k):
    """X: (B,T) long bytes. For each t, most recent t' < t whose last-k-byte window equals t's window (both complete).
    Returns (B,T) long, -1 if none. Exact: keys are packed bytes (two int64 halves for k > 7)."""
    B, T = X.shape; dev = X.device; pos = torch.arange(T, device=dev)
    if T < k: return torch.full((B, T), -1, dtype=torch.long, device=dev)
    W = X.unfold(1, k, 1)                                             # (B, T-k+1, k) windows ending at t = k-1 .. T-1
    def pack(w):
        key = torch.zeros(w.shape[:-1], dtype=torch.long, device=dev)
        for j in range(w.size(-1)): key = key * 256 + w[..., j]
        return key
    h1 = pack(W[..., :min(k, 7)]); h2 = pack(W[..., 7:]) if k > 7 else torch.zeros_like(h1)
    n = h1.size(1); invalid = torch.full((B, k - 1), -1, dtype=torch.long, device=dev)
    # sort lexicographically by (h1, h2), stable so equal keys stay in position order
    o2 = torch.sort(h2, dim=1, stable=True).indices
    o1 = torch.sort(torch.gather(h1, 1, o2), dim=1, stable=True).indices; order = torch.gather(o2, 1, o1)
    s1, s2 = torch.gather(h1, 1, order), torch.gather(h2, 1, order)
    same = torch.zeros_like(s1, dtype=torch.bool); same[:, 1:] = (s1[:, 1:] == s1[:, :-1]) & (s2[:, 1:] == s2[:, :-1])
    prev_sorted = torch.full_like(order, -1); prev_sorted[:, 1:] = torch.where(same[:, 1:], order[:, :-1], torch.full_like(order[:, 1:], -1))
    prev = torch.full_like(order, -1).scatter(1, order, prev_sorted)   # window index of previous occurrence
    prev = torch.where(prev >= 0, prev + (k - 1), prev)                # window index -> end position t'
    return torch.cat([invalid, prev], 1)

def hippo_batch(X, orders=(3, 6, 12)):
    """(len(orders), B, T): byte that followed the most recent earlier occurrence of the last-k bytes (256 = none)."""
    out = []
    for k in orders:
        p = _prev_same(X, k); nxt = torch.gather(X, 1, (p + 1).clamp(min=0, max=X.size(1) - 1))
        out.append(torch.where(p >= 0, nxt, torch.full_like(X, 256)))
    return torch.stack(out, 0)

def traj_batch(X, orders=(6, 12)):
    """Vectorised Trajectory.recall: returns (r, hit) each (B,T)."""
    B, T = X.shape; dev = X.device; pos = torch.arange(T, device=dev).expand(B, T)
    found = torch.full((B, T), -1, dtype=torch.long, device=dev)
    for k in sorted(orders):                                            # longer order overrides shorter
        p = _prev_same(X, k); found = torch.where(p >= 0, p, found)
    last = torch.where(found >= 0, pos, torch.full_like(pos, -1)).cummax(1).values      # last position with a fresh match
    base = torch.gather(found, 1, last.clamp(min=0))
    ptr = torch.where(last >= 0, base + (pos - last), torch.full_like(pos, -1))
    ok = (ptr >= 0) & (ptr + 1 <= pos)
    r = torch.where(ok, torch.gather(X, 1, (ptr + 1).clamp(min=0, max=T - 1)), torch.full_like(X, 256))
    hit = torch.full_like(X, 2); prevr = r[:, :-1]
    hit[:, 1:] = torch.where(prevr < 256, (prevr == X[:, 1:]).long(), torch.full_like(prevr, 2))
    return r, hit

if __name__ == '__main__':
    import time, l_model as M0, l1_model as M1
    d = torch.tensor(list(open('/home/claude/data/py_big_val.txt', 'rb').read()[:400000]), dtype=torch.long)
    g = torch.Generator().manual_seed(0); ix = torch.randint(0, len(d) - 2100, (16,), generator=g)
    for T in (256, 2048):
        X = torch.stack([d[i:i + T] for i in ix])
        X[:4, T // 2:] = X[:4, :T - T // 2]                               # force long repeats (copy situations)
        a = M0.Hippocampus.batch(X); b = hippo_batch(X); print('T', T, 'hippo mismatches', (a != b).sum().item(), 'of', a.numel())
        (r1, h1), (r2, h2) = M1.Trajectory.batch(X), traj_batch(X)
        print('      traj mismatches', (r1 != r2).sum().item(), (h1 != h2).sum().item(), 'tracked', (r1 < 256).float().mean().item())
    X = torch.stack([d[i:i + 256] for i in ix]).repeat(2, 1)
    t = time.time(); M0.Hippocampus.batch(X); M1.Trajectory.batch(X); t1 = time.time() - t
    t = time.time(); hippo_batch(X); traj_batch(X); t2 = time.time() - t
    print(f'batch 32x256: python {t1*1000:.0f} ms, vectorised (CPU) {t2*1000:.0f} ms')

def layout(X, cmax=64, imax=32):
    """'Place cells' for code layout, per position t (describing the NEXT byte's situation):
    col   = column of the next byte (bytes since last newline), clamped;
    ind   = indentation (leading spaces) of the current line so far;  pind = indentation of the previous line.
    Vectorised with cumulative maxima. Returns three (B,T) long tensors."""
    B, T = X.shape; dev = X.device; pos = torch.arange(T, device=dev).expand(B, T); neg = torch.full_like(pos, -1)
    nl = X == 10; ns = (X != 32) & ~nl
    ls = torch.where(nl, pos, neg).cummax(1).values                       # last newline at or before t
    col = pos - ls                                                         # next byte's column (0 right after '\n')
    lastns_prev = torch.cat([neg[:, :1], torch.where(ns, pos, neg).cummax(1).values[:, :-1]], 1)
    first_ns = ns & (lastns_prev <= ls)                                    # first non-space byte of its line
    fns = torch.where(first_ns, pos, neg).cummax(1).values
    ind = torch.where(fns > ls, fns - ls - 1, pos - ls)                    # leading spaces of the current line (so far)
    ind = torch.where(nl, torch.zeros_like(ind), ind)
    prev_end = (ls - 1).clamp(min=0)
    pind = torch.where(ls >= 1, torch.gather(ind, 1, prev_end), torch.zeros_like(ind))
    pind = torch.where(nl, torch.gather(ind, 1, (pos - 1).clamp(min=0)), pind)   # at a newline, 'previous line' = line just ended
    return col.clamp(max=cmax), ind.clamp(max=imax), pind.clamp(max=imax)

def bracket_depth(X, dmax=16):
    """Integer COUNTER channel: nesting depth of ( [ { after each byte (opens +1, closes -1), clamped to [0,dmax].
    Unbounded-counter signal that saturating recurrent units cannot hold (Weiss 2018). Strings/comments are ignored on purpose."""
    d = ((X == 40) | (X == 91) | (X == 123)).long() - ((X == 41) | (X == 93) | (X == 125)).long()
    return d.cumsum(1).clamp(0, dmax)

def copy_feat(rec, tr):
    """Copy-gate signal (integer, 0..9): index = 2*n_avail + agree. n_avail = how many of the 4 memories (3 hippocampus orders +
    trajectory) recalled a byte (0..4); agree = 1 if at least two recalled and all recalled bytes are the same.
    Works for (B,T) full windows and (B,) single steps alike. Hardware: 4 comparators."""
    mem = torch.stack([rec[0], rec[1], rec[2], tr]); av = mem < 256; n = av.sum(0)
    mx = torch.where(av, mem, torch.full_like(mem, -1)).max(0).values; mn = torch.where(av, mem, torch.full_like(mem, 999)).min(0).values
    return 2 * n + ((n >= 2) & (mx == mn)).long()

def _layout_ref(seq):
    col, ind, pind = [], [], []; c = 0; cur = 0; leading = True; prev = 0
    for b in seq:
        if b == 10: prev = cur; c = 0; cur = 0; leading = True; col.append(0); ind.append(0); pind.append(prev); continue
        c += 1
        if leading and b == 32: cur += 1
        elif b != 32: leading = False
        col.append(c); ind.append(cur); pind.append(prev)
    return col, ind, pind
