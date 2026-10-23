import torch
def kv_batch(B, N, nkeys=64, nvals=10, g=None):
    """k1 v1 ... kN vN Q kq -> vq. tokens: keys 0..nkeys-1, vals nkeys..nkeys+nvals-1, Q = nkeys+nvals. Returns tokens (B,2N+2), target (B,)"""
    keys = torch.stack([torch.randperm(nkeys, generator=g)[:N] for _ in range(B)]); vals = torch.randint(0, nvals, (B, N), generator=g) + nkeys
    seq = torch.stack([keys, vals], 2).reshape(B, 2 * N); qi = torch.randint(0, N, (B,), generator=g); ar = torch.arange(B)
    q = torch.full((B, 1), nkeys + nvals); tok = torch.cat([seq, q, keys[ar, qi][:, None]], 1); return tok, vals[ar, qi] - nkeys
def copy_batch(B, delay, nb=8, g=None):
    """bits stream, predict bit from `delay` steps ago at every position (>=delay). tokens 0/1, target (B,T)"""
    T = 24; x = torch.randint(0, 2, (B, T), generator=g); y = torch.roll(x, delay, 1); return x, y
