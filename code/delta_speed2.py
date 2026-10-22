"""Where does the delta memory spend its time on the GPU, and is a block-recursive triangular inverse faster? (test)"""
import os, time, json, math, torch, torch.nn.functional as F
OUT = os.environ.get('LOUT', '.')
import l5_model as M5
dev = 'cuda'; torch.manual_seed(0); R = {}
B, H, T, dk, C = 32, 8, 256, 44, 64; N = T // C
def tm(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return round((time.time() - t) / n * 1000, 2)
k = F.normalize(torch.randn(B, H, N, C, dk, device=dev), dim=-1); b = torch.sigmoid(torch.randn(B, H, N, C, device=dev))
g = F.logsigmoid(torch.randn(B, H, N, C, device=dev) + 3).cumsum(-1)
Gm = g[..., :, None] - g[..., None, :]; tril0 = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev), -1)
A = torch.eye(C, device=dev) + b[..., :, None] * (k @ k.transpose(-1, -2)) * Gm.masked_fill(~tril0, float('-inf')).exp()
rhs = torch.randn(B, H, N, C, 2 * dk, device=dev)
R['trsm 64x64 (current)'] = tm(lambda: torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True))
def block_inv(A, bs=16):
    """exact inverse of unit-lower-triangular A by block recursion: invert 16x16 diagonal blocks (trsm on identity),
    then merge pairs: inv([[P,0],[Q,S]]) = [[P^-1,0],[-S^-1 Q P^-1, S^-1]]  (only true inverses -> numerically stable)"""
    n = A.size(-1); I = torch.eye(bs, device=A.device).expand(A.shape[:-2] + (bs, bs))
    blocks = [torch.linalg.solve_triangular(A[..., i:i + bs, i:i + bs], I, upper=False, unitriangular=True) for i in range(0, n, bs)]
    inv = torch.block_diag  # placeholder
    size = bs; cur = blocks
    while size < n:
        nxt = []
        for j in range(0, len(cur), 2):
            r0 = (j) * size; Pi, Si = cur[j], cur[j + 1]; Q = A[..., r0 + size:r0 + 2 * size, r0:r0 + size]
            low = -(Si @ Q @ Pi); top = torch.cat([Pi, torch.zeros_like(Pi)], -1); bot = torch.cat([low, Si], -1)
            nxt.append(torch.cat([top, bot], -2))
        cur = nxt; size *= 2
    return cur[0]
Ai = block_inv(A); R['block inverse max error vs trsm'] = (Ai @ rhs - torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)).abs().max().item()
R['block inverse + matmul'] = tm(lambda: block_inv(A) @ rhs)
Ainv = torch.linalg.solve_triangular(A, torch.eye(C, device=dev).expand_as(A), upper=False, unitriangular=True)
R['matmul with precomputed inverse only'] = tm(lambda: Ainv @ rhs)
R['build A (masked exp + kk^T)'] = tm(lambda: torch.eye(C, device=dev) + b[..., :, None] * (k @ k.transpose(-1, -2)) * (g[..., :, None] - g[..., None, :]).masked_fill(~tril0, float('-inf')).exp())
q = torch.randn(B, H, T, dk, device=dev, requires_grad=True); kk = F.normalize(torch.randn(B, H, T, dk, device=dev), dim=-1).requires_grad_(); v = torch.randn(B, H, T, dk, device=dev, requires_grad=True)
la = F.logsigmoid(torch.randn(B, H, T, device=dev) + 3).requires_grad_(); bb = torch.sigmoid(torch.randn(B, H, T, device=dev)).requires_grad_()
S0 = torch.zeros(B, H, dk, dk, device=dev)
R['delta_chunk2 forward'] = tm(lambda: M5.delta_chunk2(q, kk, v, la, bb, S0, 64))
R['delta_chunk2 fwd+bwd'] = tm(lambda: M5.delta_chunk2(q, kk, v, la, bb, S0, 64)[0].sum().backward())
print(json.dumps(R, indent=1)); json.dump(R, open(os.path.join(OUT, 'delta_speed2.json'), 'w'), indent=1)
