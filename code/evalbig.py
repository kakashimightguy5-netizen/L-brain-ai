"""Big-sample evaluation of a saved L-pure LM checkpoint (fixes the noisy 6-window eval).
Usage (set the same MIX/K/D/NL/QUANT env as the training run):  python evalbig.py CKPT.pt [NWIN]"""
import os, sys, math, torch, torch.nn.functional as F
import lpure_lm8 as L
ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
L.SCHED['frac'] = 2.0; L.SCHED['alpha'] = 0.0 if L.QUANT == 0 else 1.0
m = L.LM().to(L.DEV); m.load_state_dict(torch.load(ck, map_location=L.DEV)); m.eval()
DATA = os.environ.get('LDATA', '/home/claude/data'); va = torch.tensor(list(open(os.path.join(DATA, 'py_big_val.txt'), 'rb').read()), dtype=torch.long); half = len(va) // 2
n = min(n, (len(va) - half - 1) // 1024); res = []
with torch.no_grad():
    for i in range(n):
        w = va[half + i * 1024: half + i * 1024 + 1025].to(L.DEV)[None]
        res.append(F.cross_entropy(m(w[:, :-1]).reshape(-1, 256), w[:, 1:].reshape(-1)).item() / math.log(2))
t = torch.tensor(res); print(os.path.basename(ck), 'windows', n, 'bpb mean', round(t.mean().item(), 4), '+/-', round((t.std() / math.sqrt(n)).item(), 4), '(std error)')
