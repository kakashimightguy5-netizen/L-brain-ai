"""Fix test for H2/H3: replace the FIXED hot set by a DYNAMIC working-set cache (neurons that fired in the last N bytes).
hit = share of this byte's FFN events whose neuron is in the cache; size = cache columns / all FFN columns (per layer).
Nothing is fitted (the cache is defined by the stream itself). EVAL traces."""
import numpy as np
from common import load, tags, save_csv, OUT
rows = []
for tag in tags():
    E = load(tag, 'eval'); A = E['ffn'] != 0; T, L, F = A.shape
    for N in (4, 8, 16, 32, 64, 128, 256):
        hitn = incn = 0
        for l in range(L):
            a = A[:, l]; cs = np.concatenate([np.zeros((1, F), np.int32), np.cumsum(a, 0, dtype=np.int32)]); lo = np.maximum(np.arange(T) - N, 0)
            inc = (cs[np.arange(T)] - cs[lo]) > 0; hitn += (a & inc).sum(); incn += inc.sum()
        hit = hitn / A.sum(); size = incn / A.size
        rows.append(dict(model=tag, window_bytes=N, hit_rate=round(float(hit), 4), cache_share_of_ffn_columns=round(float(size), 4), hit_per_size=round(float(hit / size), 2),
                         cache_KB_8bit=round(float(size) * L * F * E['D'] / 1024, 1))); print(rows[-1])
save_csv('fix_h3_working_set.csv', rows)
