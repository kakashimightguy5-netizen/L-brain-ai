"""Matrix-free reference (vault note 131): hashed context-count tables + interpolated Witten-Bell blending (PPM-like). Zero matrices, zero gradients, zero learned weights:
training = counting (increments), inference = table lookups + a few adds/divides. Same bytes and same val slice as gate_lm.py. Usage: python ngram_lm.py [MAXORDER] [TRAIN_MB]
Reports bits/byte (MEASURED) and modelled table storage."""
import os, sys, json, math, time, numpy as np
MAXO = int(sys.argv[1]) if len(sys.argv) > 1 else 6; MB = int(sys.argv[2]) if len(sys.argv) > 2 else 8; DATA = os.environ.get('LDATA', '/home/claude/data'); HB = int(os.environ.get('HB', 19))
tr = np.frombuffer(open(os.path.join(DATA, 'py_big_train.txt'), 'rb').read(MB * 1_000_000), dtype=np.uint8).astype(np.int64)
va = np.frombuffer(open(os.path.join(DATA, 'py_big_val.txt'), 'rb').read(), dtype=np.uint8).astype(np.int64); va = va[len(va) // 2: len(va) // 2 + 200_000]
PR = 1_000_003
def ctx_hash(a, pos, n):   # hash of the n bytes before each position in pos
    h = np.zeros(len(pos), dtype=np.int64)
    for j in range(1, n + 1): h = (h * 257 + a[pos - j] + 1) % PR
    return h % (1 << HB) if n > 2 else h   # orders 0-2 collision-free enough via direct small hash
t0 = time.time(); tables = {}
for n in range(0, MAXO + 1):
    pos = np.arange(n, len(tr)); size = (1 << HB) if n > 2 else (PR if n == 2 else 257 ** n); size = min(size, 1 << HB) if n > 2 else size
    if n == 0: h = np.zeros(len(pos), dtype=np.int64)
    elif n == 1: h = tr[pos - 1]
    elif n == 2: h = (tr[pos - 1] * 256 + tr[pos - 2]) % (1 << 16)
    else: h = ctx_hash(tr, pos, n)
    rows = 1 if n == 0 else (256 if n == 1 else ((1 << 16) if n == 2 else (1 << HB)))
    flat = np.bincount(h * 256 + tr[pos], minlength=rows * 256).astype(np.uint32); tables[n] = np.minimum(flat, 65535).astype(np.uint16).reshape(rows, 256); del flat
    print('order', n, 'rows', rows, f'{time.time()-t0:.0f}s', flush=True)
pos = np.arange(MAXO, len(va)); y = va[pos]; P = np.full((len(pos), 256), 1 / 256)
for n in range(0, MAXO + 1):
    if n == 0: h = np.zeros(len(pos), dtype=np.int64)
    elif n == 1: h = va[pos - 1]
    elif n == 2: h = (va[pos - 1] * 256 + va[pos - 2]) % (1 << 16)
    else: h = ctx_hash(va, pos, n)
    C = tables[n][h].astype(np.float64); tot = C.sum(1, keepdims=True); T = (C > 0).sum(1, keepdims=True); lam = tot / (tot + T + 1e-9)
    P = np.where(tot > 0, lam * C / np.maximum(tot, 1) + (1 - lam) * P, P)
bpb = float(-np.log2(P[np.arange(len(pos)), y] + 1e-12).mean())
store_bits = sum(t.size * 16 for t in tables.values())
R = {'model': 'hashed_ngram_WB', 'maxorder': MAXO, 'train_MB': MB, 'hash_bits': HB, 'bpb': round(bpb, 4), 'table_bits_MODELLED_dense_uint16': store_bits, 'ops_per_byte_MODELLED': 'one table-row lookup per order + 256-wide adds (no multiplies by learned weights)', 'secs': round(time.time() - t0)}
print(json.dumps(R), flush=True); os.makedirs(os.environ.get('LOUT', 'results'), exist_ok=True); json.dump(R, open(os.path.join(os.environ.get('LOUT', 'results'), f'ngram_o{MAXO}_{MB}MB.json'), 'w'))
