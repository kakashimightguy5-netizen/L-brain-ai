"""Shared activity traces for the SPARK hardware hypotheses. Runs a trained L-6 (9M) over CONTIGUOUS held-out code and stores,
per byte and per layer: membrane-state spike counts (int8, [-8,8]) and FFN spike counts (int8, [0,8]); plus per-byte loss,
top-1 confidence, hippocampus/trajectory availability. Full-window forward == token-by-token stepper (tested to 1e-14),
so activity is identical to running byte by byte. Two splits: CAL = first half of the held-out file, EVAL = second half.
Usage: python collect_traces.py CKPT TAG [N_BYTES_PER_SPLIT=50000] [WINDOW=1024]"""
import os, sys, math, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path[:0] = [ROOT, HERE]
os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l_model as M0, l5_model as M5, fastmem
torch.set_num_threads(int(os.environ.get('THREADS', 1)))
ck, tag = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 50000; W = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
ARCH = os.environ.get('ARCH', 'L6')   # L6 = 9M L-6 (default); EDGE = L-Edge (d128, 4 layers, plain L-5 options)
m = (M5.L5Model(d=352, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True) if ARCH == 'L6' else M5.L5Model(d=128, n_layers=4, heads=4)).eval(); m.load_state_dict(torch.load(ck if os.path.exists(ck) else os.path.join(os.environ.get('LOUT', '.'), ck), map_location='cpu'))
va = torch.tensor(list(open(os.path.join(os.environ.get('LDATA', '/home/claude/data'), 'py_big_val.txt'), 'rb').read()), dtype=torch.long); half = len(va) // 2
OUT = os.environ.get('HWH_TRACES', os.path.join(ROOT, 'results/hw_hypotheses/traces')); os.makedirs(OUT, exist_ok=True)
for split, start in (('cal', 0), ('eval', half)):
    nw = (N + W - 1) // W; S, Fq, LOSS, CONF, HIP, TR, BYTES = [], [], [], [], [], [], []
    with torch.no_grad():
        for w in range(nw):
            x = va[start + w * W:start + w * W + W + 1][None]; inp, y = x[:, :-1], x[:, 1:]
            store = []; orig = M0.spikes
            def spy(z, lo, hi):
                c = orig(z, lo, hi); store.append(c[0].to(torch.int8)); return c
            M0.spikes = spy; rec = fastmem.hippo_batch(inp); tr, _ = fastmem.traj_batch(inp); lp = m(inp, rec)[0].log_softmax(-1); M0.spikes = orig   # log_softmax is a no-op on mixcopy log-probs, needed for logit models
            S.append(torch.stack(store[0::2], 1)); Fq.append(torch.stack(store[1::2], 1))           # (T, layers, D) / (T, layers, ffn)
            LOSS.append(-lp[0].gather(-1, y[0][:, None])[:, 0] / math.log(2)); CONF.append(lp[0].max(-1).values.exp())
            HIP.append((rec[:, 0] < 256).any(0)); TR.append(tr[0] < 256); BYTES.append(y[0])
    cat = lambda L: torch.cat(L)[:N].numpy()
    np.savez_compressed(os.path.join(OUT, f'{tag}_{split}.npz'), state=cat(S), ffn=cat(Fq), loss=cat(LOSS), conf=cat(CONF), hippo=cat(HIP), track=cat(TR), byte=cat(BYTES), window=W)
    print(split, 'saved', cat(S).shape, cat(Fq).shape, 'bpb', float(cat(LOSS).mean()), flush=True)
