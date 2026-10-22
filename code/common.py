"""Shared helpers for the SPARK hardware-hypothesis analyses (H1-H9).
Traces come from collect_traces.py: CAL = first half of the held-out file (used ONLY to fit tables / hot sets / thresholds),
EVAL = second half (used ONLY to measure). Nothing is fitted on EVAL.
Work model (per token, per layer), from the L-5/L-6 layer code (l1_model.LLayer1 + l5_model.DeltaMemory):
  dense units (input is not a spike vector -> every weight is used every token):
     inp  D x 2D,  f1  D x F,  qkv  D x 3D,  o  D x D   (+ tiny gate heads, ignored)
  event units (input IS an integer spike-count vector -> only columns of non-zero entries are read and used):
     out  D x D  driven by the membrane-state spikes (D entries, counts in [-K, K])
     f2   F x D  driven by the FFN spikes (F entries, counts in [0, K])
  plus the read-out head D x 256 once per token.
All work numbers below are MODELLED (counts of multiply-accumulates / weight bytes), not measured on hardware."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
TR = os.environ.get('HWH_TRACES', os.path.join(ROOT, 'results/hw_hypotheses/traces')); OUT = os.environ.get('HWH_OUT', os.path.join(ROOT, 'results/hw_hypotheses')); os.makedirs(OUT, exist_ok=True)
HEADS = {352: 8, 128: 4}

def load(tag, split):
    d = np.load(os.path.join(TR, f'{tag}_{split}.npz')); r = {k: d[k] for k in d.files}
    T, L, D = r['state'].shape; F = r['ffn'].shape[2]; r.update(T=T, L=L, D=D, F=F, H=HEADS.get(D, 4))
    r['inbyte'] = np.concatenate([[10], r['byte'][:-1]]).astype(np.int64)      # input byte at step t = target byte of step t-1
    return r

def macs(r):
    D, F = r['D'], r['F']
    dense = 2 * D * D + D * F + 3 * D * D + D * D
    return dict(dense_per_layer=dense, event_full_per_layer=D * D + F * D, head=D * 256)

def tags():
    import sys; return sys.argv[1:] or [t for t in ('L6s0', 'LEdge') if os.path.exists(os.path.join(TR, f'{t}_eval.npz'))]

def save_csv(name, rows):
    import csv; p = os.path.join(OUT, name)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return p

def plt():
    try: import matplotlib
    except ImportError:
        import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib']); import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as P; return P
