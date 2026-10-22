"""H5 - address-sorted spike queue: does processing a token's spikes in weight-address order cut external-memory cost?
MODELLED memory system (stated assumptions, not measured hardware):
  * f2 / out weights stored column-major in external SDRAM, one column = D weights (8-bit -> D bytes, 4-bit -> D/2 bytes),
    columns of one layer contiguous, layers back to back.
  * SDRAM page (row) = ROW bytes (default 2048; swept 1024/2048/4096), burst = 32 bytes (32-bit bus x burst-8; assumption).
  * cost of a token = row activations x ACT_CYC + bursts x BURST_CYC, with ACT_CYC = 6 (tRP+tRCD, ~2-3 cycles each at ~100 MHz for
    SDR SDRAM; assumption) and BURST_CYC = 8. Ratios between orderings matter, not the absolute cycles.
Orderings: dense (stream all columns), unsorted event list (spikes in a random order - e.g. emitted by parallel lanes),
sorted event list (ascending address; one activation per distinct row touched). Spike data MEASURED from the model (EVAL).
Note: if spikes are emitted as a bitmap scanned in index order they are ALREADY address-sorted; a sorter is only needed for
asynchronous/out-of-order spike sources."""
import numpy as np
from common import load, tags, save_csv, plt, OUT
ACT_CYC, BURST_CYC, BURST = 6, 8, 32; rows = []; rng = np.random.default_rng(0)
for tag in tags():
    E = load(tag, 'eval'); D = E['D']
    for unit, key in (('f2 (FFN spikes)', 'ffn'), ('out (state spikes)', 'state')):
        A = E[key][:5000] != 0; T, L, N = A.shape
        for bits in (8, 4):
            col = D * bits // 8; nb = -(-col // BURST)
            for ROW in (1024, 2048, 4096):
                addr_base = (np.arange(L)[:, None] * N + np.arange(N)[None]) * col     # byte address of each column
                act_s = act_u = ev = 0
                for t in range(T):
                    a = addr_base[A[t]]                                                   # ascending within the token (sorted)
                    rs = a // ROW; re_ = (a + col - 1) // ROW
                    # sorted: distinct rows touched (a column may span two rows)
                    act_s += len(np.unique(np.concatenate([rs, re_])))
                    p = rng.permutation(len(a)); ru, rue = rs[p], re_[p]
                    seq = np.stack([ru, rue], 1).ravel(); act_u += 1 + (seq[1:] != seq[:-1]).sum() if len(seq) else 0
                    ev += len(a)
                dense_rows = -(-(L * N * col) // ROW); dense_bursts = L * N * nb
                cyc = lambda acts, bursts: acts * ACT_CYC + bursts * BURST_CYC
                c_d = cyc(dense_rows, dense_bursts); c_u = cyc(act_u / T, ev / T * nb); c_s = cyc(act_s / T, ev / T * nb)
                rows.append(dict(model=tag, unit=unit, weight_bits=bits, row_bytes=ROW, events_per_token=round(ev / T, 1),
                                 row_acts_dense=dense_rows, row_acts_unsorted=round(act_u / T, 1), row_acts_sorted=round(act_s / T, 1),
                                 bytes_dense=L * N * col, bytes_events=round(ev / T * nb * BURST),
                                 cycles_dense=round(c_d), cycles_unsorted=round(c_u), cycles_sorted=round(c_s),
                                 sorted_vs_unsorted=round(c_s / c_u, 3), sorted_vs_dense=round(c_s / c_d, 3)))
                print(rows[-1])
save_csv('h5_spike_queue.csv', rows)
P = plt(); f, ax = P.subplots(figsize=(7, 3.8)); sel = [r for r in rows if r['row_bytes'] == 2048]
lab = [f"{r['model']}\n{r['unit'][:3]} {r['weight_bits']}b" for r in sel]; x = np.arange(len(sel))
for i, (k, n) in enumerate((('cycles_dense', 'dense'), ('cycles_unsorted', 'unsorted events'), ('cycles_sorted', 'sorted events'))):
    ax.bar(x + i * .27, [r[k] for r in sel], .27, label=n)
ax.set_xticks(x + .27); ax.set_xticklabels(lab, fontsize=6); ax.set_ylabel('modelled SDRAM cycles / token'); ax.legend(); ax.set_title('H5 spike-queue memory model (2 KB rows)')
f.tight_layout(); f.savefig(f'{OUT}/h5_spike_queue.png', dpi=110)
