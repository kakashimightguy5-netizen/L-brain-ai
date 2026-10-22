"""H4 (trace side) - what grouped spikes would buy, measured on the CURRENT (ungrouped, G=1) model's FFN activity.
For group size G: active groups / token, spikes per active group, index entries (1 per spike for G=1, 1 per active group for G>1),
and weight columns fetched if a whole group is fetched when any member fires (waste = fetched/needed - 1).
This shows the structure that exists WITHOUT training for it; the trained-with-groups result (spike_groups flag, 3 seeds) is in h4_groups_train.py."""
import numpy as np
from common import load, tags, save_csv, plt, OUT
rows = []
for tag in tags():
    E = load(tag, 'eval'); A = E['ffn'] != 0; T, L, F = A.shape; nnz = A.sum() / T
    for G in (1, 4, 8, 16, 32):
        g = A.reshape(T, L, F // G, G).any(-1) if F % G == 0 else None
        ng = g.sum() / T
        # same G but neurons regrouped by co-activation on CAL would be an upper bound; we report the naive contiguous grouping
        rows.append(dict(model=tag, G=G, spikes_per_token=round(nnz, 1), active_groups_per_token=round(ng, 1), spikes_per_active_group=round(nnz / ng, 3),
                         index_entries_per_token=round(ng if G > 1 else nnz, 1), columns_fetched_per_token=round(ng * G, 1),
                         fetch_waste=round(ng * G / nnz - 1, 3), random_expectation_spikes_per_group=round(G * (nnz / (L * F)) / (1 - (1 - nnz / (L * F)) ** G), 3)))
        print(rows[-1])
save_csv('h4_groups_trace.csv', rows)
