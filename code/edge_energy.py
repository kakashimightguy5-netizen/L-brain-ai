"""'Fix the big matrices' evaluation: L-Edge variants trained with flags (3 seeds each)
   base (E_G1), ternary weights (E_tern), low-rank dense projections r=0.25 (E_lr25), both (E_tlr25), plus int8/int4/mx4 QAT runs.
Per variant and seed: bits/byte (from the training run's own held-out evaluation, results_L1_*.json), firing rates,
MODELLED work per byte (dense MACs + event-driven MACs, multiplies vs additions), weight storage bits,
ESTIMATED chip energy per byte (45 nm per-op energies, configs/sustainability.yaml values copied below),
and MEASURED laptop-GPU decode energy per byte (batch 1, GraphStepper, 5 repeats) - on a GPU ternary/low-bit weights are
still computed in floating point, so only the low-rank change can show up in GPU energy (stated, not hidden).
Writes LOUT/edge_energy.json"""
import os, sys, time, json, math, glob, torch, torch.nn as nn
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import torch.nn.utils.parametrize as P
import l_model as M0, l5_model as M5, l6_step, fastmem, hwflags
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
E_MUL, E_ADD, E_SRAM_B = 0.2e-12, 0.03e-12, 1.25e-12          # ESTIMATED, Horowitz ISSCC 2014 (45 nm)
VARIANTS = {'E_G1': ('none', 0), 'E_tern': ('ternary', 0), 'E_lr25': ('none', 0.25), 'E_tlr25': ('ternary', 0.25),
            'E_lam10': ('none', 0), 'E_lam30': ('none', 0), 'E_int8': ('int8', 0), 'E_int4': ('int4', 0), 'E_mx4p32': ('mx4_pow2', 0), 'E_mx4f32': ('mx4_float', 0)}
BITS = {'none': 8, 'int8': 8, 'int4': 4, 'mx4_pow2': 4, 'mx4_float': 4, 'ternary': 2}   # 'none' = deployed at int8 (L-HW)
data = list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_val.txt'), 'rb').read()); half = len(data) // 2
def build(q, lr):
    m = M5.L5Model(d=128, n_layers=4, heads=4)
    if lr > 0: hwflags.apply_lowrank(m, lr)
    if q != 'none': hwflags.apply_qat(m, q, 32)
    return m
def bake(m):
    for mod in m.modules():
        if isinstance(mod, nn.Linear) and P.is_parametrized(mod, 'weight'): P.remove_parametrizations(mod, 'weight', leave_parametrized=True)
    return m
def firing(m):
    store = []; orig = M0.spikes
    def spy(z, lo, hi):
        c = orig(z, lo, hi); store.append((c != 0).float().mean(-1).mean().item()); return c
    M0.spikes = spy
    with torch.no_grad():
        x = torch.tensor([data[half + i * 1024: half + i * 1024 + 1024] for i in range(4)], device=dev); m(x, fastmem.hippo_batch(x))
    M0.spikes = orig; return sum(store[0::2]) / len(store[0::2]), sum(store[1::2]) / len(store[1::2])
def work(m, q, rs, rf):
    """per byte: dense = every Linear not driven by spikes; event = out (state spikes) and f2 (FFN spikes) rows actually used"""
    dense = event = 0
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            o, i = mod.weight.shape
            if name.endswith('.out'): event += rs * o * i
            elif name.endswith('.f2'): event += rf * o * i
            else: dense += o * i
    tern = q == 'ternary'; b = BITS[q]
    # ternary: every MAC becomes a conditional add (no multiplier); spike-driven rows: count x weight = repeated add / shift-add
    e = (dense + event) * ((E_ADD if tern else E_MUL + E_ADD) + b / 8 * E_SRAM_B)
    return dense, event, e
R = {'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', 'variants': {}}
import energy as EN
idle, _ = EN.idle_watts() if dev == 'cuda' else (None, None); R['idle_W'] = idle
for tag, (q, lr) in VARIANTS.items():
    rows = []
    for s in range(3):
        ck = os.path.join(OUT, f'{tag}_s{s}.pt'); rj = os.path.join(OUT, f'results_L1_{tag}_s{s}.json')
        if not os.path.exists(ck): continue
        m = build(q, lr); m.load_state_dict(torch.load(ck, map_location='cpu')); m = bake(m).to(dev).eval()
        res = json.load(open(rj)) if os.path.exists(rj) else {}
        rs, rf = firing(m); dense, event, eJ = work(m, q, rs, rf)
        row = {'seed': s, 'bits_per_byte': res.get('val_bits_per_byte'), 'state_firing': round(rs, 4), 'ffn_firing': round(rf, 4),
               'dense_MACs_per_byte': int(dense), 'event_MACs_per_byte': int(event), 'weight_storage_KB': round(hwflags.storage_bits(build(q, lr), q if q != 'none' else 'int8', 32) / 8 / 1024, 1),
               'chip_nJ_per_byte (ESTIMATED)': round(eJ * 1e9, 3), 'multiplier_free': q == 'ternary'}
        if dev == 'cuda':
            st = l6_step.GraphStepper(m, 1); seq = data[half: half + 1024]
            def f():
                for t in range(1024): st.step([seq[t]])
            f(); summ, _ = EN.repeat(f, 5, 1024, idle, torch.cuda.synchronize); row['GPU_measured'] = summ; del st
        rows.append(row); print(tag, row, flush=True)
    if rows: R['variants'][tag] = rows
    json.dump(R, open(os.path.join(OUT, 'edge_energy.json'), 'w'), indent=1)
print('done')
