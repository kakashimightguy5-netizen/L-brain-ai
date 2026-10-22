"""Parts 1 + 4 of the efficiency-stack prompt: MEASURED GPU energy for L vs the transformer baseline.
 A. quality-matched decode (trained 9M checkpoints, batch 1, windows of 256 bytes = the transformer's trained context):
    J/token, J per correctly predicted byte, J per 1,000 output bytes, bits/byte, avg/peak W, temperature rise. 5 repeats.
 B. batch x context scaling (same-size configs, random weights: cost does not depend on weight values):
    batch 1/4/16/64 x context 256/2048/16384, L GraphStepper (really walks the whole context) vs transformer KV-cache decode
    as one CUDA graph (TGraphStepper, cache capacity = context). Points that do not fit in 6 GB are skipped and listed.
 C. training energy per token (batch 32 x 256), 5 repeats.
Warm runs only (warm-up before every measurement). Writes LOUT/eff_bench.json. One GPU job at a time.
Usage: python eff_bench.py [quick]"""
import os, sys, time, json, math, torch
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import torch.nn.functional as F
import l_model as M0, l5_model as M5, l6_step, t_step, fastmem, energy as EN
QUICK = len(sys.argv) > 1 and sys.argv[1] == 'quick'; REP = 2 if QUICK else 5; dev = 'cuda'
data = list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_val.txt'), 'rb').read()); half = len(data) // 2
sync = torch.cuda.synchronize
R = {'gpu': EN.gpu_info(), 'torch': torch.__version__, 'repeats': REP, 'labels': 'all numbers MEASURED on this machine unless marked'}
R['idle_W'], R['idle_detail'] = EN.idle_watts(); IDLE = R['idle_W']; print('idle', IDLE, R['gpu'], flush=True)
def save(): json.dump(R, open(os.path.join(OUT, 'eff_bench_quick.json' if QUICK else 'eff_bench.json'), 'w'), indent=1)
def mkL(): return M5.L5Model(d=352, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True).to(dev).eval()
def mkT(cap): return M0.TModel(d=352, n_layers=6, h=4, T=cap).to(dev).eval()
def reset_L(st):
    for t in st.convbuf + st.S + st.h:
        if t is not None: t.zero_()
    B = st.B; st.seq = [[] for _ in range(B)]; st.hidx = [[{} for _ in M0.Hippocampus.ORDERS] for _ in range(B)]
    st.tidx = [{6: {}, 12: {}} for _ in range(B)]; st.ptr = [None] * B; st.last_pred = [None] * B
    st.col = [0] * B; st.ind = [0] * B; st.pind = [0] * B; st.leading = [True] * B
# ---------------- A. quality-matched decode ----------------
W = 256; NWIN = 4 if QUICK else 12
L = mkL(); L.load_state_dict(torch.load(os.path.join(OUT, 'L6_9M_s0.pt'), map_location=dev))
T = mkT(256); T.load_state_dict(torch.load(os.path.join(OUT, 'T_9M_lr2_s0.pt'), map_location=dev))
gL = l6_step.GraphStepper(L, 1); gT = t_step.TGraphStepper(T, 1)
def run_L(acc):
    for w in range(NWIN):
        reset_L(gL); seg = data[half + w * W: half + w * W + W + 1]
        for t in range(W):
            lp = gL.step([seg[t]])[0]
            if acc is not None: acc.append((lp[seg[t + 1]].item(), int(lp.argmax().item() == seg[t + 1])))
def run_T(acc):
    for w in range(NWIN):
        gT.t = 0; seg = data[half + w * W: half + w * W + W + 1]
        for t in range(W):
            lp = gT.step([seg[t]])[0]
            if acc is not None: acc.append((lp[seg[t + 1]].item(), int(lp.argmax().item() == seg[t + 1])))
A = {}
for name, fn in (('L-6 9M (GraphStepper)', run_L), ('Transformer 9M (KV cache, CUDA graph)', run_T)):
    acc = []; fn(acc)                                                   # warm-up pass also gives quality (identical every pass)
    bpb = -sum(a for a, _ in acc) / len(acc) / math.log(2); correct = sum(c for _, c in acc)
    # timing/energy passes without the .item() quality reads (those force a sync per byte for both models equally - excluded)
    summ, runs = EN.repeat(lambda: fn(None), REP, NWIN * W, IDLE, sync)
    J = summ['mJ_per_token_mean'] / 1000; Ja = summ['mJ_per_token_above_idle_mean'] / 1000
    A[name] = {'bits_per_byte': round(bpb, 4), 'top1_acc': round(correct / len(acc), 4), 'tokens_per_repeat': NWIN * W, **summ,
               'mJ_per_correct_byte': round(J * len(acc) / correct * 1000, 4), 'mJ_per_correct_byte_above_idle': round(Ja * len(acc) / correct * 1000, 4),
               'J_per_1000_output_bytes': round(J * 1000, 3), 'ms_per_token': round(summ['seconds_mean'] / (NWIN * W) * 1000, 3)}
    print(name, A[name], flush=True)
R['A_quality_matched_decode_ctx256_batch1'] = A; save()
del gL, gT, L, T; torch.cuda.empty_cache()
# ---------------- B. batch x context scaling ----------------
CTX = [256, 2048] if QUICK else [256, 2048, 16384]; BS = [1, 4] if QUICK else [1, 4, 16, 64]; N = 128; FREE = 3.5e9
Bres, skipped = {}, []
torch.manual_seed(0); Lr = mkL()
for ctx in CTX:
    for B in BS:
        key = f'ctx {ctx} | batch {B}'; Bres[key] = {}
        # L: walk the real context, then time N more bytes for B streams
        st = l6_step.GraphStepper(Lr, B); src = [data[(b * 997) % half:] for b in range(B)]
        t0 = time.time()
        for t in range(ctx): st.step([s[t] for s in src])
        walk = time.time() - t0; pos = [ctx]
        def fL():
            for i in range(N): st.step([s[pos[0] + i] for s in src])
            pos[0] += N
        fL(); summ, _ = EN.repeat(fL, REP, N * B, IDLE, sync)
        Bres[key]['L'] = {**summ, 'ms_per_step': round(summ['seconds_mean'] / N * 1000, 3), 'walk_seconds': round(walk, 1),
                          'state_MB_per_stream': round(sum(x[0].numel() * 4 for x in st.S + st.h + [c for c in st.convbuf if c is not None]) / 1e6, 3)}
        del st; torch.cuda.empty_cache()
        # transformer: cache capacity ctx + headroom; skip if the KV cache would not fit
        cap = ctx + (REP + 2) * N + 8; kv = 2 * 6 * B * cap * 352 * 4
        if kv > FREE: skipped.append({'point': key, 'model': 'transformer', 'kv_cache_GB_needed': round(kv / 1e9, 2)}); Bres[key]['T'] = 'skipped (KV cache > 3.5 GB of the 6 GB card)'
        else:
          try:
            torch.manual_seed(0); Tm = mkT(cap); stT = t_step.TGraphStepper(Tm, B)
            stT.prefill(torch.tensor([s[:ctx] for s in src], device=dev)); posT = [ctx]
            def fT():
                for i in range(N): stT.step([s[posT[0] + i] for s in src])
                posT[0] += N
            fT(); summ, _ = EN.repeat(fT, REP, N * B, IDLE, sync)
            Bres[key]['T'] = {**summ, 'ms_per_step': round(summ['seconds_mean'] / N * 1000, 3), 'kv_cache_MB': round(kv / 1e6, 1)}
            del stT, Tm
          except torch.cuda.OutOfMemoryError:
            Bres[key]['T'] = 'out of memory on the 6 GB card'; skipped.append({'point': key, 'model': 'transformer', 'reason': 'OOM'})
          torch.cuda.empty_cache()
        print(key, {m: (v['mJ_per_token_above_idle_mean'] if isinstance(v, dict) else v) for m, v in Bres[key].items()}, flush=True)
        R['B_batch_context_scaling'] = Bres; R['B_skipped'] = skipped; save()
del Lr; torch.cuda.empty_cache()
# ---------------- C. training energy ----------------
tr = torch.tensor(list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_val.txt'), 'rb').read()[:half]), dtype=torch.long)
def batch(B=32, Tn=256):
    ix = torch.randint(0, len(tr) - Tn - 1, (B,)); return torch.stack([tr[i:i + Tn + 1] for i in ix]).to(dev)
C = {}
for kind in ('L', 'T'):
    torch.manual_seed(0); m = (mkL() if kind == 'L' else mkT(256)).train(); opt = torch.optim.AdamW(m.parameters(), 1e-4); xs = [batch() for _ in range(10)]
    def f():
        for x in xs:
            inp = x[:, :-1]; lg = m(inp, fastmem.hippo_batch(inp))[0] if kind == 'L' else m(inp)[0]
            loss = F.cross_entropy(lg.float().reshape(-1, 256), x[:, 1:].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    f(); summ, _ = EN.repeat(f, REP, 10 * 32 * 256, IDLE, sync)
    C[kind] = {**summ, 'ms_per_step': round(summ['seconds_mean'] / 10 * 1000, 1), 'peak_VRAM_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print('train', kind, C[kind], flush=True); del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
R['C_training_energy_batch32x256'] = C; R['gpu_after'] = EN.gpu_info(); save(); print('done', flush=True)
