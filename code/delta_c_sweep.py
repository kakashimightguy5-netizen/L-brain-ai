"""Energy fix F3 (GPU job): delta-memory chunk size sweep. Same math at any chunk size C (exact chunked delta rule), so output must match C=64 to float rounding.
Builds L5 with DELTA_C in {16,32,64,128} from ONE state dict, checks max |logit diff| vs C=64, times a full training step (eager fp32 and eager bf16), interleaved reps.
Writes ../results/delta_c_sweep.json. No power/clock settings changed. Shares the GPU with other running jobs (timings indicative)."""
import os, sys, json, time, copy, statistics as st, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]; os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, fastmem
assert torch.cuda.is_available(); DEV = 'cuda'; B, T = 16, 256; REPS = int(os.environ.get('AB_REPS', 5)); STEPS = int(os.environ.get('AB_STEPS', 10)); CS = [16, 32, 64, 128]
torch.manual_seed(0); X = torch.randint(0, 256, (B, T)); REC = fastmem.hippo_batch(X).to(DEV); X = X.to(DEV)
os.environ['DELTA_C'] = '64'; ref = M5.L5Model(d=128, n_layers=4, heads=4).to(DEV); sd = copy.deepcopy(ref.state_dict())
def build(C):
    os.environ['DELTA_C'] = str(C); m = M5.L5Model(d=128, n_layers=4, heads=4).to(DEV); m.load_state_dict(sd); return m
models = {C: build(C) for C in CS}
R = {'device': torch.cuda.get_device_name(0), 'batch': [B, T], 'note': 'shares the GPU with other running jobs', 'max_logit_diff_vs_C64': {}}
with torch.no_grad():
    y0 = models[64](X[:, :-1], REC[:, :, :-1])[0].float()
    for C in CS: R['max_logit_diff_vs_C64'][str(C)] = float((models[C](X[:, :-1], REC[:, :, :-1])[0].float() - y0).abs().max())
opts = {C: torch.optim.AdamW(models[C].parameters(), 1e-3) for C in CS}
def step(C, bf16):
    m = models[C]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16): lg = m(X[:, :-1], REC[:, :, :-1])[0]
    loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1)); opts[C].zero_grad(); loss.backward(); opts[C].step()
def timed(C, bf16):
    for _ in range(2): step(C, bf16)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(STEPS): step(C, bf16)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / STEPS * 1000
for bf16 in (False, True):
    ms = {C: [] for C in CS}
    for _ in range(REPS):
        for C in CS: ms[C].append(timed(C, bf16))
    R['bf16' if bf16 else 'fp32'] = {str(C): {'ms_median': round(st.median(ms[C]), 2), 'ms_all': [round(v, 1) for v in ms[C]]} for C in CS}
R['peak_vram_GB'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True); json.dump(R, open(os.path.join(HERE, '..', 'results', 'delta_c_sweep.json'), 'w'), indent=1); print(json.dumps(R), flush=True)
