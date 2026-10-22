"""Long-context generation cost: L (constant state) vs transformer (KV cache grows). Batch 1, token-by-token, GPU.
Weights are the same-size configs (cost does not depend on training). Also checks GraphStepper == Stepper.
Writes results/longctx.json"""
import os, sys, time, json, threading, torch
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import l_model as M0, l5_model as M5, l6_step, t_step
try: import pynvml
except ImportError:
    import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nvidia-ml-py']); import pynvml
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def joules(fn):
    smp = []; run = [True]
    def loop():
        while run[0]: smp.append((time.time(), pynvml.nvmlDeviceGetPowerUsage(H) / 1000.0)); time.sleep(0.01)
    th = threading.Thread(target=loop, daemon=True); torch.cuda.synchronize(); t0 = time.time(); th.start(); fn(); torch.cuda.synchronize(); run[0] = False; th.join()
    return time.time() - t0, sum((smp[i + 1][0] - smp[i][0]) * (smp[i][1] + smp[i + 1][1]) / 2 for i in range(len(smp) - 1))
time.sleep(2); _, j = joules(lambda: time.sleep(5)); IDLE = j / 5
dev = 'cuda'; DM = 352; CTX = [256, 1024, 4096, 16384]; N = 512
data = list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_val.txt'), 'rb').read()[:CTX[-1] + N + 10])
R = {'gpu': torch.cuda.get_device_name(0), 'idle_W': round(IDLE, 2), 'batch': 1, 'tokens_timed_per_point': N}
torch.manual_seed(0); L = M5.L5Model(d=DM, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True).to(dev).eval()
# equivalence: graph stepper vs plain stepper (first 300 bytes)
a, b = l6_step.Stepper(L, 1), l6_step.GraphStepper(L, 1); err = 0.0
for t in range(300): err = max(err, (a.step([data[t]]) - b.step([data[t]])).abs().max().item())
R['GraphStepper_vs_Stepper_max_diff'] = err; print('graph equivalence', err, flush=True)
def L_point(stepper_cls, ctx):
    st = stepper_cls(L, 1)
    for t in range(ctx): st.step([data[t]])                       # really walk through the whole context (state is O(1))
    sec, J = joules(lambda: [st.step([data[ctx + i]]) for i in range(N)])
    return {'ms_per_token': round(sec / N * 1000, 3), 'mJ_per_token_above_idle': round((J - IDLE * sec) / N * 1000, 2), 'state_MB': round(sum(x.numel() * 4 for x in st.S + st.h + [c for c in st.convbuf if c is not None]) / 1e6, 3)}
def T_point(ctx, cls):
    torch.manual_seed(0); Tm = M0.TModel(d=DM, n_layers=6, T=ctx + N + 8).to(dev).eval()     # cache capacity = this context
    st = cls(Tm, 1); st.prefill(torch.tensor([data[:ctx]], device=dev))
    sec, J = joules(lambda: [st.step([data[ctx + i]]) for i in range(N)])
    return {'ms_per_token': round(sec / N * 1000, 3), 'mJ_per_token_above_idle': round((J - IDLE * sec) / N * 1000, 2), 'kv_cache_MB_used': round(sum(k[:, :, :ctx].numel() * 4 * 2 for k in st.K) / 1e6, 2)}
for ctx in CTX:
    R[f'ctx {ctx}'] = {'transformer KV-cache (eager)': T_point(ctx, t_step.TStepper), 'transformer KV-cache (CUDA graph)': T_point(ctx, t_step.TGraphStepper), 'L GraphStepper': L_point(l6_step.GraphStepper, ctx)}
    print(ctx, R[f'ctx {ctx}'], flush=True)
json.dump(R, open(os.path.join(OUT, 'longctx.json'), 'w'), indent=1); print(json.dumps(R, indent=1))
