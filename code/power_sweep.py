"""Part 2 - GPU power management (no model changes): power-limit / clock-lock sweep for
 (a) L-6 9M training, (b) L-6 batch-1 generation (GraphStepper), (c) transformer 9M training.
Method: try `nvidia-smi -pl W` (power limit). If refused (laptop GPUs usually expose no software power limit, and it needs
administrator rights), try `nvidia-smi -lgc 0,MHz` (lock max graphics clock; also needs admin). If both are refused, the
sweep is reported as NOT PERMITTED and nothing is changed. Every change is undone in a finally: block
(`nvidia-smi -pl <default>` / `nvidia-smi -rgc`) and the final state is read back and saved.
Writes LOUT/power_sweep.json. All numbers MEASURED.
Admin variant (optional, by hand): open an Administrator PowerShell in the code folder and run
  ..\\venv\\Scripts\\python.exe -B power_sweep.py   (it restores defaults itself; or restore manually with: nvidia-smi -rgc)"""
import os, sys, time, json, subprocess, math, torch
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import torch.nn.functional as F, l_model as M0, l5_model as M5, l6_step, fastmem, energy as EN
dev = 'cuda'; REP = 3
def smi(*a):
    try:
        p = subprocess.run(['nvidia-smi', *a], capture_output=True, text=True, timeout=60); out = (p.stdout + p.stderr).strip()[-400:]
        bad = any(k in out.lower() for k in ('not supported', 'not permitted', 'insufficient', 'failed', 'unable', 'error', 'requires'))   # nvidia-smi returns 0 even when refused
        return (1 if bad else p.returncode), out
    except Exception as e: return -1, repr(e)
R = {'gpu_before': EN.gpu_info(), 'attempts': []}
info = R['gpu_before']; settings = []
rc, msg = smi('-pl', str(info['power_limit_W'] or 60)); R['attempts'].append({'cmd': f"nvidia-smi -pl {info['power_limit_W']}", 'rc': rc, 'out': msg})
if rc == 0 and info.get('power_limit_range_W'):
    lo, hi = info['power_limit_range_W']; settings = [('pl', round(hi - f * (hi - lo))) for f in (0.0, 0.2, 0.4, 0.6, 0.8)]
else:
    mx = info.get('sm_clock_max_MHz') or 1800
    rc, msg = smi('-lgc', f'405,{mx}'); R['attempts'].append({'cmd': f'nvidia-smi -lgc 405,{mx}', 'rc': rc, 'out': msg})
    if rc == 0: settings = [('lgc', v) for v in (2100, 1800, 1500, 1200, 900, 600)]   # all inside the supported 405-2100 MHz list
R['mode'] = settings[0][0] if settings else 'NOT PERMITTED (no power-limit or clock-lock control from this account)'
data = list(open(os.path.join(os.environ.get('LDATA', '.'), 'py_big_val.txt'), 'rb').read()); half = len(data) // 2
tr = torch.tensor(data[:half], dtype=torch.long)
def batch(B=32, Tn=256): ix = torch.randint(0, len(tr) - Tn - 1, (B,)); return torch.stack([tr[i:i + Tn + 1] for i in ix]).to(dev)
def workloads():
    torch.manual_seed(0); L = M5.L5Model(d=352, n_layers=6, heads=8, conv=4, colpos=True, mixcopy=True).to(dev); T = M0.TModel(d=352, n_layers=6, h=4, T=256).to(dev)
    oL, oT = torch.optim.AdamW(L.parameters(), 1e-4), torch.optim.AdamW(T.parameters(), 1e-4); xs = [batch() for _ in range(6)]
    def trainL():
        for x in xs:
            inp = x[:, :-1]; loss = F.cross_entropy(L(inp, fastmem.hippo_batch(inp))[0].float().reshape(-1, 256), x[:, 1:].reshape(-1)); oL.zero_grad(); loss.backward(); oL.step()
    def trainT():
        for x in xs:
            loss = F.cross_entropy(T(x[:, :-1])[0].float().reshape(-1, 256), x[:, 1:].reshape(-1)); oT.zero_grad(); loss.backward(); oT.step()
    g = l6_step.GraphStepper(L.eval(), 1)
    def genL():
        for t in range(512): g.step([data[half + t]])
    return [('L training (6 steps x 32 x 256)', trainL, 6 * 32 * 256), ('L batch-1 generation (512 bytes)', genL, 512), ('Transformer training (6 x 32 x 256)', trainT, 6 * 32 * 256)]
R['results'] = []
try:
    idle, _ = EN.idle_watts(); R['idle_W'] = idle; W = workloads()
    for name, f, n in W: f()                                                           # warm-up at default settings
    for kind, val in (settings or [('default', None)]):
        if kind == 'pl': rc, msg = smi('-pl', str(val))
        elif kind == 'lgc': rc, msg = smi('-lgc', f'{val},{val}')                 # min = max: a fixed graphics clock
        else: rc, msg = 0, 'no change'
        time.sleep(3); row = {'setting': f'{kind} {val}', 'rc': rc, 'msg': msg, 'clocks_after_set': smi('--query-gpu=clocks.sm,clocks.max.sm,power.limit', '--format=csv,noheader')[1]}
        for name, f, n in W:
            f(); summ, _ = EN.repeat(f, REP, n, idle, torch.cuda.synchronize); row[name] = summ
        R['results'].append(row); print(json.dumps(row)[:600], flush=True)
        json.dump(R, open(os.path.join(OUT, 'power_sweep.json'), 'w'), indent=1)
finally:
    if settings and settings[0][0] == 'pl': R['restore'] = smi('-pl', str(info['default_power_limit_W'] or info['power_limit_W']))
    if settings and settings[0][0] == 'lgc': R['restore'] = smi('-rgc')
    R['gpu_after'] = EN.gpu_info(); json.dump(R, open(os.path.join(OUT, 'power_sweep.json'), 'w'), indent=1); print('restored, final state', R['gpu_after'], flush=True)
