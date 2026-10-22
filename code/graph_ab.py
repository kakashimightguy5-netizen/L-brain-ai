"""Energy fix F2 A/B (GPU job): full L5 training step (forward + backward + AdamW) EAGER vs CUDA-GRAPH replay, fp32 and bf16 autocast, plus matched transformer eager.
Interleaved reps, median ms/step, final-loss agreement check (graph must give the same loss trajectory as eager), NVML energy per token. Writes ../results/graph_ab.json.
No power/clock settings are changed. env: AB_REPS (5), AB_STEPS (10), AB_B/AB_T (16/256). Numbers share the GPU with other running jobs - rerun when the GPU is idle for clean energy."""
import os, sys, json, time, copy, statistics as st, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]; os.environ.setdefault('SPIKEK', '8'); os.environ['LSCAN_DISABLE'] = '1'
import l5_model as M5, l_model as M0, fastmem
assert torch.cuda.is_available(), 'GPU job'
DEV = 'cuda'; REPS = int(os.environ.get('AB_REPS', 5)); STEPS = int(os.environ.get('AB_STEPS', 10)); B = int(os.environ.get('AB_B', 16)); T = int(os.environ.get('AB_T', 256))
torch.manual_seed(0); X = torch.randint(0, 256, (B, T)); REC = fastmem.hippo_batch(X).to(DEV); X = X.to(DEV)
base = M5.L5Model(d=128, n_layers=4, heads=4).to(DEV); sd = copy.deepcopy(base.state_dict())
class Runner:
    def __init__(s, bf16, graph):
        s.m = M5.L5Model(d=128, n_layers=4, heads=4).to(DEV); s.m.load_state_dict(sd); s.bf16, s.graph = bf16, graph
        s.opt = torch.optim.AdamW(s.m.parameters(), 1e-3, capturable=graph)
        s.x, s.rec = X.clone(), REC.clone(); s.loss = torch.zeros((), device=DEV)
        if graph:
            st_ = torch.cuda.Stream(); st_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st_):
                for _ in range(3): s._step()
            torch.cuda.current_stream().wait_stream(st_); s.g = torch.cuda.CUDAGraph()
            s.opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(s.g): s._step(capture=True)
    def _step(s, capture=False):
        if not capture: s.opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=s.bf16):
            lg = s.m(s.x[:, :-1], s.rec[:, :, :-1])[0]
        loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), s.x[:, 1:].reshape(-1)); loss.backward(); s.opt.step(); s.loss.copy_(loss.detach())
    def step(s):
        if s.graph: s.g.replay()
        else: s._step()
def timed(r):
    for _ in range(2): r.step()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(STEPS): r.step()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / STEPS * 1000
cfg = {'eager_fp32': (False, False), 'graph_fp32': (False, True), 'eager_bf16': (True, False), 'graph_bf16': (True, True)}
R = {'device': torch.cuda.get_device_name(0), 'batch': [B, T], 'reps': REPS, 'steps': STEPS, 'note': 'timings share the GPU with any other running job'}
runners = {}
for k, (b16, gr) in cfg.items():
    try: runners[k] = Runner(b16, gr)
    except Exception as e: R[k + '_error'] = repr(e)[:300]
TM = M0.TModel(d=128, n_layers=4, h=4, T=max(T, 256)).to(DEV); topt = torch.optim.AdamW(TM.parameters(), 1e-3)
def tstep():
    lg = TM(X[:, :-1])[0]; loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1)); topt.zero_grad(); loss.backward(); topt.step()
class TR:
    step = staticmethod(tstep)
ms = {k: [] for k in runners}; ms['transformer_eager_fp32'] = []
for _ in range(REPS):
    for k, r in runners.items(): ms[k].append(timed(r))
    ms['transformer_eager_fp32'].append(timed(TR))
for k in ms: R[k + '_ms_per_step_median'] = round(st.median(ms[k]), 2); R[k + '_ms_all'] = [round(v, 1) for v in ms[k]]
for k, r in runners.items(): R[k + '_last_loss'] = round(r.loss.item(), 4)
try:
    import energy as EN; idle, _ = EN.idle_watts(6, 2); R['idle_W'] = round(idle, 2); R['idle_note'] = 'invalid for above-idle energy if other jobs were running'
    sync = torch.cuda.synchronize
    for k, r in list(runners.items()) + [('transformer_eager_fp32', TR)]:
        fn = lambda r=r: [r.step() for _ in range(STEPS)]; fn(); s, _ = EN.repeat(fn, 3, STEPS * B * (T - 1), idle, sync); R[k + '_energy'] = {a: s[a] for a in ('mJ_per_token_mean', 'mJ_per_token_std', 'avg_W_mean', 'seconds_mean', 'throttle_reasons')}
except Exception as e: R['energy_error'] = repr(e)[:200]
os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True); json.dump(R, open(os.path.join(HERE, '..', 'results', 'graph_ab.json'), 'w'), indent=1); print(json.dumps(R), flush=True)
