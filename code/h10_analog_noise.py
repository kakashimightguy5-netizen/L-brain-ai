"""H10 - analog in-memory compute tolerance: how much weight / read noise can L-Edge take?
MEASURED model quality under SIMULATED noise (the noise model is an assumption, not a measured device):
  programming variation: every weight w -> w (1 + s e), e ~ N(0,1), fixed for the whole evaluation (one 'chip'); 5 chips per s
  read noise: every nn.Linear output y -> y + s rms(y) e, fresh every byte
Plus a 1-seed noise-aware fine-tune (500 steps, multiplicative weight noise s=0.05 resampled every step) and a matched
no-noise fine-tune control (same steps/LR) so the effect of extra training is separated from the effect of noise training.
EVAL = second half of the held-out file (CAL is not needed: nothing is fitted). Writes LOUT/h10_analog_noise.json
Usage: python h10_analog_noise.py CKPT"""
import os, sys, time, json, math, copy, torch, torch.nn as nn, torch.nn.functional as F
OUT = os.environ.get('LOUT', '.'); wf = os.environ.get('WAITFOR', '')
while wf and not all(os.path.exists(os.path.join(OUT, w)) for w in wf.split(',')): time.sleep(30)
os.environ.setdefault('SPIKEK', '8')
import l5_model as M5, fastmem
ck = sys.argv[1]; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; DATA = os.environ.get('LDATA', '/home/claude/data')
QUICK = os.environ.get('QUICK', '0') == '1'
base = M5.L5Model(d=128, n_layers=4, heads=4).to(dev).eval(); base.load_state_dict(torch.load(ck if os.path.exists(ck) else os.path.join(OUT, ck), map_location=dev))
va = torch.tensor(list(open(os.path.join(DATA, 'py_big_val.txt'), 'rb').read()), dtype=torch.long); half = len(va) // 2
NW, W = (4, 512) if QUICK else (32, 1024)
X = torch.stack([va[half + i * W: half + i * W + W + 1] for i in range(NW)]).to(dev)
@torch.no_grad()
def evaluate(m, read_s=0.0):
    hooks = []
    if read_s > 0:
        def hk(mod, inp, out): return out + read_s * out.pow(2).mean(-1, keepdim=True).sqrt() * torch.randn_like(out)
        hooks = [mod.register_forward_hook(hk) for mod in m.modules() if isinstance(mod, nn.Linear)]
    tot = 0.0
    for i in range(0, NW, 8):
        x = X[i:i + 8]; inp, y = x[:, :-1], x[:, 1:]; lp = m(inp, fastmem.hippo_batch(inp))[0].log_softmax(-1)
        tot += -lp.gather(-1, y[..., None]).sum().item()
    for h in hooks: h.remove()
    return tot / (NW * W) / math.log(2)
def chip(m, s, seed):
    c = copy.deepcopy(m); g = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        for mod in c.modules():
            if isinstance(mod, nn.Linear): mod.weight.mul_(1 + s * torch.randn(mod.weight.shape, generator=g, device=dev))
    return c
def sweep(m, tag):
    r = {'clean': round(evaluate(m), 4)}
    for s in (0.01, 0.02, 0.05, 0.1, 0.2):
        v = [evaluate(chip(m, s, k)) for k in range(5)]; r[f'weight var {s}'] = {'mean': round(sum(v) / 5, 4), 'std': round(torch.tensor(v).std().item(), 4)}
    for s in (0.01, 0.02, 0.05, 0.1):
        torch.manual_seed(0); v = [evaluate(m, s) for _ in range(3)]; r[f'read noise {s}'] = {'mean': round(sum(v) / 3, 4), 'std': round(torch.tensor(v).std().item(), 4)}
    print(tag, r, flush=True); return r
R = {'checkpoint': ck, 'eval_bytes': NW * W, 'noise_model': 'ASSUMED (see docstring)'}
R['pretrained'] = sweep(base, 'pretrained')
tr = torch.tensor(list(open(os.path.join(DATA, 'py_big_train.txt'), 'rb').read()[:20_000_000]), dtype=torch.long)
def finetune(noise_s, steps=150 if QUICK else 500, seed=0):
    m = copy.deepcopy(base).train(); torch.manual_seed(seed); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.0)
    lins = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    for it in range(steps):
        ix = torch.randint(0, len(tr) - 257, (16,)); x = torch.stack([tr[i:i + 257] for i in ix]).to(dev); inp, y = x[:, :-1], x[:, 1:]
        saved = []
        if noise_s > 0:
            with torch.no_grad():
                for mod in lins: saved.append(mod.weight.detach().clone()); mod.weight.mul_(1 + noise_s * torch.randn_like(mod.weight))
        loss = F.cross_entropy(m(inp, fastmem.hippo_batch(inp))[0].reshape(-1, 256), y.reshape(-1)) + 0.03 * m.rate_cost()
        opt.zero_grad(); loss.backward()
        if noise_s > 0:
            with torch.no_grad():
                for mod, w in zip(lins, saved): mod.weight.copy_(w)            # gradient computed at the noisy point, applied to the clean weights
        opt.step()
    return m.eval()
R['finetune_no_noise_control'] = sweep(finetune(0.0), 'ft clean')
R['finetune_noise_aware_s0.05'] = sweep(finetune(0.05), 'ft noise')
json.dump(R, open(os.path.join(OUT, 'h10_analog_noise.json'), 'w'), indent=1); print('done')
