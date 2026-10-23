import sys, time, json, torch, torch.nn.functional as F
sys.path[:0] = ['/home/claude/lp']; from lpure import LPure; from tasks import kv_batch
torch.set_num_threads(2); N = int(sys.argv[1]); steps = int(sys.argv[2]); variant = sys.argv[3]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
torch.manual_seed(seed); V = 64 + 10 + 1
kw = dict(mem=True, dense=False)
if variant == 'nomem': kw['mem'] = False
if variant == 'dense': kw['dense'] = True
m = LPure(V, d=64, n_layers=2, K=16, **kw); opt = torch.optim.AdamW(m.parameters(), 3e-3); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.1)
t0 = time.time(); ge = torch.Generator().manual_seed(999)
def evaluate():
    with torch.no_grad():
        tok, y = kv_batch(256, N, g=ge); return (m(tok)[:, -1].argmax(-1) == y).float().mean().item()
for it in range(steps):
    tok, y = kv_batch(32, N); lg = m(tok)[:, -1]; loss = F.cross_entropy(lg, y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    if (it + 1) % max(1, steps // 5) == 0: print(variant, 'N', N, 'step', it + 1, 'loss', round(loss.item(), 3), 'acc', round(evaluate(), 3), f'{time.time() - t0:.0f}s', flush=True)
acc = evaluate(); json.dump({'variant': variant, 'N': N, 'steps': steps, 'seed': seed, 'acc': acc, 'adds_per_token': None if kw['dense'] else m.adds_per_token(), 'params': sum(p.numel() for p in m.parameters())}, open(f'/home/claude/lp/kv_{variant}_N{N}_s{seed}.json', 'w')); print('FINAL', variant, N, acc)
