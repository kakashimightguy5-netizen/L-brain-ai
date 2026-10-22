"""Delayed-copy diagnostic for the MatMul-free spiking cell (follows Gemini's advice: normalise ternary pre-activations, higher LR on thresholds). MEASURED, CPU.
Usage: python delaycopy_test.py CONFIG SEED [STEPS]   CONFIG: old | norm | norm_thr5 | norm_thr5_nofast"""
import sys, os, torch, torch.nn.functional as F_
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from matmulfree_cell import MatMulFreeSpikingCell
cfg, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000; torch.set_num_threads(1); torch.manual_seed(seed)
norm = cfg.startswith('norm'); thr_lr = 5.0 if 'thr5' in cfg else 1.0; fast = 'nofast' not in cfg; D = int(os.environ.get("DELAY", 3))
c = MatMulFreeSpikingCell(8, int(os.environ.get('HID', 48)), 8, 16, 16, norm=norm, taps=int(os.environ.get('TAPS', 1)))
th = [p for n, p in c.named_parameters() if n.startswith(('theta', 'th_'))]; w = [p for n, p in c.named_parameters() if not n.startswith(('theta', 'th_'))]
LR = float(os.environ.get('LR', 1e-2)); opt = torch.optim.Adam([{'params': w, 'lr': LR}, {'params': th, 'lr': LR * thr_lr}]); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if os.environ.get('COSINE', '0') == '1' else None   # LR / COSINE flags: default off
def batch(B):
    xb = (torch.rand(B, 16, 8) > 0.5).float(); return xb, (xb if D == 0 else torch.cat([torch.zeros(B, D, 8), xb[:, :-D]], 1))
for it in range(steps):
    xb, tgt = batch(32); o, pre, _, st = c(xb, fast=fast); loss = F_.binary_cross_entropy_with_logits(pre[:, D:], tgt[:, D:]); opt.zero_grad(); loss.backward(); opt.step(); sch and sch.step()
    if (it + 1) % (steps // 4) == 0:
        with torch.no_grad(): xe, te = batch(256); oe, _, _, se = c(xe, fast=fast); acc = (oe[:, D:] == te[:, D:]).float().mean().item()
        print(cfg, 'seed', seed, 'step', it + 1, 'loss', round(loss.item(), 4), 'bit_acc', round(acc, 4), 'hid_rate', round(se['hid_rate'], 3), flush=True)
