"""State tracking test: running parity of a bit stream (needs a neuron that can FLIP).
Train on length 64, test on 64 / 256 / 1024. Models: L (leak in [-1,1]), L-positive (leak in [0,1], like Mamba), transformer.
Usage: python parity_L0.py MODEL SEED"""
import sys, json, torch, torch.nn.functional as F, l_model as M
torch.set_num_threads(1); name, seed = sys.argv[1], int(sys.argv[2]); torch.manual_seed(seed)
m = {'L': lambda: M.LModel(d=64, n_layers=2, hippo=False), 'L-positive': lambda: M.LModel(d=64, n_layers=2, neg=False, hippo=False),
     'T': lambda: M.TModel(d=64, n_layers=2, T=1024), 'L-nospike': lambda: M.LModel(d=64, n_layers=2, hippo=False, spiking=False), 'L-homeo': lambda: M.LModel(d=64, n_layers=2, hippo=False, homeo=True)}[name]()
def data(n, B=32): x = torch.randint(0, 2, (B, n)); return x, x.cumsum(1) % 2
opt = torch.optim.AdamW(m.parameters(), 3e-3); 
for it in range(1500):
    x, y = data(64); lg, _ = m(x); loss = F.cross_entropy(lg[..., :2].reshape(-1, 2), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
m.eval(); R = {'model': name, 'seed': seed, 'params': M.nparams(m)}
with torch.no_grad():
    for n in [64, 256, 1024]:
        x, y = data(n, 16); lg, _ = m(x); R[f'acc_len{n}'] = round((lg[..., :2].argmax(-1) == y).float().mean().item(), 4)
print(json.dumps(R)); import os; json.dump(R, open(os.path.join(os.environ.get('LOUT', '.'), f'results_parity_{name}_s{seed}.json'), 'w'))
