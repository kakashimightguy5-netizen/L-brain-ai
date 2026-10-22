"""Energy-proportional serving runtime for L (Part 3 of the efficiency-stack prompt).
Runs an L-5 model WITH the post-hoc effort gate (train_L5E.py: L4E2 'skim' model, exit read-outs + halt heads) byte by byte,
and turns the think-less + metacognition mechanisms into ONE per-token policy:
  instinct   : the previous byte was predicted with confidence >= tau_instinct (M1 uncertainty signal)  -> exit after layer 1
               (optional recall-bypass: also when the order-12 hippocampus recall and the trajectory replay agree)
  deliberate : otherwise run layers until a halt head fires (halt prob >= tau_halt) or the energy cap is reached
  flag       : after the run, if top-1 confidence < tau_flag the byte is flagged as 'unsure' (logged; for an application
               this is where 'ask / abstain / verify' hooks in). Flag does not add compute.
Layers after an exit are SKIMMED: their membrane state and fast-weight memory are still updated from the current
representation (memory stays intact, same semantics as L4E2Model.forward(thr=...)), but out/f1/f2 and the residual update are skipped.
User knobs: effort in {'low','balanced','high','full'} -> thresholds fitted on the CALIBRATION split (fit_policy.py),
max_energy_per_token -> hard cap on modelled work per byte as a FRACTION of the full-depth work (max_work_frac), enforced by
forcing an exit as soon as running one more full layer could exceed it (the floor is exit-after-layer-1 + skims = (11+6(L-1))/(11L)).
Skill memory (M2) is NOT included: M2 measured ~2 % work saved with a small quality cost, below the noise of this runtime.
Default (effort='full', no cap) == the reference Stepper output (tests/test_serve_runtime.py).
Work accounting per layer (MACs, D=d, F=2d):  full = qkv 3D^2 + o D^2 + inp 2D^2 + out D^2 + f1 2D^2 + f2 2D^2 = 11 D^2
                                               skim = qkv 3D^2 + o D^2 + inp 2D^2                              =  6 D^2
(the older L4E2Model.SKIM_COST = 5/11 under-counted the skim: the memory READ and its o projection are needed because the
membrane input of a skimmed layer includes it; corrected here to 6/11.)"""
import math, torch, torch.nn.functional as F
import l_model as M0, l6_step

EFFORT = {'full': dict(tau_instinct=2.0, tau_halt=2.0, recall_bypass=False)}       # 'low'/'balanced'/'high' filled by fit_policy

class EffortStepper(l6_step.Stepper):
    def __init__(s, m, B=1, effort='full', policy=None, max_work_frac=None, tau_flag=0.5):
        super().__init__(m, B); s.pol = dict((policy or EFFORT)[effort]); s.cap = max_work_frac; s.tau_flag = tau_flag
        s.prev_conf = torch.zeros(B, device=s.dev); s.log = []
        D = m.emb.weight.shape[1]; s.D = D; s.nL = len(m.layers)
        s.FULL, s.SKIM = 11 * D * D, 6 * D * D
    @torch.no_grad()
    def neural(s, ints):
        m = s.m; x = ints[0]; rec = ints[1:4]; tr = ints[4]; hit = ints[5]; B = s.B
        he = m.hemb(hit); g = torch.sigmoid(m.tgate(he))
        z = m.emb(x) + sum(e(r) for e, r in zip(m.remb, rec)) + g * m.temb(tr) + he
        active = torch.ones(B, dtype=torch.bool, device=s.dev)
        instinct = s.prev_conf >= s.pol['tau_instinct']
        if s.pol.get('recall_bypass'): instinct = instinct | ((rec[2] < 256) & (rec[2] == tr))
        out = torch.zeros(B, 256, device=s.dev); work = torch.zeros(B, device=s.dev); layers_full = torch.zeros(B, device=s.dev)
        fired = torch.zeros(B, device=s.dev); halt_cost = torch.zeros(B, device=s.dev)
        for li, (fw, l) in enumerate(zip(m.fw, m.layers)):
            # ---- fast-weight memory (always: write keeps memory intact; read feeds the membrane input) ----
            xn = fw.n(z); zz = fw.qkv(xn)
            if fw.conv is not None:
                buf = torch.cat([s.convbuf[li], zz[:, None]], 1); s.convbuf[li].copy_(buf[:, 1:]); zz = (buf * fw.conv.weight[:, 0, :].T[None]).sum(1)
            zz = F.silu(zz); q, k, v = zz.view(B, 3, fw.h, fw.dk).unbind(1); k = F.normalize(k, dim=-1); q = F.normalize(q, dim=-1)
            a = F.logsigmoid(fw.ga(xn)).exp()[..., None, None]; bb = torch.sigmoid(fw.gb(xn))[..., None]
            Sa = a * s.S[li]; s.S[li].copy_(Sa + (bb * (v - (Sa @ k[..., None])[..., 0]))[..., None] * k[..., None, :])
            o = (s.S[li] @ q[..., None])[..., 0]; zm = z + fw.o(o.reshape(B, -1))
            # ---- membrane (always updated) ----
            u, zg = l.inp(l.n1(zm)).chunk(2, -1); aa = l.gate(zg); s.h[li].copy_(aa * s.h[li] + (1 - aa.abs()) * u); h = s.h[li]
            if active.any():
                th = l.theta.abs().clamp(min=0.05); gain = h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() if l.homeo else 1.0
                cnt = M0.spikes(h / (gain * th), -l.K, l.K); zf = zm + l.out(cnt * th)
                gg = l.f1(l.n2(zf)); th2 = l.theta2.abs().clamp(min=0.05); cnt2 = M0.spikes(F.relu(gg) / th2, 0, l.K); zf = zf + l.f2(cnt2 * th2)
                fired = fired + active * ((cnt != 0).sum(-1) + (cnt2 != 0).sum(-1))
                z = torch.where(active[:, None], zf, z)
            work = work + torch.where(active, s.FULL, s.SKIM); layers_full = layers_full + active
            if li < s.nL - 1 and (s.pol['tau_halt'] <= 1 or s.cap is not None or bool(instinct.any())):
                hz = m.enorm[li](z); stop = active & instinct & (li == 0)
                if s.pol['tau_halt'] <= 1:
                    p = torch.sigmoid(m.halt[li](hz)[..., 0]); halt_cost = halt_cost + active * s.D; stop = stop | (active & (p >= s.pol['tau_halt']))
                if s.cap is not None: stop = stop | (active & (work + s.FULL + (s.nL - 2 - li) * s.SKIM > s.cap * s.nL * s.FULL))
                if stop.any():
                    lg = m.readout(hz, m.etrust[li], rec, tr, g); out = torch.where(stop[:, None], lg, out); active = active & ~stop
        if active.any():
            lg = m.readout(m.norm(z), m.trust, rec, tr, g); out = torch.where(active[:, None], lg, out)
        lp = out.log_softmax(-1); conf = lp.max(-1).values.exp(); s.prev_conf = conf
        s.log.append(dict(layers_full=layers_full.cpu(), work=(work + halt_cost + s.D * 256).cpu(), fired=fired.cpu(),
                          instinct=instinct.cpu(), flag=(conf < s.tau_flag).cpu()))
        return lp
    def full_work(s): return s.nL * s.FULL + s.D * 256
