"""One-hidden-layer MLP readout on raw and projected states. Usage: python reply_experiments/mlp/mlp.py <workers>"""
import os, sys, time
CUDA = os.environ.get("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from lib import *
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
from multiprocessing import get_context
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "cells")
WIDTHS = [32, 64, 128]; WDS = [1e-4, 1e-3, 1e-2]; SEEDS = [0, 1, 2]; LR = 1e-3; EMAX = 300; PAT = 30
GRID = [(h, wd) for h in WIDTHS for wd in WDS]


def init(d, seeds, hmax=128):
    P = []
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        b1 = 1 / np.sqrt(d); b2 = 1 / np.sqrt(hmax)
        P.append((torch.empty(d, hmax).uniform_(-b1, b1, generator=g), torch.empty(hmax).uniform_(-b1, b1, generator=g),
                  torch.empty(hmax).uniform_(-b2, b2, generator=g), torch.empty(1).uniform_(-b2, b2, generator=g)))
    return P


def train(X, y, trm, vam, nets, P, dev, epochs=None):
    N = len(nets); d = X.shape[1]; H = 128
    W1 = torch.stack([P[s][0] for (_, h, _, s) in nets]).to(dev); b1 = torch.stack([P[s][1] for (_, h, _, s) in nets]).to(dev)
    w2 = torch.stack([P[s][2] for (_, h, _, s) in nets]).to(dev); b2 = torch.stack([P[s][3] for (_, h, _, s) in nets]).to(dev)
    hm = torch.stack([(torch.arange(H) < h).float() for (_, h, _, _) in nets]).to(dev)
    W1 = (W1 * hm[:, None, :]).requires_grad_(); b1 = (b1 * hm).requires_grad_(); w2 = (w2 * hm).requires_grad_(); b2 = b2.clone().requires_grad_()
    wd = torch.tensor([w for (_, _, w, _) in nets], device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    TR = torch.stack([torch.as_tensor(trm[sp]) for (sp, _, _, _) in nets]).float().to(dev); VA = None if vam is None else torch.stack([torch.as_tensor(vam[sp]) for (sp, _, _, _) in nets]).float().to(dev)
    opt = torch.optim.Adam([W1, b1, w2, b2], lr=LR); bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    fwd = lambda: torch.einsum("knh,kh->kn", torch.relu(torch.einsum("nd,kdh->knh", X, W1) + b1[:, None, :]) * hm[:, None, :], w2) + b2
    E = epochs if epochs is not None else EMAX; best = torch.full((N,), float("inf"), device=dev); bep = torch.zeros(N, dtype=torch.long, device=dev); bsc = None; since = torch.zeros(N, device=dev)
    for ep in range(1, E + 1):
        opt.zero_grad(); L = bce(fwd(), yt.expand(N, -1)); loss = ((L * TR).sum(1) / TR.sum(1)).sum(); loss.backward(); opt.step()
        with torch.no_grad():
            W1.mul_(1 - LR * wd[:, None, None]); w2.mul_(1 - LR * wd[:, None]); W1.mul_(hm[:, None, :]); b1.mul_(hm); w2.mul_(hm)
            if VA is not None:
                out = fwd(); vl = (bce(out, yt.expand(N, -1)) * VA).sum(1) / VA.sum(1); imp = vl < best - 1e-6
                best = torch.where(imp, vl, best); bep = torch.where(imp, torch.full_like(bep, ep), bep); since = torch.where(imp, torch.zeros_like(since), since + 1)
                bsc = out.detach().clone() if bsc is None else torch.where(imp[:, None], out, bsc)
                if bool((since >= PAT).all()): break
    if VA is None: return (W1.detach(), b1.detach(), w2.detach(), b2.detach(), hm)
    va = []
    for i, (sp, _, _, _) in enumerate(nets):
        m = vam[sp]; va.append(est.rank_auroc(y[m], bsc[i].cpu().numpy()[m]))
    return bep.cpu().numpy(), np.array(va)


def score(params, X):
    W1, b1, w2, b2, hm = params
    with torch.no_grad():
        return (torch.einsum("knh,kh->kn", torch.relu(torch.einsum("nd,kdh->knh", X, W1) + b1[:, None, :]) * hm[:, None, :], w2) + b2).cpu().numpy()


def fit_select(Xtr, ytr, qtr, P, dev):
    sp = [(a, b) for a, b in est._inner_splits(qtr) if len(np.unique(ytr[a])) == 2 and len(np.unique(ytr[b])) == 2]
    if not sp: return None, None
    X = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    nets = [(i, h, wd, s) for i in range(len(sp)) for (h, wd) in GRID for s in range(len(SEEDS))]
    ep, va = train(X, ytr, [a for a, _ in sp], [b for _, b in sp], nets, P, dev)
    tab = {}
    for (i, h, wd, s), e, v in zip(nets, ep, va): tab.setdefault((h, wd), []).append((e, v))
    h, wd = max(GRID, key=lambda g: (np.nanmean([v for _, v in tab[g]]), -g[0], g[1]))
    E = max(1, int(round(np.mean([e for e, _ in tab[(h, wd)]]))))
    fin = train(X, ytr, [np.ones(len(ytr), bool)], None, [(0, h, wd, s) for s in range(len(SEEDS))], P, dev, epochs=E)
    return dict(h=h, wd=wd, E=E, inner_auroc=float(np.nanmean([v for _, v in tab[(h, wd)]])), n_splits=len(sp)), fin


def cell_run(cell):
    name = cell["name"]; tag = name.replace(" ", "_"); fn = f"{OUT}/cell_{tag}.npz"
    if os.path.exists(fn): print("SKIP", name, flush=True); return name
    torch.set_num_threads(2); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    S, ctask, cqid, cword = load(cell); y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; conds = ["yes_no"] + [t for t in config.HELD_OUT if all((t, p) in H for p in pids)]
    P = init(H[("yes_no", pids[0])].shape[1], SEEDS); SC = {}; CH = []
    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); Xte = {t: ss.transform(H[(t, p)][te]) for t in conds}
            B8, _ = build_B(H, tr, p, q, cross, ctask, cqid, cword, config.FORMATS, True, k_fixed=8)
            prj = {"mlp_raw": None, "mlp_proj8": std_basis(B8, ss)}
            for arm, Bs in prj.items():
                f = (lambda Z: Z) if Bs is None else (lambda Z, Bs=Bs: Z - (Z @ Bs.T) @ Bs)
                Xa = f(X); Xt = {t: torch.as_tensor(f(Xte[t]), dtype=torch.float32, device=dev) for t in conds}
                ch, fin = fit_select(Xa, y[tr], q[tr], P, dev)
                CH.append(dict(cell=name, fold=fi, wording=p, arm=arm, **(ch or dict(h=np.nan, wd=np.nan, E=np.nan, inner_auroc=np.nan, n_splits=0))))
                for t in conds:
                    sc = score(fin, Xt[t]) if fin is not None else None
                    for si in range(len(SEEDS)):
                        key = (arm, t, p, str(si))
                        if sc is None: SC.setdefault(key, np.full(n, np.nan)); continue
                        pool_put(SC, n, key, te, sc[si])
        print(f"[{name}] fold {fi} {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(CH).to_csv(f"{OUT}/chosen_{tag}.csv", index=False)
    np.savez_compressed(fn, y=y, q=q, keys=np.array(["|".join(k) for k in SC]), S=np.array(list(SC.values()), dtype=np.float32)); print("DONE", name, f"{time.time()-t0:.0f}s", flush=True); return name


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cs = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]; assert len(cs) == 14
    with get_context("spawn").Pool(int(sys.argv[1]), maxtasksperchild=1) as Pl: Pl.map(cell_run, cs, chunksize=1)
