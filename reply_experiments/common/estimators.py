"""Estimator variants: principal-angle rank, per-set centering, question-disjoint split. Usage: python reply_experiments/common/estimators.py one "<cell name>" | all"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import *
from multiprocessing import Pool
from sklearn.utils.extmath import randomized_svd
OUT = os.path.join(ROOT, "reply_experiments", "estimator_ablations", "cells"); KMAX = est.KMAX; TH = 0.95

ARMS = {
    "submitted": ("aligned_prefix", "global", "indep", 1),
    "z1":        ("pa_largest",     "global", "indep", 1),
    "z2":        ("aligned_prefix", "block",  "indep", 1),
    "z3":        ("aligned_prefix", "global", "coord", 1),
    "all3":      ("pa_largest",     "block",  "coord", 1),
}


def svd_top(M, k):
    _, _, Vt = randomized_svd(M, n_components=min(k, min(M.shape) - 1), n_iter=4, random_state=0)
    return Vt


def stack(blocks, ids, center):
    if center == "global":
        D = np.concatenate(blocks); return D - D.mean(0)
    out = []
    for b, s in zip(blocks, ids):
        if center == "block" and s is not None:
            c = b.copy()
            for u in np.unique(s):
                m = s == u; c[m] -= b[m].mean(0)
            out.append(c)
        else:
            out.append(b - b.mean(0))
    return np.concatenate(out)


def pa_cos(V1, V2, j):
    return np.linalg.svd(V1[:j] @ V2[:j].T, compute_uv=False)


def rank_from(V1, V2, rule):
    K = min(len(V1), len(V2))
    if rule == "aligned_prefix":
        k = 0
        for c in np.abs((V1 * V2).sum(1)):
            if c >= TH: k += 1
            else: break
        return max(k, 1)
    ok = [pa_cos(V1, V2, j).min() >= TH for j in range(1, K + 1)]
    return max([j for j, o in zip(range(1, K + 1), ok) if o] + [1])


def masks_indep(grp, seed):
    rng = np.random.default_rng(seed); out = []
    for g in grp:
        ug = np.unique(g); half = set(rng.permutation(ug)[: len(ug) // 2].tolist()); out.append(np.array([x in half for x in g], dtype=bool))
    return out


def masks_coord(qtr, nfmt, cross_gid, cross_task, seed):
    rng = np.random.default_rng(seed); uq = np.unique(qtr); half = set(rng.permutation(uq)[: len(uq) // 2].tolist())
    m = np.array([x in half for x in qtr], dtype=bool); out = [m] * nfmt
    if cross_gid is not None:
        hc = set()
        for t in np.unique(cross_task):
            ug = np.unique(cross_gid[cross_task == t]); hc |= set(rng.permutation(ug)[: len(ug) // 2].tolist())
        out.append(np.array([x in hc for x in cross_gid], dtype=bool))
    return out


def cross_provenance(model):
    tk_, qd, wd = [], [], []
    for ti, tk in enumerate(config.CROSS_TASKS):
        f1 = storage.path("states", model=model, task=tk, condition="negated"); f0 = storage.path("states", model=model, task=tk, condition="yes_no")
        if os.path.exists(f1) and os.path.exists(f0):
            Za = np.load(f1, allow_pickle=True); Zb = np.load(f0, allow_pickle=True)
            assert np.array_equal(Za["qid"], Zb["qid"]), f"cross qid misaligned {model} {tk}"
            for p in [str(x) for x in Za["prompt_ids"]]:
                if f"G_{p}" in Zb:
                    n = len(Za[f"G_{p}"]); assert n == len(Za["qid"])
                    tk_.append(np.full(n, ti)); qd.append(Za["qid"].astype(np.int64)); wd.append(np.full(n, int(p[1:])))
    return np.concatenate(tk_), np.concatenate(qd), np.concatenate(wd)


def cell_run(cell):
    name = cell["name"]; tag = name.replace(" ", "_"); fn = os.path.join(OUT, f"cell_{tag}.npz")
    os.makedirs(OUT, exist_ok=True)
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    ctask, cqid, cword = cross_provenance(cell["model"]); assert len(ctask) == len(cross)
    pids = [p for p in config.FIVE if ("yes_no", p) in H]
    conds = [t for t in config.HELD_OUT if all((t, p) in H for p in pids)]; SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for tr, te in fld:
        for p in pids:
            ytr = y[tr]; Hy = H[("yes_no", p)]; ss = StandardScaler().fit(Hy[tr]); X = ss.transform(Hy[tr]); wcm = est.class_mean(X, ytr)
            Xte = {t: ss.transform(H[(t, p)][te]) for t in conds}
            pairs = [H[(g, p)][tr] - H[(f, p)][tr] for i, f in enumerate(config.FORMATS) for g in config.FORMATS[i + 1:] if (f, p) in H and (g, p) in H]
            nf = len(pairs); qtr = q[tr]; halves = {}; fulls = {}
            for arm, (rule, center, split, sub) in ARMS.items():
                idx = np.random.default_rng(sub).choice(len(cross), min(len(cross), 4 * len(tr)), replace=False)
                cr = cross[idx]; gid = ctask[idx] * 10**8 + cqid[idx]; sid = ctask[idx] * 100 + cword[idx]
                blocks = pairs + [cr]; ids = [None] * nf + [sid]; kh = (center, split, sub)
                if kh not in halves:
                    grp = [qtr] * nf + [np.arange(len(cr))]
                    ms = masks_indep(grp, 0) if split == "indep" else masks_coord(qtr, nf, gid, ctask[idx], 0)
                    V1 = svd_top(stack([b[m] for b, m in zip(blocks, ms)], [None if s is None else s[m] for s, m in zip(ids, ms)], center), KMAX)
                    V2 = svd_top(stack([b[~m] for b, m in zip(blocks, ms)], [None if s is None else s[~m] for s, m in zip(ids, ms)], center), KMAX)
                    halves[kh] = (V1, V2)
                V1, V2 = halves[kh]; k = rank_from(V1, V2, rule); kf = (center, sub, k)
                if kf not in fulls:
                    fulls[kf] = svd_top(stack(blocks, ids, center), k)
                Bs, _ = np.linalg.qr((fulls[kf] / ss.scale_).T); Bs = Bs.T; w = wcm - Bs.T @ (Bs @ wcm)
                for t in conds:
                    put((arm, t, p), te, Xte[t] @ w)
    np.savez_compressed(fn, y=y, q=q, keys=np.array(["|".join(k) for k in SC]), S=np.array(list(SC.values()), dtype=np.float32))
    print(name, flush=True)
    return name


if __name__ == "__main__":
    if sys.argv[1] == "one":
        cell_run([c for c in config.cells() if c["name"] == sys.argv[2]][0])
    elif sys.argv[1] == "all":
        cs = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]
        with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4, maxtasksperchild=1) as pool:
            pool.map(cell_run, cs, chunksize=1)
