"""Expression subspace with the rank fixed at 4 to 12. Usage: python reply_experiments/rank_sweep/sweep.py ["<cell name>"]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from lib import *
from multiprocessing import Pool
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells"); F = config.FORMATS; KS = [4, 5, 6, 7, 8, 9, 10, 11, 12]


def cell_run(cell):
    name = cell["name"]; tag = name.replace(" ", "_"); fn = os.path.join(OUT, f"cell_{tag}.npz")
    S, ctask, cqid, cword = load(cell); y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; conds = [t for t in config.HELD_OUT if all((t, p) in H for p in pids)]
    SC = {}
    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ytr = y[tr]; ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); Xte = {t: ss.transform(H[(t, p)][te]) for t in conds}; wcm = est.class_mean(X, ytr)
            for kf in KS:
                B, k = build_B(H, tr, p, q, cross, ctask, cqid, cword, F, True, k_fixed=kf); Bs = std_basis(B, ss); w = wcm - Bs.T @ (Bs @ wcm)
                for t in conds: pool_put(SC, n, (f"k{kf}", t, p), te, Xte[t] @ w)
    ks = list(SC); np.savez_compressed(fn, y=y, q=q, keys=np.array(["|".join(k) for k in ks]), S=np.array([SC[k] for k in ks]))
    print(name, flush=True); return name


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cs = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]
    if len(sys.argv) > 1: cs = [c for c in cs if c["name"] == sys.argv[1]]
    with Pool(min(4, len(cs)), maxtasksperchild=1) as pool: pool.map(cell_run, cs, chunksize=1)
