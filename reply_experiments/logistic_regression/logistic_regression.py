"""Logistic regression on raw and projected training states. Usage: python reply_experiments/logistic_regression/logistic_regression.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from lib import *
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = HERE + "/cells"


def cell_run(cell):
    name = cell["name"]; tag = name.replace(" ", "_"); fn = f"{OUT}/cell_{tag}.npz"
    S, ctask, cqid, cword = load(cell); y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; conds = ["yes_no"] + [t for t in config.HELD_OUT if all((t, p) in H for p in pids)]
    SC = {}
    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ytr = y[tr]; ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); Xte = {t: ss.transform(H[(t, p)][te]) for t in conds}
            wlr, _ = lr_own(X, ytr, q[tr])
            B, k = build_B(H, tr, p, q, cross, ctask, cqid, cword, config.FORMATS, True, k_fixed=8); Bs = std_basis(B, ss)
            wtp, _ = lr_own(X - (X @ Bs.T) @ Bs, ytr, q[tr])
            for t in conds:
                pool_put(SC, n, ("lr_raw", "-", t, p), te, Xte[t] @ wlr)
                pool_put(SC, n, ("lr_trainproj", "k8", t, p), te, (Xte[t] - (Xte[t] @ Bs.T) @ Bs) @ wtp)
    np.savez_compressed(fn, y=y, q=q, keys=np.array(["|".join(k) for k in SC]), S=np.array(list(SC.values()), dtype=np.float64)); print(name, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cs = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]; assert len(cs) == 14
    with Pool(4, maxtasksperchild=1) as P: P.map(cell_run, cs, chunksize=1)
