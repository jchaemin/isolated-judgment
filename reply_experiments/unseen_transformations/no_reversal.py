"""Readouts with the subspace built only from non-reversing formats. Usage: python reply_experiments/unseen_transformations/no_reversal.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from lib import *
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = HERE + "/cells"; F = config.FORMATS; NON_REVERSING = [f for f in F if f != "b_a"]
TARGETS = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct"]


def cell_run(cell):
    name = cell["name"]; tag = name.replace(" ", "_"); fn = f"{OUT}/cell_{tag}.npz"
    S, ctask, cqid, cword = load(cell); y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; conds = [t for t in TARGETS if all((t, p) in H for p in pids)]
    SC = {}
    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); Xte = {t: ss.transform(H[(t, p)][te]) for t in conds}; wcm = est.class_mean(X, y[tr])
            for t in conds: pool_put(SC, n, ("readout", "-", t, p), te, Xte[t] @ wcm)
            B, k = build_B(H, tr, p, q, cross, ctask, cqid, cword, NON_REVERSING, False, k_fixed=8); Bs = std_basis(B, ss); w = wcm - Bs.T @ (Bs @ wcm)
            for t in conds: pool_put(SC, n, ("no_reversal", "k8", t, p), te, Xte[t] @ w)
    np.savez_compressed(fn, y=y, q=q, keys=np.array(["|".join(k) for k in SC]), S=np.array(list(SC.values()), dtype=np.float64)); print(name, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cs = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]; assert len(cs) == 14
    with Pool(4, maxtasksperchild=1) as P: P.map(cell_run, cs, chunksize=1)
