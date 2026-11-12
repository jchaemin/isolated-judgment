"""Change in AUROC for each estimator variant. Usage: python reply_experiments/estimator_ablations/aggregate.py"""
import glob, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ARMS = ["z1", "z2", "z3", "all3"]
HO = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct"]


def auroc(y, s):
    o = np.argsort(s, kind="mergesort"); py = (y == 1)[o]; negprev = np.r_[0, np.cumsum(~py)[:-1]]
    return float((py * negprev).sum() / (py.sum() * (~py).sum()))


cells = {}
for f in sorted(glob.glob(os.path.join(HERE, "cells", "cell_*.npz"))):
    Z = np.load(f); K = [tuple(str(k).split("|")) for k in Z["keys"]]
    cells[os.path.basename(f)[5:-4]] = dict(y=Z["y"], idx={k: i for i, k in enumerate(K)}, S=Z["S"].astype(np.float64))


def cell_auroc(c, arm, t):
    return float(np.mean([auroc(c["y"], c["S"][c["idx"][k]]) for k in c["idx"] if k[0] == arm and k[1] == t]))


rows = []
for t in HO:
    for arm in ARMS:
        d = np.array([cell_auroc(c, arm, t) - cell_auroc(c, "submitted", t) for c in cells.values()])
        r = dict(condition=t, what="delta_vs_submitted", arm=arm, n_cells=len(d), mean=d.mean())
        if arm == "all3":
            m = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
            r.update(cell_lo=np.percentile(m, 2.5), cell_hi=np.percentile(m, 97.5))
        rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "auroc_deltas.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
