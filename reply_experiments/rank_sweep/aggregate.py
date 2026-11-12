"""AUROC by subspace rank. Usage: python reply_experiments/rank_sweep/aggregate.py"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); HO = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct"]; KS = [f"k{k}" for k in range(4, 13)]


def auc(y, s):
    ok = np.isfinite(s); y, s = y[ok], s[ok]; r = rankdata(s); p = y == 1; n1 = p.sum(); return (r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


rows = []
for f in sorted(glob.glob(os.path.join(HERE, "cells", "cell_*.npz"))):
    Z = np.load(f); y = Z["y"]; K = [tuple(str(k).split("|")) for k in Z["keys"]]; S = Z["S"].astype(float); cell = os.path.basename(f)[5:-4]
    for arm in KS:
        for c in HO:
            rows.append(dict(cell=cell, arm=arm, cond=c, auroc=np.mean([auc(y, S[i]) for i, k in enumerate(K) if k[0] == arm and k[1] == c])))
D = pd.DataFrame(rows)
curve = D.groupby(["arm", "cond"]).auroc.mean().unstack("cond")[HO]
curve = curve[HO].mean(1).round(4).loc[KS].rename("HO4").to_frame()
curve.to_csv(os.path.join(HERE, "auroc_by_rank.csv")); print(curve.to_string())
H = pd.read_csv(os.path.join(HERE, "..", "..", "results", "subspace", "heldout.csv"))
rule = H[H.method == "isolated"].groupby("condition").auroc.mean()
k8 = D[D.arm == "k8"].groupby("cond").auroc.mean()
cmp = pd.DataFrame(dict(condition=HO, k8=k8.loc[HO].values, rule=rule.loc[HO].values)); cmp["difference"] = cmp.k8 - cmp.rule
cmp.to_csv(os.path.join(HERE, "k8_vs_rule.csv"), index=False); print(cmp.round(4).to_string(index=False))
