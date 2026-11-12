"""Change in AUROC from projecting the training states. Usage: python reply_experiments/logistic_regression/aggregate.py"""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from env import *
HERE = os.path.dirname(os.path.abspath(__file__))


def boot(v, seed=0):
    rng = np.random.default_rng(seed); m = v[rng.integers(0, len(v), (10000, len(v)))].mean(1); return np.quantile(m, .025), np.quantile(m, .975)


rows = []
for f in sorted(glob.glob(HERE + "/cells/cell_*.npz")):
    Z = np.load(f); y = Z["y"]; K = [str(k).split("|") for k in Z["keys"]]; S = Z["S"]; name = os.path.basename(f)[5:-4].replace("_", " "); d = {}
    for i, k in enumerate(K): d.setdefault(tuple(k[:3]), []).append(est.auroc(y, S[i]))
    rows += [dict(cell=name, arm=k[0], spec=k[1], condition=k[2], auroc=float(np.mean(v))) for k, v in d.items()]
Zr = pd.DataFrame(rows); g = lambda a: Zr[Zr.arm == a].set_index(["cell", "condition"]).auroc; O = []
D = (g("lr_trainproj") - g("lr_raw")).dropna().reset_index()
for c, h in D.groupby("condition"):
    v = h.auroc.values; lo, hi = boot(v); O.append(dict(delta="lr_trainproj-lr_raw", condition=c, cells=len(v), mean=v.mean(), lo=lo, hi=hi))
ORDER = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct", "yes_no"]
O = pd.DataFrame(O).set_index("condition").loc[ORDER].reset_index()[["delta", "condition", "cells", "mean", "lo", "hi"]]; O.to_csv(HERE + "/deltas.csv", index=False); print(O.round(4).to_string(index=False))
