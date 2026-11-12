"""Change in AUROC from projecting the training states, ridge or linear SVM. Usage: python reply_experiments/ridge_svm/aggregate.py {ridge|svm}"""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from env import *
HERE = os.path.dirname(os.path.abspath(__file__)); CLF = sys.argv[1]


def boot(v, seed=0):
    rng = np.random.default_rng(seed); m = v[rng.integers(0, len(v), (10000, len(v)))].mean(1); return np.quantile(m, .025), np.quantile(m, .975)


rows = []; ch = []
for f in sorted(glob.glob(f"{HERE}/cells_{CLF}/cell_*.npz")):
    Z = np.load(f); y = Z["y"]; K = [str(k).split("|") for k in Z["keys"]]; S = Z["S"]; name = os.path.basename(f)[5:-4].replace("_", " "); d = {}
    for i, k in enumerate(K): d.setdefault(tuple(k[:3]), []).append(est.auroc(y, S[i]))
    rows += [dict(cell=name, arm=k[0], spec=k[1], condition=k[2], auroc=float(np.mean(v))) for k, v in d.items()]
    ch += [dict(cell=name, fit=s.split("|")[0], C=float(s.split("|")[3])) for s in Z["chosen"]]
Zr = pd.DataFrame(rows); Zr.to_csv(f"{HERE}/auroc_{CLF}.csv", index=False); g = lambda a: Zr[Zr.arm == a].set_index(["cell", "condition"]).auroc; O = []; P = []
D = (g(f"{CLF}_trainproj") - g(f"{CLF}_raw")).dropna().reset_index()
for c, h in D.groupby("condition"):
    v = h.auroc.values; lo, hi = boot(v); O.append(dict(delta=f"{CLF}_trainproj-{CLF}_raw", condition=c, cells=len(v), mean=v.mean(), lo=lo, hi=hi)); P.append(dict(condition=c, pos=int((v > 0).sum()), cells=len(v)))
ORDER = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct", "yes_no"]
O = pd.DataFrame(O).set_index("condition").loc[ORDER].reset_index()[["delta", "condition", "cells", "mean", "lo", "hi"]]; O.to_csv(f"{HERE}/deltas_{CLF}.csv", index=False)
P = pd.DataFrame(P).set_index("condition").loc[ORDER].reset_index(); P.to_csv(f"{HERE}/pos_{CLF}.csv", index=False)
C = pd.DataFrame(ch).groupby(["fit", "C"]).size().rename("n").reset_index(); C.to_csv(f"{HERE}/chosen_{CLF}.csv", index=False)
print(O.merge(P, on=["condition", "cells"]).round(4).to_string(index=False)); print(C.to_string(index=False))
