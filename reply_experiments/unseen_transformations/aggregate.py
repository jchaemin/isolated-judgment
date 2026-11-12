"""Gain of the non-reversing isolated readout over the plain readout. Usage: python reply_experiments/unseen_transformations/aggregate.py"""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from env import *
HERE = os.path.dirname(os.path.abspath(__file__)); TARGETS = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct"]


def boot(v, seed=6):
    m = v[np.random.default_rng(seed).integers(0, len(v), (10000, len(v)))].mean(1); return np.quantile(m, .025), np.quantile(m, .975)


rows = []
for f in sorted(glob.glob(HERE + "/cells/cell_*.npz")):
    Z = np.load(f); y = Z["y"]; K = [str(k).split("|") for k in Z["keys"]]; S = Z["S"]; name = os.path.basename(f)[5:-4].replace("_", " "); d = {}
    for i, k in enumerate(K): d.setdefault(tuple(k[:3]), []).append(est.auroc(y, S[i]))
    rows += [dict(cell=name, arm=k[0], spec=k[1], condition=k[2], auroc=float(np.mean(v))) for k, v in d.items()]
A = pd.DataFrame(rows)
R = A[A.arm == "readout"].set_index(["cell", "condition"]).auroc; out = []
g = A[A.arm == "no_reversal"]
for t in TARGETS:
    h = g[g.condition == t]; v = (h.set_index(["cell", "condition"]).auroc - R).dropna().values
    lo, hi = boot(v); out.append(dict(target=t, no_reversal_gain=v.mean(), no_reversal_lo=lo, no_reversal_hi=hi, no_reversal_pos=int((v > 0).sum())))
out = pd.DataFrame(out); out.to_csv(HERE + "/non_reversing_gain.csv", index=False); print(out.round(4).to_string(index=False))
