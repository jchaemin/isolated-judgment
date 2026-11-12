"""Change in AUROC of the MLP readout from projecting the states. Usage: python reply_experiments/mlp/aggregate.py"""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from env import *
HERE = os.path.dirname(os.path.abspath(__file__)); ARMS = ["mlp_raw", "mlp_proj8"]
ORDER = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct", "yes_no"]


def boot(v, seed=0):
    rng = np.random.default_rng(seed); m = v[rng.integers(0, len(v), (10000, len(v)))].mean(1); return np.quantile(m, .025), np.quantile(m, .975)


rows = []
for f in sorted(glob.glob(os.path.join(HERE, "cells", "cell_*.npz"))):
    Z = np.load(f); y = Z["y"].astype(int); K = [str(k).split("|") for k in Z["keys"]]; S = Z["S"].astype(np.float64); name = os.path.basename(f)[5:-4].replace("_", " "); d = {}
    for i, k in enumerate(K): d.setdefault((k[0], k[1]), []).append(est.auroc(y, S[i]) if np.isfinite(S[i]).all() else np.nan)
    rows += [dict(cell=name, arm=a, condition=c, auroc=float(np.nanmean(v)), n_scores=int(np.isfinite(v).sum()), n_nan=int(np.isnan(v).sum())) for (a, c), v in d.items()]
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "auroc.csv"), index=False); O = []
for c in ORDER:
    T = R[R.condition == c].pivot(index="cell", columns="arm", values="auroc")
    if T.empty: continue
    v = (T["mlp_proj8"] - T["mlp_raw"]).dropna().values; lo, hi = boot(v)
    O.append(dict(condition=c, raw=T["mlp_raw"].mean(), proj8=T["mlp_proj8"].mean(), delta=v.mean(), lo=lo, hi=hi, pos=int((v > 0).sum()), cells=len(v)))
O = pd.DataFrame(O); O.to_csv(os.path.join(HERE, "deltas.csv"), index=False); pd.set_option("display.width", 200); print(O.round(4).to_string(index=False))
CH = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "cells", "chosen_*.csv")))]).dropna(subset=["h"])
C = CH.groupby(["arm", "h", "wd"]).agg(n=("E", "size"), mean_epochs=("E", "mean")).reset_index(); C.to_csv(os.path.join(HERE, "chosen.csv"), index=False); print(C.to_string(index=False))
