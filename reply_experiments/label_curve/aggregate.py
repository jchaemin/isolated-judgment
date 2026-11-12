"""AUROC by label budget. Usage: python reply_experiments/label_curve/aggregate.py"""
import glob, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct"]
NS = ["16", "32", "64", "80", "all"]; ARMS = ["pooled_L", "iso_Ln"]
p = pd.concat([pd.read_csv(f, dtype={"n": str}) for f in sorted(glob.glob(HERE + "/cells/auroc_*.csv"))]); p = p[p.cond.isin(C)]; assert p.cell.nunique() == 14
sk = pd.concat([pd.read_csv(f, dtype={"n": str}) for f in sorted(glob.glob(HERE + "/cells/skip_*.csv")) if os.path.getsize(f) > 1])
CM = p.groupby(["arm", "n", "cell", "cond"])["z"].mean().unstack("cond")[C].mean(axis=1)
CM.rename("auroc").reset_index().to_csv(HERE + "/curve_by_setting.csv", index=False)
U = p.groupby(["arm", "n", "cell", "draw", "wording"]).z.apply(lambda v: v.isna().any())
rows = []
for a in ARMS:
    for n in NS:
        if (a, n) not in CM.index: continue
        x = CM.loc[(a, n)]; u = U.loc[(a, n)]
        tot = int(u.size); s = sk[(sk.arm == a) & (sk.n == n)]
        rows.append(dict(arm=a, n=n, auroc=x.dropna().mean(), settings=int(x.notna().sum()), nan_units=int(u.sum()), units=tot,
                         skipped_fits=len(s), skipped_one_class=int((s.why == "one class").sum()), skipped_subspace=int(s.why.str.startswith("subspace").sum())))
out = pd.DataFrame(rows); out.to_csv(HERE + "/curve.csv", index=False)
pd.set_option("display.width", 200); print(out.round(4).to_string(index=False))
