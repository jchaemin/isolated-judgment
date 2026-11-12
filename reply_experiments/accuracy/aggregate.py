"""Accuracy and balanced accuracy. Usage: python reply_experiments/accuracy/aggregate.py"""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); d = pd.read_csv(HERE + "/counts.csv"); rows = []
for (cell, c), g_ in d.groupby(["cell", "condition"]):
    ww = []
    for w, h in g_.groupby("wording"):
        n = h.n.sum(); n1 = h.n1.sum(); n0 = h.n0.sum(); r = {}
        for m in ["model", "isolated", "isolated_cv"]:
            tp, tn = h[m + "_tp"].sum(), h[m + "_tn"].sum(); r[m + "_acc"] = (tp + tn) / n; r[m + "_ba"] = (tp / n1 + tn / n0) / 2
        r["always_correct_acc"] = n1 / n; r["always_correct_ba"] = 0.5
        ww.append(r)
    r = pd.DataFrame(ww).mean(numeric_only=True).to_dict(); r.update(cell=cell, condition=c); rows.append(r)
C = pd.DataFrame(rows); m = C.groupby("cell").mean(numeric_only=True); out = []
for met in ("acc", "ba"):
    out.append(dict(metric=met, cells=len(m), isolated=m[f"isolated_{met}"].mean(), isolated_cv=m[f"isolated_cv_{met}"].mean(), model=m[f"model_{met}"].mean(), always_correct=m[f"always_correct_{met}"].mean()))
out = pd.DataFrame(out).round(3); out.to_csv(HERE + "/accuracy.csv", index=False); print(out.to_string(index=False))
