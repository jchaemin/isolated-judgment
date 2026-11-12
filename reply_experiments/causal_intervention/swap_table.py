"""Verdict changes and readout AUROC after the swaps. Usage: python reply_experiments/causal_intervention/swap_table.py"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
ARM = {"component": "E_manuscript_a1.0", "complement": "comp_scaled_manuscript", "full": "full", "random": "haar_manuscript", "noop": "noop"}


def auc(y, s):
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return np.nan if n1 == 0 or n0 == 0 else float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wf(d, c): return float(np.nanmean([auc(g.y.values, g[c].values) for _, g in d.groupby(["fold", "wording"])])) if len(d) else np.nan


rows = []; over = []
for f in sorted(glob.glob(HERE + "/cells/swap_*.parquet")):
    d = pd.read_parquet(f); cell = d.cell.iloc[0]
    d.loc[d.arm.str.fullmatch(r"haar\d_manuscript"), "arm"] = "haar_manuscript"; d = d[d.arm.isin(ARM.values())]
    k = ["row", "wording", "rev", "recipient"]
    t0 = d[d.arm == "noop"].set_index(k).top.rename("top0")
    m = d.join(t0, on=k); m["flip"] = (m.top != m.top0).astype(float)
    out = []
    for (rev, rec, fo, wo), g in m.groupby(["rev", "recipient", "fold", "wording"]):
        n = g[g.arm == "noop"]; z = {}
        for s in ["s_plain", "s_own_negated", "s_own_no_means_correct"]: z[s] = (g[s].values - n[s].mean()) / n[s].std()
        out.append(g.assign(rev_all=(z["s_plain"] + (z["s_own_negated"] + z["s_own_no_means_correct"]) / 2) / 2))
    m = pd.concat(out)
    for (rev, rec), g in m.groupby(["rev", "recipient"]):
        direction = "reversed->ordinary" if rec == "yes_no" else "ordinary->reversed"
        for nm, a in ARM.items():
            hh = g[g.arm == a]
            if len(hh) == 0: continue
            rows.append(dict(cell=cell, recipient=rec, direction=direction, intervention=nm, n=len(hh), flip=hh.flip.mean(), reading_rev_all=wf(hh, "rev_all")))
        W = {nm: g[g.arm == v].set_index(k) for nm, v in ARM.items()}; c = W["component"].flip == 1; base = dict(cell=cell, recipient=rec, direction=direction)
        if len(W["full"]):
            ff = (W["full"].flip.loc[c.index] == 1).values
            for sub, sel in [("full_changed", ff), ("full_unchanged", ~ff)]: over.append(dict(base, quantity="flip", intervention="component", subset=sub, value=c.values[sel].mean()))
            for nm in ["component", "complement", "random"]:
                t = W[nm].top; over.append(dict(base, quantity="same_answer_as_full", intervention=nm, subset="all", value=(t.values == W["full"].top.loc[t.index].values).mean()))
        for sub, sel in [("component_changed", c.values), ("component_unchanged", ~c.values)]:
            for nm in ["noop", "component"]: over.append(dict(base, quantity="reading_rev_all", intervention=nm, subset=sub, value=wf(W[nm].loc[c.index[sel]], "rev_all")))
C = pd.DataFrame(rows); C.to_csv(HERE + "/swap_cells.csv", index=False)
S = []
for key, g in C.groupby(["recipient", "direction", "intervention"]):
    r = dict(zip(["recipient", "direction", "intervention"], key)); r["cells"] = g.cell.nunique()
    for c in ["flip", "reading_rev_all"]: r[c] = g[c].dropna().values.mean()
    S.append(r)
S = pd.DataFrame(S); S.to_csv(HERE + "/swap_summary.csv", index=False); print(S.round(3).to_string(index=False))
K = ["recipient", "direction", "quantity", "intervention", "subset"]; V = []
for key, g in pd.DataFrame(over).groupby(K, sort=False):
    v = g.value.dropna(); V.append(dict(zip(K, key), cells=len(v), value=v.values.mean()))
V = pd.DataFrame(V); V.to_csv(HERE + "/swap_overlap.csv", index=False); print(V.round(3).to_string(index=False))
