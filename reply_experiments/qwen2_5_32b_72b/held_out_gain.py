"""Qwen2.5-7B to 72B: plain and isolated readouts on the held-out conditions. Usage: python reply_experiments/qwen2_5_32b_72b/held_out_gain.py"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from readout import config, data, estimators as est
import model_entries as me
MODELS = {"7B": "Qwen2.5-7B-Instruct", "14B": "Qwen2.5-14B-Instruct", "32B": "Qwen2.5-32B-Instruct", "72B": "Qwen2.5-72B-Instruct"}
METHODS = ["plain", "rank8", "fitted"]


def scores(cell):
    me.link_view()
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    assert np.array_equal(q, np.arange(400)), cell["name"]
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; assert len(pids) == 5 and all((t, p) in H for t in config.HELD_OUT for p in pids), cell["name"]
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan))[te] = v
    for tr, te in fld:
        for p in pids:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); wcm = est.class_mean(X, y[tr])
            Bm, _ = est.expression_subspace(H, tr, p, q, cross, rule="fixed8"); Bs = np.linalg.qr((Bm / ss.scale_).T)[0].T
            W = {"plain": wcm, "rank8": est.unit(wcm - Bs.T @ (Bs @ wcm))}
            for t in config.HELD_OUT:
                Xt = ss.transform(H[(t, p)][te])
                for m, w in W.items(): put((m, t, p), te, Xt @ w)
                st = StandardScaler().fit(H[(t, p)][tr]); wt = est.class_mean(st.transform(H[(t, p)][tr]), y[tr])
                put(("fitted", t, p), te, st.transform(H[(t, p)][te]) @ wt)
    return [dict(cell=cell["name"], condition=t, **{m: float(est.aurocs_mean(y, [SC[(m, t, p)] for p in pids], fld)) for m in METHODS}) for t in config.HELD_OUT]


def main():
    me.link_view()
    cs = [c for c in config.cells() if c["family"] == "correctness" and c["model"] in MODELS.values() and c["primary"]]
    for m in MODELS.values():
        assert sorted(c["task"] for c in cs if c["model"] == m) == ["gsm8k", "mmlu", "triviaqa"], m
    with Pool(min(4, len(cs))) as P: out = P.map(scores, cs)
    C = pd.DataFrame([r for o in out for r in o]); C.to_csv(os.path.join(HERE, "cells.csv"), index=False)
    G = C.groupby("cell", sort=False)[METHODS].mean(); G["gain"] = G.rank8 - G.plain
    model_of = {c["name"]: c["model"] for c in cs}; rows = []
    for name, ms in [("7B", ["7B"]), ("14B", ["14B"]), ("32B", ["32B"]), ("72B", ["72B"])]:
        g = G[[model_of[c] in [MODELS[k] for k in ms] for c in G.index]]; v = g.gain.values
        rng = np.random.default_rng(0); cm = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
        rows.append(dict(model=name, cells=len(v), plain=g.plain.mean(), rank8=g.rank8.mean(), fitted=g.fitted.mean(), gain=v.mean(),
                         positive_cells=int((v > 0).sum()), gain_lo=np.percentile(cm, 2.5), gain_hi=np.percentile(cm, 97.5)))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "by_model.csv"), index=False)


if __name__ == "__main__":
    main()
