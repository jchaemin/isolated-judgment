"""Qwen3.5-9B under negation: plain, isolated and fitted readouts and the model's verdict. Usage: python reply_experiments/qwen3_5_9b/negated_improvement.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from readout import config, data, estimators as est, storage
import model_entries as me


def link_view():
    D = os.path.join(HERE, "data")
    for sub in ("cells", "labels", "states", "steering"):
        os.makedirs(os.path.join(D, sub), exist_ok=True)
        for m in os.listdir(os.path.join(ROOT, "data", sub)):
            L = os.path.join(D, sub, m)
            if not os.path.lexists(L): os.symlink(os.path.join(ROOT, "data", sub, m), L)
    for f in os.listdir(os.path.join(ROOT, "data", "prompts")):
        L = os.path.join(D, "prompts", f)
        if not os.path.lexists(L): os.symlink(os.path.join(ROOT, "data", "prompts", f), L)
    storage.DATA = D


def scores(cell):
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, marg, cross, fld = (S[k] for k in ("y", "q", "n", "H", "marg", "cross", "folds"))
    assert np.array_equal(q, np.arange(len(q))) and len(q) == 400
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; assert all(("negated", p) in H and ("negated", p) in marg for p in pids)
    SC = {}
    for tr, te in fld:
        for p in pids:
            ytr = y[tr]; ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); wcm = est.class_mean(X, ytr)
            Bm, _ = est.expression_subspace(H, tr, p, q, cross, rule="fixed8"); Bs = np.linalg.qr((Bm / ss.scale_).T)[0].T
            Xt = ss.transform(H[("negated", p)][te])
            for nm, w in (("readout", wcm), ("iso_k8", est.unit(wcm - Bs.T @ (Bs @ wcm)))):
                SC.setdefault((nm, p), np.full(n, np.nan))[te] = Xt @ w
            st = StandardScaler().fit(H[("negated", p)][tr]); wt = est.class_mean(st.transform(H[("negated", p)][tr]), ytr)
            SC.setdefault(("fitted", p), np.full(n, np.nan))[te] = st.transform(H[("negated", p)][te]) @ wt
    for p in pids:
        SC[("verdict", p)] = np.asarray(marg[("negated", p)], float)
    Z = {}
    for k, s in SC.items():
        assert not np.isnan(s).any(), k; z = s.copy()
        for _, te in fld: z[te] = (s[te] - s[te].mean()) / (s[te].std() or 1.0)
        Z[k] = z
    return y, pids, Z


def auc_rows(y, s):
    r = rankdata(s, axis=1); npos = y.sum(1); nneg = y.shape[1] - npos
    return (np.where(y == 1, r, 0).sum(1) - npos * (npos + 1) / 2) / (npos * nneg)


def run(cell):
    y, pids, Z = scores(cell); n = len(y)
    A_ = {m: np.mean([auc_rows(y[None], Z[(m, p)][None]) for p in pids], 0) for m in ("readout", "iso_k8", "fitted", "verdict")}
    print(cell["name"], flush=True)
    return [dict(cell=cell["name"], condition="negated", rows=n, quantity=m, auroc=A_[m][0]) for m in ("readout", "iso_k8", "fitted", "verdict")]


if __name__ == "__main__":
    link_view()
    out = [r for c in config.cells() if c["model"] == me.NAME and c["family"] == "correctness" for r in run(c)]
    R = pd.DataFrame(out); R.to_csv(os.path.join(HERE, "negated_improvement.csv"), index=False); print(R.round(3).to_string(index=False))
