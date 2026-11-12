"""Verdict accuracy of the isolated readout and of the model's own verdict. Usage: python reply_experiments/accuracy/accuracy.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from tasks import *
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))


def best_cut(s, y):
    o = np.argsort(s); s_, y_ = s[o], y[o]; n1 = y.sum(); n0 = len(y) - n1
    cum1 = np.r_[0, np.cumsum(y_)]; cum0 = np.r_[0, np.cumsum(1 - y_)]
    ba = 0.5 * ((n1 - cum1) / n1 + cum0 / n0); i = int(np.argmax(ba))
    if i == 0: return s_[0] - 1e-9
    if i == len(s_): return s_[-1] + 1e-9
    return 0.5 * (s_[i - 1] + s_[i])


def cv_cut(X, y, qtr, Bs):
    sc = np.zeros(len(y))
    for itr, iva in est._inner_splits(qtr):
        w = est.class_mean(X[itr], y[itr]); w = w - Bs.T @ (Bs @ w); sc[iva] = X[iva] @ (w / np.linalg.norm(w))
    return best_cut(sc, y)


def run(cell):
    S = data.load_cell(cell, config.FIVE, rows=ROWS); y, q, H, marg, cross, folds = (S[k] for k in ("y", "q", "H", "marg", "cross", "folds")); prov = cross_provenance(cell["model"])
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; conds = [t for t in config.HELD_OUT if all((t, p) in H and (t, p) in marg for p in pids)]; rows = []
    for fi, (tr, te) in enumerate(folds):
        ytr, yte = y[tr], y[te]
        for p in pids:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); wcm = est.class_mean(X, ytr)
            w, Bs = isolated_direction(H, tr, p, q, cross, prov, ss, wcm, basis=True); cut = cv_cut(X, ytr, q[tr], Bs)
            for t in conds:
                s = ss.transform(H[(t, p)][te]) @ w; mg = marg[(t, p)][te]; r = dict(cell=cell["name"], split=fi, wording=p, condition=t, n=len(te), n1=int(yte.sum()), n0=int((1 - yte).sum()))
                r["model_tp"], r["model_tn"] = counts((mg > 0).astype(int), yte)
                r["isolated_tp"], r["isolated_tn"] = counts((s > float(np.quantile(s, 1 - ytr.mean()))).astype(int), yte)
                r["isolated_cv_tp"], r["isolated_cv_tn"] = counts((s > cut).astype(int), yte); rows.append(r)
    print(cell["name"], flush=True); return rows


if __name__ == "__main__":
    cs = [c for c in config.cells() if c["family"] == "correctness" and c["primary"]]; assert len(cs) == 14
    with Pool(4, maxtasksperchild=1) as P: res = P.map(run, cs, chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(HERE + "/counts.csv", index=False)
