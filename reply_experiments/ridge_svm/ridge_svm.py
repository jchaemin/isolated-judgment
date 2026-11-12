"""Ridge and linear SVM readouts on raw and projected training states. Usage: python reply_experiments/ridge_svm/ridge_svm.py {ridge|svm} [workers]"""
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from lib import *
from multiprocessing import Pool
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
HERE = os.path.dirname(os.path.abspath(__file__)); CLF = sys.argv[1]; OUT = f"{HERE}/cells_{CLF}"


def fit(X, y, C):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = RidgeClassifier(alpha=1.0 / C) if CLF == "ridge" else LinearSVC(C=C, max_iter=20000, random_state=0)
        w = np.ravel(m.fit(X, y).coef_)
    return w / np.linalg.norm(w)


def own(X, y, q):
    vo = {C: [] for C in est.GRID}
    for itr, iva in est._inner_splits(q):
        if len(np.unique(y[itr])) < 2 or len(np.unique(y[iva])) < 2: continue
        for C in est.GRID:
            w = fit(X[itr], y[itr], C); vo[C].append(est.rank_auroc(y[iva], X[iva] @ w))
    C_own = max(est.GRID, key=lambda C: (np.mean(vo[C]) if vo[C] else -1, -C))
    return fit(X, y, C_own), C_own


def cell_run(cell):
    name = cell["name"]; tag = name.replace(" ", "_"); fn = f"{OUT}/cell_{tag}.npz"
    S, ctask, cqid, cword = load(cell); y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; conds = ["yes_no"] + [t for t in config.HELD_OUT if all((t, p) in H for p in pids)]
    SC = {}; CS = []
    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ytr = y[tr]; ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); Xte = {t: ss.transform(H[(t, p)][te]) for t in conds}
            wlr, c1 = own(X, ytr, q[tr])
            B, k = build_B(H, tr, p, q, cross, ctask, cqid, cword, config.FORMATS, True, k_fixed=8); Bs = std_basis(B, ss)
            wtp, c2 = own(X - (X @ Bs.T) @ Bs, ytr, q[tr]); CS += [("raw", fi, p, c1), ("trainproj", fi, p, c2)]
            for t in conds:
                pool_put(SC, n, (f"{CLF}_raw", "-", t, p), te, Xte[t] @ wlr)
                pool_put(SC, n, (f"{CLF}_trainproj", "k8", t, p), te, (Xte[t] - (Xte[t] @ Bs.T) @ Bs) @ wtp)
    np.savez_compressed(fn, y=y, q=q, keys=np.array(["|".join(k) for k in SC]), S=np.array(list(SC.values()), dtype=np.float64),
                        chosen=np.array([f"{a}|{b}|{c}|{d}" for a, b, c, d in CS])); print(name, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cs = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]; assert len(cs) == 14
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4, maxtasksperchild=1) as P: P.map(cell_run, cs, chunksize=1)
