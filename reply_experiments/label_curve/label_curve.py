"""Isolated and pooled readouts at equal numbers of labels. Usage: python reply_experiments/label_curve/label_curve.py <workers>"""
import os, sys, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from lib import *
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "cells"); os.makedirs(OUT, exist_ok=True)
NS = [16, 32, 64, 80, "all"]; NDRAW = 10


def cell_run(cell):
    name = cell["name"]; fn = os.path.join(OUT, "auroc_" + name.replace(" ", "_") + ".csv")
    S, ctask, cqid, cword = load(cell); y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; conds = [t for t in config.HELD_OUT if all((t, p) in H for p in pids)]
    SC = {}; fold_of = np.full(n, -1); SKIP = []

    def put(arm, t, nn, d, p, te, w, X):
        key = (arm, t, str(nn), d, p)
        if key not in SC: SC[key] = np.full(n, np.nan)
        if w is not None: SC[key][te] = X @ (w / np.linalg.norm(w))

    def cmean(X, yy):
        return None if len(np.unique(yy)) < 2 else est.class_mean(X, yy)

    def draw(fi, p, nn, d, uq, qs):
        k = len(uq) if nn == "all" else min(nn, len(uq))
        rng = np.random.default_rng(zlib.crc32(f"{name}|{fi}|{p}|{nn}|{d}".encode())); return np.isin(qs, rng.permutation(uq)[:k])

    for fi, (tr, te) in enumerate(fld):
        fold_of[te] = fi; qs = q[tr]; uq = np.unique(qs)
        for p in pids:
            ytr = y[tr]; Hy = H[("yes_no", p)]; ss = StandardScaler().fit(Hy[tr]); Xte = {t: ss.transform(H[(t, p)][te]) for t in conds}
            for nn in NS:
                for d in range(NDRAW):
                    W = {}; WHY = {}
                    if not (nn == "all" and d > 0):
                        sm = draw(fi, p, nn, d, uq, qs); trs = tr[sm]; ys = ytr[sm]; wL = cmean(ss.transform(Hy[trs]), ys); W["iso_Ln"] = None
                        if wL is not None:
                            try:
                                B, _ = build_B(H, trs, p, q, cross, ctask, cqid, cword, config.FORMATS, True); Bs = std_basis(B, ss); W["iso_Ln"] = wL - Bs.T @ (Bs @ wL)
                            except Exception as e:
                                WHY["iso_Ln"] = "subspace:" + type(e).__name__
                    h = (len(uq) // 2) if nn == "all" else nn // 2; sm = draw(fi, p, h, d, uq, qs); trs = tr[sm]; ys = ytr[sm]
                    Xf = {f: ss.transform(H[(f, p)][trs]) for f in ("a_b", "b_a")}; W["pooled_L"] = cmean(np.concatenate([Xf["a_b"], Xf["b_a"]]), np.concatenate([ys, ys]))
                    for a, w in W.items():
                        if w is None: SKIP.append(dict(cell=name, fold=fi, wording=p, arm=a, n=str(nn), draw=d, why=WHY.get(a, "one class")))
                        for t in conds: put(a, t, nn, d, p, te, w, Xte[t])
    rows = []
    for (a, t, nn, d, p), v in SC.items():
        if not np.isfinite(v).all(): rows.append(dict(cell=name, arm=a, cond=t, n=nn, draw=d, wording=p, z=np.nan)); continue
        z = v.copy()
        for fi in np.unique(fold_of): est.pool(z, np.where(fold_of == fi)[0])
        rows.append(dict(cell=name, arm=a, cond=t, n=nn, draw=d, wording=p, z=est.auroc(y, z)))
    pd.DataFrame(rows).to_csv(fn, index=False); pd.DataFrame(SKIP).to_csv(fn.replace("auroc_", "skip_"), index=False)
    print(name, flush=True); return name


if __name__ == "__main__":
    cs = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]; assert len(cs) == 14
    if sys.argv[1] == "one": cell_run([c for c in cs if c["name"] == sys.argv[2]][0])
    else:
        with Pool(int(sys.argv[1]), maxtasksperchild=1) as pool: pool.map(cell_run, cs, chunksize=1)
