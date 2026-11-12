"""Readouts on 20 new instruction wordings, with the subspace from the original wordings. Usage: python reply_experiments/new_wordings/analyze.py [--indir DIR]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from scope_lib import *
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ARGS = None; F = config.FORMATS
MODELS7 = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3"]


def build_B_src(H, tr, ps, q, cross, ctask, cqid, cword, k):
    qtr = q[tr]; blocks, ids = [], []
    for p in ps:
        for i, f in enumerate(F):
            for g in F[i + 1:]:
                if (f, p) in H and (g, p) in H: blocks.append(H[(g, p)][tr] - H[(f, p)][tr]); ids.append(None)
    pool_ = np.flatnonzero(np.isin(cword, [int(str(p)[1:]) for p in ps]))
    idx = np.random.default_rng(SUB).choice(pool_, min(len(pool_), 4 * len(tr)), replace=False); cr = cross[idx]
    blocks.append(cr); ids.append(ctask[idx] * 100 + cword[idx])
    return svd_top(stack(blocks, ids, CENTER), k)


def job(cell):
    S, ctask, cqid, cword = load(cell); y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    Z = np.load(os.path.join(ARGS.indir, cell["model"], cell["task"], "nwords.npz")); assert (Z["qid"].astype(int) == q).all() and (Z["y"].astype(int) == y).all()
    src = [p for p in config.FIVE if sum((f, p) in H for f in F) >= 2]
    ws = sorted({int(k.split("_")[-1][1:]) for k in Z.files if k.startswith("G_yes_no_N")}); nws = [f"N{i:02d}" for i in ws if f"G_negated_N{i:02d}" in Z.files]
    SC = {}
    for fi, (tr, te) in enumerate(fld):
        Bo = build_B_src(H, tr, src, q, cross, ctask, cqid, cword, 8)
        for w in nws:
            Gy, Gn = Z[f"G_yes_no_{w}"].astype(np.float64), Z[f"G_negated_{w}"].astype(np.float64); ss = StandardScaler().fit(Gy[tr]); Xy, Xn = ss.transform(Gy), ss.transform(Gn); wcm = est.class_mean(Xy[tr], y[tr])
            Bs = std_basis(Bo, ss)
            pool_put(SC, n, ("plain", "negated", w), te, Xn[te] @ wcm); pool_put(SC, n, ("rank8", "negated", w), te, Xn[te] @ (wcm - Bs.T @ (Bs @ wcm)))
            pool_put(SC, n, ("fitted", "negated", w), te, Xn[te] @ est.class_mean(Xn[tr], y[tr]))
    print(cell["name"], flush=True)
    return [dict(cell=cell["name"], wording=w, form=f, method=m, auroc=float(est.auroc(y, SC[(m, f, w)]))) for (m, f, w) in SC]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--indir", default=os.path.join(HERE, "out")); ap.add_argument("--procs", type=int, default=2); ARGS = ap.parse_args()
    cs = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"] and c["model"] in MODELS7]; assert len(cs) == 11
    with Pool(min(ARGS.procs, len(cs))) as P: out = P.map(job, cs)
    R = pd.DataFrame([r for o in out for r in o]); R.to_csv(os.path.join(HERE, "by_cell.csv"), index=False)
    P_ = R.pivot_table(index=["cell", "wording", "form"], columns="method", values="auroc"); cw = []
    for (w, f), g in P_.groupby(level=["wording", "form"]):
        d = (g["rank8"] - g["plain"]).dropna(); cw.append(dict(wording=w, form=f, mean=d.mean()))
    CW = pd.DataFrame(cw)
    counts(R.drop(columns=["wording"])).to_csv(os.path.join(HERE, "summary.csv"), index=False)
    pd.DataFrame([dict(form="negated", contrast="rank8-plain", wordings_pos=int((CW["mean"] > 0).sum()), n_wordings=len(CW))]).to_csv(os.path.join(HERE, "summary_wordings.csv"), index=False)
    m = means(R.drop(columns=["wording"])); m.to_csv(os.path.join(HERE, "means.csv"), index=False); print(m.round(3).to_string(index=False))
