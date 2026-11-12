"""Shared analysis for the new wordings, explanations and critiques."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *


def score_cell(S, ctask, cqid, cword, NEW, methods):
    y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds")); F = config.FORMATS
    pids = sorted({p for _, p in NEW}); forms = sorted({f for f, _ in NEW}); SC = {}
    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); wcm = est.class_mean(X, y[tr]); Ws = {"plain": wcm}
            if "rank8" in methods:
                B, k = build_B(H, tr, p, q, cross, ctask, cqid, cword, F, True, k_fixed=8); Bs = std_basis(B, ss); Ws["rank8"] = wcm - Bs.T @ (Bs @ wcm)
            for f in forms:
                if (f, p) not in NEW: continue
                Xt = ss.transform(NEW[(f, p)]); Xte = Xt[te]
                for m, w in Ws.items():
                    if m in methods: pool_put(SC, n, (m, f, p), te, Xte @ w)
                if "fitted" in methods: pool_put(SC, n, ("fitted", f, p), te, Xte @ est.class_mean(Xt[tr], y[tr]))
    return SC


def readout_rows(cell, y, SC):
    rows = []
    for m, f in sorted({(m, f) for (m, f, p) in SC}):
        ps = sorted(p for (mm, ff, p) in SC if (mm, ff) == (m, f)); a = np.array([float(est.auroc(y, SC[(m, f, p)])) for p in ps])
        rows.append(dict(cell=cell, form=f, method=m, auroc=a.mean(), n_wordings=len(ps)))
    return rows


def own_row(cell, form, y, per_p):
    a = np.array([est.auroc(y, s) for s in per_p.values()])
    return dict(cell=cell, form=form, method="model", auroc=float(a.mean()), n_wordings=len(per_p))


def counts(R, iso="rank8", other="plain"):
    out = []; T = R[R.method.isin(["plain", "rank8", "fitted", "model"])].pivot_table(index=["cell", "form"], columns="method", values="auroc")
    for form in sorted(R.form.unique()):
        t = T.xs(form, level="form"); d = (t[iso] - t[other]).dropna().values
        out.append(dict(form=form, contrast=f"{iso}-{other}", cells_pos=int((d > 0).sum()), n_cells=len(d)))
    return pd.DataFrame(out)


def means(R):
    T = R.pivot_table(index=["cell", "form"], columns="method", values="auroc"); g = T.groupby(level="form")
    return pd.concat([g.mean().stack().rename("auroc"), g.count().stack().rename("n_cells")], axis=1).reset_index().rename(columns={"level_1": "method"})


def explanation_cells():
    return ["Qwen2.5-7B GSM8K", "Qwen2.5-7B MMLU", "Qwen2.5-7B TriviaQA", "Llama-3.1-8B GSM8K", "Llama-3.1-8B TriviaQA", "Mistral-7B TriviaQA"]
