"""Qwen2.5-7B on entailment under negation: the verdict, plain and isolated readouts. Usage: python reply_experiments/anli/anli.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from tasks import *
HERE = os.path.dirname(os.path.abspath(__file__)); MODEL, TASK = "Qwen2.5-7B-Instruct", "entailment"


def run():
    S = task_states(MODEL, TASK); y, q, H, marg, folds = (S[k] for k in ("y", "q", "H", "marg", "folds")); prov, cross = task_prov(MODEL); name = config.cell_name(MODEL, TASK); rows = []
    pids = [p for p in config.FIVE if ("yes_no", p) in H]
    for fi, (tr, te) in enumerate(folds):
        ytr, yte = y[tr], y[te]
        for p in pids:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); wcm = est.class_mean(ss.transform(H[("yes_no", p)][tr]), ytr); w = isolated_direction(H, tr, p, q, cross, prov, ss, wcm)
            Xt = ss.transform(H[("negated", p)][te])
            rows.append(dict(cell=name, condition="negated", split=fi, wording=p, readout_auroc=est.auroc(yte, Xt @ (wcm / np.linalg.norm(wcm))), iso_k8_auroc=est.auroc(yte, Xt @ w), verdict_auroc=est.auroc(yte, marg[("negated", p)][te])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = run(); d.to_csv(HERE + "/auroc_by_split.csv", index=False)
    r = pd.DataFrame([{a: h[a].mean() for a in ("verdict_auroc", "iso_k8_auroc", "readout_auroc")} for _, h in d.groupby("wording")]).mean(numeric_only=True)
    out = pd.DataFrame([dict(cell=d.cell.iloc[0], condition="negated", verdict_auroc=r.verdict_auroc, iso_k8_auroc=r.iso_k8_auroc, readout_auroc=r.readout_auroc)]); out.to_csv(HERE + "/auroc.csv", index=False); print(out.round(3).to_string(index=False))
