"""ETHICS, BBQ and ToxiGen: plain and isolated readouts on the held-out conditions. Usage: python reply_experiments/domains/analyze.py [--models a,b] [--procs 2]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); RE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RE, "common"))
from tasks import *
from multiprocessing import Pool
TASKS = ["ethics", "bbq", "toxigen"]
TC = {"false_means_correct": "false_means_positive", "no_means_correct": "no_means_positive", "zero_means_correct": "zero_means_positive"}
CONDS = ["yes_no", "negated", "no_means_positive", "false_means_positive", "zero_means_positive"]


def dom(model, task):
    b = os.path.join(RE, task, "data", "states", model, task)
    D = {f: np.load(os.path.join(b, f + ".npz"), allow_pickle=True) for f in ("yes_no", "a_b", "b_a", "negated", "conditions")}
    y = D["yes_no"]["y"].astype(int); q = D["yes_no"]["qid"].astype(int); H = {}
    for p in config.FIVE:
        for f in ("yes_no", "a_b", "b_a", "negated"):
            assert (D[f]["qid"].astype(int) == q).all(); H[(f, p)] = D[f][f"G_{p}"].astype(np.float64)
    C = D["conditions"]; nc = len(C["qid"]); assert (C["qid"].astype(int) == q[:nc]).all()
    for cs in [str(x) for x in C["conventions"]]:
        if TC.get(cs, cs) not in config.FORMATS + CONDS: continue
        for p in config.FIVE: H[(TC.get(cs, cs), p)] = C[storage.key("G", cs, p)].astype(np.float64)
    it = json.load(open(os.path.join(RE, task, "data", "prompts", f"{task}_items.json")))
    assert len(it) == len(y) and [i["y"] for i in it] == list(y) and [i["example_id"] for i in it] == list(q)
    ix = {str(t): i for i, t in enumerate(D["yes_no"]["token_names"])}; vm = {}
    for p in config.FIVE:
        Zn = D["negated"][f"Z_{p}"].astype(np.float64); vm[p] = Zn[:, ix["No"]] - Zn[:, ix["Yes"]]
    return y, q, H, nc, vm


def run(model):
    prov, cross = task_prov(model); rows = []
    for task in TASKS:
        y, q, H, nc, vm = dom(model, task); conds = [t for t in CONDS if all((t, p) in H for p in config.FIVE)]
        y4, q4 = y[:nc], q[:nc]; H4 = {k: v[:nc] for k, v in H.items()}; folds = data.folds(y4, q4, rows=nc, order=q)
        for fi, (tr, te) in enumerate(folds):
            for p in config.FIVE:
                ss = StandardScaler().fit(H4[("yes_no", p)][tr]); wcm = est.class_mean(ss.transform(H4[("yes_no", p)][tr]), y4[tr]); w = isolated_direction(H4, tr, p, q4, cross, prov, ss, wcm)
                for t in conds:
                    Xt = ss.transform(H4[(t, p)][te]); lab = y4[te] >= 0
                    for meth, d in (("readout", wcm / np.linalg.norm(wcm)), ("isolated", w)):
                        rows.append(dict(model=model, task=task, method=meth, condition=t, wording=p, fold=fi, auroc=est.auroc(y4[te][lab], (Xt @ d)[lab])))
                    if task == "ethics" and t == "negated":
                        rows.append(dict(model=model, task=task, method="verdict", condition=t, wording=p, fold=fi, auroc=est.auroc(y4[te][lab], vm[p][:nc][te][lab])))
        print(f"[domains] {model} {task}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default=""); ap.add_argument("--procs", type=int, default=2); A = ap.parse_args()
    ms = [m for m in A.models.split(",") if m] or list(config.MODELS)
    with Pool(min(A.procs, len(ms)), maxtasksperchild=1) as P: res = P.map(run, ms, chunksize=1)
    L = pd.DataFrame([r for rr in res for r in rr]); V = L[L.method == "verdict"]; L = L[L.method != "verdict"]
    V = V.groupby(["model", "task", "condition", "wording"]).auroc.mean().groupby(["model", "task", "condition"]).mean().groupby(["task", "condition"]).agg(["size", "mean"])
    V.columns = ["models", "verdict"]; V.reset_index().to_csv(os.path.join(HERE, "verdict.csv"), index=False)
    W = L.groupby(["model", "task", "method", "condition", "wording"]).auroc.mean().reset_index()
    PM = W.groupby(["model", "task", "method", "condition"]).auroc.mean().unstack("method").reset_index()
    PM["task"] = pd.Categorical(PM.task, TASKS); PM["condition"] = pd.Categorical(PM.condition, CONDS)
    PM = PM.sort_values(["task", "condition", "model"])[["model", "task", "condition", "readout", "isolated"]]; PM.to_csv(os.path.join(HERE, "per_model.csv"), index=False)
    S = PM.groupby(["task", "condition"], observed=True).agg(models=("model", "size"), readout=("readout", "mean"), isolated=("isolated", "mean")).reset_index()
    S["gain"] = S.isolated - S.readout; S.to_csv(os.path.join(HERE, "summary.csv"), index=False)
    print(S.round(3).to_string(index=False))
