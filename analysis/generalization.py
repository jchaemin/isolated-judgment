"""Entailment and safety, additional models, other answer sources and a new instruction family (Appendix C.5 to C.8). Usage: python analysis/generalization.py <command> (or all)."""
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import config, data, estimators, prompts, storage


_PROCS = 4


_ROWS = 400


_TASK_CONDITION = {"correct_incorrect": "word_pair", "zero_means_correct": "zero_means_positive", "false_means_correct": "false_means_positive", "no_means_correct": "no_means_positive"}


_TASK_TARGETS = ["true_false", "a_b", "b_a", "sentence", "negated", "one_zero", "word_pair", "zero_means_positive", "false_means_positive", "no_means_positive"]


_JUDGE_PAIRS = list(zip(config.MODELS, config.MODELS[1:] + config.MODELS[:1]))


_ANSWER_SOURCE_TASKS = ["triviaqa", "mmlu"]


_CANDIDATE_TASKS = ["mmlu", "arc"]


_ALL_CELLS = None


_ORDER_CACHE = {}


def _all_cells():
    global _ALL_CELLS
    if _ALL_CELLS is None:
        _ALL_CELLS = config.cells()
    return _ALL_CELLS


def _cell_for(model, task):
    return next(c for c in _all_cells() if c["family"] == "correctness" and c["model"] == model and c["task"] == task)


def _row_order(model, task):
    key = (model, task)
    if key not in _ORDER_CACHE:
        _ORDER_CACHE[key] = data.load_cell(_cell_for(model, task), wordings=["P01"], rows=None)["q"]
    return _ORDER_CACHE[key]


def _first_rows(S, rows, fq):
    y, q = S["y"], S["q"]; N = min(rows, len(y))
    order = fq if (fq is not None and len(fq) >= len(q) and np.array_equal(fq[:len(q)], q)) else None
    out = dict(S, y=y[:N], q=q[:N], n=N, folds=data.folds(y, q, rows=N, order=order))
    for k in ("H", "marg", "comp"):
        if isinstance(out.get(k), dict):
            out[k] = {kk: (vv[:N] if hasattr(vv, "__len__") and len(vv) == len(y) else vv) for kk, vv in out[k].items()}
    return out


def _correctness_negation_pairs(model):
    X = []
    for c in _all_cells():
        if c["family"] == "correctness" and c["primary"] and c["model"] == model:
            S = data.load_cell(c, wordings=config.FIVE, rows=_ROWS)
            for p in config.FIVE:
                if ("negated", p) in S["H"] and ("yes_no", p) in S["H"]:
                    X.append(S["H"][("negated", p)] - S["H"][("yes_no", p)])
    return np.concatenate(X)


def _task_states(model, task):
    D = {f: data.states(model, task, f) for f in ("yes_no", "a_b", "b_a")}
    Nn = data.states(model, task, "negated")
    conds = data.states(model, task, "conditions")
    qf = D["yes_no"]["qid"].astype(int); y = D["yes_no"]["y"].astype(int)[:_ROWS]; q = qf[:_ROWS]
    assert (conds["qid"].astype(int) == q).all() and (Nn["qid"].astype(int)[:_ROWS] == q).all()
    H = {}; marg = {}
    nm = [str(x) for x in D["yes_no"]["token_names"]]; ix = {k: nm.index(k) for k in ("Yes", "No", "A", "B")}
    for p in config.FIVE:
        for f, (pos, neg) in (("yes_no", ("Yes", "No")), ("a_b", ("A", "B")), ("b_a", ("B", "A"))):
            H[(f, p)] = D[f][f"G_{p}"][:_ROWS].astype(np.float64)
            Z = D[f][f"Z_{p}"][:_ROWS].astype(np.float64); marg[(f, p)] = Z[:, ix[pos]] - Z[:, ix[neg]]
        H[("negated", p)] = Nn[f"G_{p}"][:_ROWS].astype(np.float64)
        Zn = Nn[f"Z_{p}"][:_ROWS].astype(np.float64); marg[("negated", p)] = Zn[:, ix["No"]] - Zn[:, ix["Yes"]]
        for c_stored in [str(x) for x in conds["conventions"]]:
            c = _TASK_CONDITION.get(c_stored, c_stored); H[(c, p)] = conds[storage.key("G", c_stored, p)].astype(np.float64)
            if c_stored != "sentence":
                Zz = conds[storage.key("Z", c_stored, p)].astype(np.float64); marg[(c, p)] = np.logaddexp(Zz[:, 0], Zz[:, 2]) - np.logaddexp(Zz[:, 1], Zz[:, 3])
    folds = data.folds(y, q, rows=_ROWS, order=qf)
    return dict(y=y, q=q, H=H, marg=marg, folds=folds)


def _tasks_cell(model, task, cross):
    S = _task_states(model, task); y, q, H, marg = S["y"], S["q"], S["H"], S["marg"]
    pids = [p for p in config.FIVE if ("yes_no", p) in H]; have = [t for t in _TASK_TARGETS if all((t, p) in H for p in pids)]
    n = len(y); SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; estimators.pool(SC[key], te)

    for tr, te in S["folds"]:
        for p in pids:
            ytr = y[tr]; Hy = H[("yes_no", p)]; ss = StandardScaler().fit(Hy[tr]); X = ss.transform(Hy[tr])
            Bm, _ = estimators.expression_subspace(H, tr, p, q, cross); Bs, _ = np.linalg.qr((Bm / ss.scale_).T); Bs = Bs.T
            wcm = estimators.class_mean(X, ytr); w_iso = wcm - Bs.T @ (Bs @ wcm)
            for nm, w_ in (("readout", wcm), ("isolated", w_iso)):
                for t in ["yes_no"] + have:
                    put((nm, t, p), te, ss.transform(H[(t, p)][te]) @ w_)
            for t in ["yes_no"] + have:
                st = StandardScaler().fit(H[(t, p)][tr]); Xt = st.transform(H[(t, p)][tr]); wt = estimators.class_mean(Xt, ytr)
                put(("fitted", t, p), te, st.transform(H[(t, p)][te]) @ wt)
    rows = []
    for nm in ("readout", "isolated", "fitted"):
        for t in ["yes_no"] + have:
            auroc_v = float(np.mean([estimators.auroc(y, SC[(nm, t, p)]) for p in pids if (nm, t, p) in SC])) if any((nm, t, p) in SC for p in pids) else float("nan")
            native_v = (float(np.mean([estimators.auroc(y, marg[(t, p)]) for p in pids if (t, p) in marg])) if t != "yes_no"
                        else float(np.mean([estimators.auroc(y, marg[("yes_no", p)]) for p in pids])))
            rows.append(dict(model=model, dataset=task, condition=t, method=nm, auroc=auroc_v, native=native_v))
    return rows


def _tasks_job(model):
    cross = _correctness_negation_pairs(model); rows = []
    for task in config.CROSS_TASKS:
        rows.extend(_tasks_cell(model, task, cross))
    print(f"[tasks] {model}", flush=True); return rows


def run_tasks():
    _all_cells()
    with Pool(_PROCS) as pool:
        out = pool.map(_tasks_job, config.MODELS)
    per_model = pd.DataFrame([r for rows in out for r in rows])
    rows, dec = [], {}
    for t in config.CROSS_TASKS:
        g = per_model[per_model.dataset == t]; piv = g.pivot_table(index=["model", "condition"], columns="method", values="auroc"); nat = g.groupby(["model", "condition"]).native.first()
        for c in ["yes_no"] + _TASK_TARGETS:
            sub = [m for m in g.model.unique() if (m, c) in piv.index]
            if not sub:
                continue
            pl = np.array([piv.loc[(m, c)]["readout"] for m in sub], float); pj = np.array([piv.loc[(m, c)]["isolated"] for m in sub], float)
            tf = np.array([piv.loc[(m, c)]["fitted"] for m in sub], float); nv = np.array([nat.get((m, c), np.nan) for m in sub], float)
            rows.append(dict(task=t, condition=c, readout=float(pl.mean()), isolated=float(pj.mean()), fitted=float(tf.mean()), verdict=float(np.nanmean(nv))))
        per = {}
        for m in g.model.unique():
            cm = lambda c: piv.loc[(m, c)]["readout"]; fj = lambda c: piv.loc[(m, c)]["isolated"]; tf = lambda c: piv.loc[(m, c)]["fitted"]
            per[config.model_name(m)] = dict(readout_vs_fitted=float(max(abs(cm(c) - tf(c)) for c in ("a_b", "b_a", "true_false"))),
                                              isolated_vs_fitted=float(max(abs(fj(c) - tf(c)) for c in ("negated", "no_means_positive"))))
        dec[t] = per
    os.makedirs("results/generalization", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/generalization/tasks.csv", index=False)
    json.dump(dec, open("results/generalization/tasks_per_model.json", "w"), indent=1, default=float)


def _answers_of_states(judge, cand, task):
    J, C = judge.split("/")[-1], cand.split("/")[-1]
    Z = np.load(storage.path("answers_of", model=J, task=task, source=C), allow_pickle=True)
    y = Z["y"].astype(int); q = Z["qid"].astype(int); H = {("yes_no", p): Z[storage.key("G", "yes_no", p)].astype(np.float64) for p in config.FIVE}
    return _first_rows(dict(y=y, q=q, H=H), _ROWS, _row_order(cand, task))


def _answer_source(judge, cand, task):
    So = data.load_cell(_cell_for(judge, task), wordings=config.FIVE, rows=_ROWS); Sf = _answers_of_states(judge, cand, task)
    assert (So["q"] == Sf["q"]).all(), "the two row sets must be the same questions"
    yo, yf, n = So["y"], Sf["y"], Sf["n"]; sc = {"own": {}, "for": {}}
    for tr, te in Sf["folds"]:
        for p in config.FIVE:
            for tag, S_, yy in (("own", So, yo), ("for", Sf, yf)):
                Hy = S_["H"][("yes_no", p)]; ss = StandardScaler().fit(Hy[tr]); X = ss.transform(Hy[tr]); w = estimators.class_mean(X, yy[tr])
                v = sc[tag].setdefault(p, np.full(n, np.nan)); v[te] = ss.transform(Sf["H"][("yes_no", p)][te]) @ w; estimators.pool(v, te)
    return dict(cell=f"{config.model_name(judge)} reads {config.cell_name(cand, task)}", negatives=int((1 - yf).sum()),
                own_answer_readout=float(np.mean([estimators.auroc(yf, v) for v in sc["own"].values()])),
                fitted_here=float(np.mean([estimators.auroc(yf, v) for v in sc["for"].values()])))


def _answer_source_job(job):
    r = _answer_source(*job); print(f"[other-model] {r['cell']}", flush=True); return r


def run_other_model():
    _all_cells(); jobs = [(j, c, t) for j, c in _JUDGE_PAIRS for t in _ANSWER_SOURCE_TASKS]
    with Pool(_PROCS) as pool:
        res = pool.map(_answer_source_job, jobs)
    os.makedirs("results/generalization", exist_ok=True)
    pd.DataFrame(res).to_csv("results/generalization/other_model_answers.csv", index=False)


def _candidate_folds(qx, fq):
    full = data.folds(fq, fq); fold_of = {int(fq[i]): k for k, (_, te) in enumerate(full) for i in te}
    fx = np.array([fold_of[int(v)] for v in qx]); return [(np.where(fx != k)[0], np.where(fx == k)[0]) for k in range(5)]


def _dataset_candidates_cell(job):
    model, task = job; cell = _cell_for(model, task); So = data.load_cell(cell, wordings=config.FIVE, rows=None)
    q_full, cross = So["q"], So["cross"]
    Z = data.states(model, task, "dataset_candidates"); yx = Z["y"].astype(int); qx = Z["qid"].astype(int)
    conv = [str(c) for c in Z["conventions"]]; Hx = {(c, p): Z[storage.key("G", c, p)].astype(np.float64) for c in conv for p in config.FIVE}
    nx = len(yx); SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(nx, np.nan)); SC[key][te] = v; estimators.pool(SC[key], te)

    for trx, tex in _candidate_folds(qx, q_full):
        for p in config.FIVE:
            if ("yes_no", p) not in Hx:
                continue
            sx = StandardScaler().fit(Hx[("yes_no", p)][trx]); X = sx.transform(Hx[("yes_no", p)][trx]); wx = estimators.class_mean(X, yx[trx])
            n_pairs = sum(1 for i, f in enumerate(config.FORMATS) for g in config.FORMATS[i + 1:] if (f, p) in Hx and (g, p) in Hx)
            wxs = None
            if n_pairs >= 10:
                Bm, _ = estimators.expression_subspace(Hx, trx, p, qx, cross); Qs = np.linalg.qr((Bm / sx.scale_).T)[0].T; wxs = wx - Qs.T @ (Qs @ wx)
            for c in conv:
                Ht = Hx[(c, p)]; st = StandardScaler().fit(Ht[trx]); Xt = st.transform(Ht[trx]); wt = estimators.class_mean(Xt, yx[trx])
                put(("fitted", c, p), tex, st.transform(Ht[tex]) @ wt)
                Zx = sx.transform(Ht[tex]); put(("readout", c, p), tex, Zx @ wx)
                if wxs is not None:
                    put(("isolated", c, p), tex, Zx @ wxs)
    rows = []
    for (rd, c) in sorted({k[:2] for k in SC}):
        rows.append(dict(cell=config.cell_name(model, task), condition=c, method=rd, auroc=float(np.mean([estimators.auroc(yx, v) for k, v in SC.items() if k[:2] == (rd, c)]))))
    for c in conv:
        zz = [Z[storage.key("Z", c, p)].astype(np.float64) for p in config.FIVE]
        if all(np.abs(z).sum() > 0 for z in zz):
            rows.append(dict(cell=config.cell_name(model, task), condition=c, method="verdict", auroc=float(np.mean([estimators.auroc(yx, z[:, 0] - z[:, 1]) for z in zz]))))
    print(f"[dataset-candidates] {model} {task}", flush=True); return rows


def run_dataset_candidates():
    _all_cells(); jobs = [(m, t) for m in config.MODELS for t in _CANDIDATE_TASKS if os.path.exists(storage.path("states", model=m, task=t, condition="dataset_candidates"))]
    with Pool(_PROCS) as pool:
        out = pool.map(_dataset_candidates_cell, jobs)
    os.makedirs("results/generalization", exist_ok=True)
    pd.DataFrame([r for o in out for r in o]).to_csv("results/generalization/dataset_candidates.csv", index=False)


def _instruction_family_cell(job):
    model, task = job; So = data.load_cell(_cell_for(model, task), wordings=config.FIVE, rows=_ROWS)
    y, q, cross, n = So["y"], So["q"], So["cross"], So["n"]
    Z = data.states(model, task, "instruction_family"); assert (Z["qid"].astype(int) == q).all()
    Hn = {(c, w): Z[storage.key("G", c, w)].astype(np.float64) for c in prompts.SEVEN for w in prompts.NEWW}
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; estimators.pool(SC[key], te)

    for tr, te in So["folds"]:
        for p in config.FIVE:
            ss = StandardScaler().fit(So["H"][("yes_no", p)][tr]); X = ss.transform(So["H"][("yes_no", p)][tr]); w0 = estimators.class_mean(X, y[tr])
            Bm, _ = estimators.expression_subspace(So["H"], tr, p, q, cross); Bs, _ = np.linalg.qr((Bm / ss.scale_).T); Bs = Bs.T; wp = w0 - Bs.T @ (Bs @ w0)
            for w in prompts.NEWW:
                for c in prompts.SEVEN:
                    Zt = ss.transform(Hn[(c, w)][te]); put(("readout", c, p, w), te, Zt @ w0); put(("isolated", c, p, w), te, Zt @ wp)
        for w in prompts.NEWW:
            for c in prompts.SEVEN:
                st = StandardScaler().fit(Hn[(c, w)][tr]); Xt = st.transform(Hn[(c, w)][tr]); wt = estimators.class_mean(Xt, y[tr])
                put(("fitted_new_family", c, "-", w), te, st.transform(Hn[(c, w)][te]) @ wt)
    rows = []
    for (rd, c) in sorted({k[:2] for k in SC}):
        rows.append(dict(cell=config.cell_name(model, task), method=rd, condition=c, auroc=float(np.mean([estimators.auroc(y, v) for k, v in SC.items() if k[:2] == (rd, c)]))))
    print(f"[instruction-family] {config.cell_name(model, task)}", flush=True); return rows


def run_instruction_family():
    _all_cells(); jobs = [(m, t) for m in config.MODELS for t in config.TASKS if os.path.exists(storage.path("states", model=m, task=t, condition="instruction_family"))]
    with Pool(_PROCS) as pool:
        out = pool.map(_instruction_family_cell, jobs)
    os.makedirs("results/generalization", exist_ok=True)
    pd.DataFrame([r for o in out for r in o]).to_csv("results/generalization/instruction_family.csv", index=False)


_EXTRA_FULL = ["sentence", "one_zero", "correct_incorrect", "zero_means_correct", "false_means_correct", "no_means_correct"]


_EXTRA_LEAN = ["sentence", "no_means_correct"]


_FORMAT_WORDS = {"yes_no": ("Yes", "No"), "true_false": ("True", "False"), "a_b": ("A", "B"), "b_a": ("B", "A"), "negated": ("No", "Yes")}


_MODEL_LABEL = dict(config.MODEL_NAMES, **{"gemma-2-9b-it": "Gemma-2-9B"})


_OTHER_MODELS = {
    "gemma-2-9b": dict(t0="gemma-2-9b-it", tasks=config.TASKS, wordings=config.FIVE, rows=400, rule="split_half", full=True),
    "qwen2.5-32b": dict(t0="Qwen2.5-32B-Instruct", tasks=["triviaqa", "mmlu"], wordings=config.THREE, rows=None, rule="fixed8", full=False),
    "gemma-2-27b": dict(t0="gemma-2-27b-it", tasks=["triviaqa", "mmlu"], wordings=config.THREE, rows=None, rule="fixed8", full=False),
    "mistral-nemo-12b": dict(t0="Mistral-Nemo-Instruct-2407", tasks=["triviaqa", "mmlu"], wordings=config.FIVE, rows=400, rule="split_half", full=False),
    "ministral-8b": dict(t0="Ministral-8B-Instruct-2410", tasks=["triviaqa", "mmlu"], wordings=config.FIVE, rows=400, rule="split_half", full=False),
    "olmo-2-7b": dict(t0="OLMo-2-1124-7B-Instruct", tasks=["triviaqa", "mmlu"], wordings=config.FIVE, rows=400, rule="split_half", full=False),
}


def _other_cell_name(t0, task):
    return f"{_MODEL_LABEL[t0]} {config.TASK_NAMES[task]}"


def _other_model_states(t0, task, wordings, extra):
    H, marg = {}, {}; y = q = None
    for f in _FORMAT_WORDS:
        for p in wordings:
            fp = storage.path("final_layer", model=t0, task=task, condition=f, wording=p)
            if not os.path.exists(fp):
                continue
            Z = np.load(fp, allow_pickle=True)
            if y is None:
                y = Z["y"].astype(int); q = Z["qid"].astype(int)
            H[(f, p)] = Z["G"].astype(np.float64)
            nm = [str(x) for x in Z["token_names"]]; zz = Z["Z"].astype(np.float64); s0, s1 = _FORMAT_WORDS[f]
            marg[(f, p)] = zz[:, nm.index(s0)] - zz[:, nm.index(s1)]
    for name in extra:
        for p in wordings:
            fp = storage.path("final_layer", model=t0, task=task, condition=name, wording=p)
            if not os.path.exists(fp):
                continue
            Z = np.load(fp, allow_pickle=True); H[(name, p)] = Z["G"].astype(np.float64)
            if name != "sentence":
                zz = Z["Z"].astype(np.float64); marg[(name, p)] = np.logaddexp(zz[:, 0], zz[:, 2]) - np.logaddexp(zz[:, 1], zz[:, 3])
    X = []
    for tk in config.CROSS_TASKS:
        f1 = storage.path("states", model=t0, task=tk, condition="negated"); f0 = storage.path("states", model=t0, task=tk, condition="yes_no")
        if os.path.exists(f1) and os.path.exists(f0):
            Za = np.load(f1, allow_pickle=True); Zb = np.load(f0, allow_pickle=True)
            for p_ in [str(x) for x in Za["prompt_ids"]]:
                if f"G_{p_}" in Zb:
                    X.append(Za[f"G_{p_}"].astype(np.float64) - Zb[f"G_{p_}"].astype(np.float64))
    return dict(y=y, q=q, n=len(y), H=H, marg=marg, cross=np.concatenate(X) if X else None)


_FULL_ORDER = {}


def _full_order(task):
    if task not in _FULL_ORDER:
        full = None
        for c in config.cells():
            if c["task"] == task and c["family"] == "correctness" and c["primary"]:
                full = data.states(c["model"], task, "yes_no")["qid"].astype(int); break
        _FULL_ORDER[task] = full
    return _FULL_ORDER[task]


def _other_model_cell(name, S, rule, full):
    y, q, n, H, marg, fld, cross = (S[k] for k in ("y", "q", "n", "H", "marg", "folds", "cross"))
    pids = [p for p in S["wordings"] if all((f, p) in H for f in config.FORMATS)]
    have = [t for t in config.CONDITIONS if t != "yes_no" and all((t, p) in H for p in pids)]
    if not full:
        have = [t for t in have if t != "sentence"]
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; estimators.pool(SC[key], te)

    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ytr = y[tr]; Hy = H[("yes_no", p)]; ss = StandardScaler().fit(Hy[tr]); X = ss.transform(Hy[tr])
            Bm, _ = estimators.expression_subspace(H, tr, p, q, cross, rule=rule)
            Bs, _ = np.linalg.qr((Bm / ss.scale_).T); Bs = Bs.T
            wcm = estimators.class_mean(X, ytr); w_iso = wcm - Bs.T @ (Bs @ wcm)
            R = {"readout": wcm, "isolated": w_iso}
            if full:
                other = [ss.transform(H[("yes_no", p2)][tr]) for p2 in pids if p2 != p]
                wlr, _, _, _ = estimators.tuned_probe(X, ytr, q[tr], other)
                Xp = np.r_[ss.transform(H[("a_b", p)][tr]), ss.transform(H[("b_a", p)][tr])]; yp = np.r_[ytr, ytr]
                R["tuned_probe"] = wlr; R["pooled"] = estimators.class_mean(Xp, yp)
            for nm, w_ in R.items():
                for t in ["yes_no"] + have:
                    put((nm, t, p), te, ss.transform(H[(t, p)][te]) @ w_)
            for t in ["yes_no"] + have:
                st = StandardScaler().fit(H[(t, p)][tr]); wt = estimators.class_mean(st.transform(H[(t, p)][tr]), ytr)
                put(("fitted", t, p), te, st.transform(H[(t, p)][te]) @ wt)

    def A(nm, t):
        return float(np.mean([estimators.auroc(y, SC[(nm, t, p)]) for p in pids if (nm, t, p) in SC]))

    methods = ["readout", "isolated", "tuned_probe", "pooled", "fitted"] if full else ["readout", "isolated", "fitted"]
    rows = []
    for nm in methods:
        for t in ["yes_no"] + have:
            native = [estimators.auroc(y, marg[(t, p)]) for p in pids if (t, p) in marg]
            rows.append(dict(cell=name, method=nm, condition=t, auroc=A(nm, t), verdict_auroc=float(np.mean(native)) if native else np.nan))
    return rows


def _other_model(stem):
    spec = _OTHER_MODELS[stem]; extra = _EXTRA_FULL if spec["full"] else _EXTRA_LEAN
    rows = []
    for task in spec["tasks"]:
        S = _other_model_states(spec["t0"], task, spec["wordings"], extra)
        if spec["rows"]:
            fq = _full_order(task); q = S["q"]
            order = fq if (fq is not None and len(fq) >= len(q) and np.array_equal(fq[:len(q)], q)) else None
            fld = data.folds(S["y"], q, rows=spec["rows"], order=order)
            N = min(spec["rows"], len(S["y"]))
            S = dict(S, y=S["y"][:N], q=S["q"][:N], n=N, H={k: v[:N] for k, v in S["H"].items()}, marg={k: v[:N] for k, v in S["marg"].items()})
        else:
            fld = data.folds(S["y"], S["q"]); S = dict(S)
        S["folds"] = fld; S["wordings"] = spec["wordings"]
        rows += _other_model_cell(_other_cell_name(spec["t0"], task), S, spec["rule"], spec["full"])
        print(f"[models] {stem} {task}", flush=True)
    return rows


def run_models(model=None):
    stems = [model] if model else list(_OTHER_MODELS)
    os.makedirs("results/generalization/models", exist_ok=True)
    for stem in stems:
        rs = _other_model(stem)
        cols = ["cell", "method", "condition", "auroc", "verdict_auroc"] if _OTHER_MODELS[stem]["full"] else ["cell", "condition", "method", "auroc"]
        pd.DataFrame(rs)[cols].to_csv(f"results/generalization/models/{stem}.csv", index=False)


COMMANDS = {"tasks": run_tasks, "other-model": run_other_model, "dataset-candidates": run_dataset_candidates, "instruction-family": run_instruction_family, "models": run_models}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("command", choices=list(COMMANDS) + ["all"]); ap.add_argument("--model", default=None, help="one model (models command)")
    a = ap.parse_args()
    for c in (COMMANDS if a.command == "all" else [a.command]):
        f = COMMANDS[c]; f(model=a.model) if "model" in f.__code__.co_varnames[:f.__code__.co_argcount] else f()
