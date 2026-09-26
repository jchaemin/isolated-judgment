"""Labels, loading stored states and verdict margins, first-n-rows restriction, folds."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from readout import config, storage
from readout.storage import key

FORMATS4 = ["yes_no", "true_false", "a_b", "b_a"]


def states(model, task, condition):
    return np.load(storage.path("states", model=model, task=task, condition=condition), allow_pickle=True)


def labels(cell, files):
    fs = list(files); y = files[fs[0]]["y"].astype(int); q = files[fs[0]]["qid"].astype(int)
    for f in fs:
        assert (files[f]["y"].astype(int) == y).all() and (files[f]["qid"].astype(int) == q).all(), f
    if cell["task"] != "triviaqa":
        return y, q
    lab = pd.read_csv(storage.path("triviaqa_labels", model=cell["model"]))
    assert len(lab) == len(y) and (lab.qid.values.astype(int) == q).all() and (lab.original.values.astype(int) == y).all(), "label file misaligned with the states"
    return lab.regraded.values.astype(int), q


_FULL_ORDER = {}


def _frozen_order(q, task):
    if task not in _FULL_ORDER:
        found = None
        for c in config.cells():
            if c["task"] == task and c["family"] == "correctness" and c["primary"]:
                _, q_ = labels(c, {"yes_no": states(c["model"], task, "yes_no")}); found = np.asarray(q_); break
        _FULL_ORDER[task] = found
    fq = _FULL_ORDER[task]
    return fq if (fq is not None and len(fq) >= len(q) and np.array_equal(fq[:len(q)], q)) else None


def folds(y, q, rows=None, order=None):
    base = order if order is not None else q
    full = list(GroupKFold(n_splits=5).split(np.zeros((len(base), 1)), groups=base))
    return full if rows is None else [(tr[tr < rows], te[te < rows]) for tr, te in full]


def verdict_margin(Z, meta, condition, wording):
    Zc = Z[key("Z", condition, wording)].astype(np.float64); top = Z[key("t20i", condition, wording)][:, 0]; pair = list(meta["tokens"][condition]["pair"])
    if condition == "correct_incorrect" and key("Zlower", condition, wording) in Z and "lower_pair" in meta["tokens"][condition]:
        Zl = Z[key("Zlower", condition, wording)].astype(np.float64); lp = list(meta["tokens"][condition]["lower_pair"])
        marg = np.logaddexp(Zc[:, 0], Zl[:, 0]) - np.logaddexp(Zc[:, 1], Zl[:, 1]); comp = np.isin(top, pair + lp)
    else:
        marg = Zc[:, 0] - Zc[:, 1]; comp = np.isin(top, pair)
    return marg, comp


def _restrict(S, rows):
    y, q = S["y"], S["q"]
    if rows is None:
        return dict(S, folds=folds(y, q))
    N = min(rows, len(y)); order = _frozen_order(q, S["task"])
    out = dict(S, y=y[:N], q=q[:N], n=N, folds=folds(y, q, rows=N, order=order))
    for k in ("b_yn",):
        v = out.get(k)
        if v is not None and hasattr(v, "__len__") and len(v) == len(y):
            out[k] = np.asarray(v)[:N]
    for k in ("H", "marg", "comp"):
        if isinstance(out.get(k), dict):
            out[k] = {kk: (vv[:N] if hasattr(vv, "__len__") and len(vv) == len(y) else vv) for kk, vv in out[k].items()}
    return out


def load_cell(cell, wordings, rows=None):
    model, task = cell["model"], cell["task"]
    D = {f: states(model, task, f) for f in FORMATS4}
    y, q = labels(cell, {"yes_no": D["yes_no"]}); y = np.asarray(y).astype(int); q = np.asarray(q); n = len(y)
    Mt = states(model, task, "sentence")
    nf_ = storage.path("states", model=model, task=task, condition="negated"); Nn = np.load(nf_, allow_pickle=True) if os.path.exists(nf_) else None
    H, marg, comp = {}, {}, {}
    des, ids = config.response_tokens(cell); tn = [str(t) for t in D["true_false"]["token_names"]]
    for f in FORMATS4:
        for p in wordings:
            H[(f, p)] = D[f][f"G_{p}"].astype(np.float64)
            Z = D[f][f"Z_{p}"].astype(np.float64); c_, i_ = des[f]
            marg[(f, p)] = Z[:, tn.index(c_)] - Z[:, tn.index(i_)]
            comp[(f, p)] = np.isin(D[f][f"t20i_{p}"][:, 0], [int(ids[c_]), int(ids[i_])])
    for p in wordings:
        H[("sentence", p)] = Mt[f"G_{p}"].astype(np.float64)
        if Nn is not None:
            H[("negated", p)] = Nn[f"G_{p}"].astype(np.float64)
            Zn = Nn[f"Z_{p}"].astype(np.float64)
            marg[("negated", p)] = Zn[:, tn.index("No")] - Zn[:, tn.index("Yes")]
            comp[("negated", p)] = np.isin(Nn[f"t20i_{p}"][:, 0], [int(ids["No"]), int(ids["Yes"])])
    uf = storage.path("states", model=model, task=task, condition="more_wordings"); More = np.load(uf, allow_pickle=True) if os.path.exists(uf) else None
    f2 = storage.path("states", model=model, task=task, condition="one_zero_correct_incorrect")
    if os.path.exists(f2):
        Z2 = np.load(f2, allow_pickle=True); m2 = json.load(open(storage.path("states_meta", model=model, task=task, condition="one_zero_correct_incorrect")))
        for name in ("one_zero", "correct_incorrect"):
            for p in wordings:
                if key("G", name, p) in Z2:
                    H[(name, p)] = Z2[key("G", name, p)].astype(np.float64); mm, cc = verdict_margin(Z2, m2, name, p); marg[(name, p)] = mm; comp[(name, p)] = cc
                elif More is not None and key("G", name, p) in More:
                    H[(name, p)] = More[key("G", name, p)].astype(np.float64); mm, cc = verdict_margin(More, m2, name, p); marg[(name, p)] = mm; comp[(name, p)] = cc
    f3 = storage.path("states", model=model, task=task, condition="zero_false_means_correct")
    if os.path.exists(f3):
        Z3 = np.load(f3, allow_pickle=True); m3 = json.load(open(storage.path("states_meta", model=model, task=task, condition="zero_false_means_correct")))
        for name in ("zero_means_correct", "false_means_correct"):
            for p in wordings:
                src = Z3 if key("G", name, p) in Z3 else (More if More is not None and key("G", name, p) in More else None)
                if src is not None:
                    H[(name, p)] = src[key("G", name, p)].astype(np.float64); Zz = src[key("Z", name, p)].astype(np.float64)
                    marg[(name, p)] = np.logaddexp(Zz[:, 0], Zz[:, 2]) - np.logaddexp(Zz[:, 1], Zz[:, 3])
                    pr = m3["tokens"][name]; ids4 = [int(pr["pair"][0]), int(pr["pair"][1]), int(pr["alt_pair"][0]), int(pr["alt_pair"][1])]
                    comp[(name, p)] = np.isin(src[key("t20i", name, p)][:, 0], ids4)
    for p in wordings:
        rf = storage.path("layers", model=model, task=task, condition="no_means_correct", wording=p)
        if not os.path.exists(rf) and More is not None and key("G", "no_means_correct", p) in More:
            H[("no_means_correct", p)] = More[key("G", "no_means_correct", p)].astype(np.float64)
            nm = [str(x) for x in More[key("token_names", "no_means_correct")]]; sem = [str(x) for x in More[key("semantic", "no_means_correct")]]; Zz = More[key("Z", "no_means_correct", p)].astype(np.float64)
            marg[("no_means_correct", p)] = Zz[:, nm.index(sem[0])] - Zz[:, nm.index(sem[1])]
            idr = More[key("token_ids", "no_means_correct")]; comp[("no_means_correct", p)] = np.isin(More[key("t20i", "no_means_correct", p)][:, 0], [int(idr[nm.index(sem[0])]), int(idr[nm.index(sem[1])])])
        if os.path.exists(rf):
            Zr = np.load(rf, allow_pickle=True)
            H[("no_means_correct", p)] = Zr["H"][:, -1, :].astype(np.float64)
            nm = [str(x) for x in Zr["token_names"]]; sem = [str(x) for x in Zr["semantic"]]; Zz = Zr["Z"].astype(np.float64)
            marg[("no_means_correct", p)] = Zz[:, nm.index(sem[0])] - Zz[:, nm.index(sem[1])]
            idr = Zr["token_ids"]; comp[("no_means_correct", p)] = np.isin(Zr["t20i"][:, 0], [int(idr[nm.index(sem[0])]), int(idr[nm.index(sem[1])])])
    X = []
    for tk in config.CROSS_TASKS:
        f1 = storage.path("states", model=model, task=tk, condition="negated"); f0 = storage.path("states", model=model, task=tk, condition="yes_no")
        if os.path.exists(f1) and os.path.exists(f0):
            Za = np.load(f1, allow_pickle=True); Zb = np.load(f0, allow_pickle=True)
            for p in [str(x) for x in Za["prompt_ids"]]:
                if f"G_{p}" in Zb:
                    X.append(Za[f"G_{p}"].astype(np.float64) - Zb[f"G_{p}"].astype(np.float64))
    cross = np.concatenate(X) if X else None
    b_yn = np.where(comp[("yes_no", wordings[0])], (marg[("yes_no", wordings[0])] > 0).astype(int), -1)
    S = dict(y=y, q=q, n=n, H=H, marg=marg, comp=comp, cross=cross, b_yn=b_yn, task=task)
    return _restrict(S, rows)
