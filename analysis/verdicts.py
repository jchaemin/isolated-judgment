"""The isolated judgment and the model's own verdicts (Section 6, Appendix D). Usage: python analysis/verdicts.py <command> (or all)."""
import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import config, data, estimators


OUT = "results/verdicts"


MISTRAL_ARC = config.cell_name("Mistral-7B-Instruct-v0.3", "arc")


HELD3 = ["negated", "false_means_correct", "no_means_correct"]


def _correctness_cells():
    return [c for c in config.cells() if c["family"] == "correctness" and c["primary"]]


def _self_labeled_cell(cell):
    S = data.load_cell(cell, wordings=config.FIVE, rows=400)
    q, n, H, marg, comp, cross = (S[k] for k in ("q", "n", "H", "marg", "comp", "cross"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]
    own = {p: np.where(comp[("yes_no", p)], (marg[("yes_no", p)] > 0).astype(int), -1) for p in pids}
    have = [t for t in config.CONDITIONS if all((t, p) in H for p in pids)]
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; estimators.pool(SC[key], te)

    for tr, te in S["folds"]:
        for p in pids:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr])
            ok = own[p][tr] >= 0
            w_self = estimators.class_mean(X[ok], own[p][tr][ok])
            Bm, _ = estimators.expression_subspace(H, tr, p, q, cross)
            Bq = np.linalg.qr((Bm / ss.scale_).T)[0].T
            W = {"readout": w_self, "isolated": w_self - Bq.T @ (Bq @ w_self)}
            for nm, w in W.items():
                for t in have:
                    put((nm, t, p), te, ss.transform(H[(t, p)][te]) @ w)
    OK = {p: own[p] >= 0 for p in pids}
    rows = []
    for nm in ("readout", "isolated"):
        for t in have:
            a_o = [estimators.auroc(own[p][OK[p]], SC[(nm, t, p)][OK[p]]) for p in pids]
            rows.append(dict(cell=cell["name"], method=nm, condition=t, auroc=float(np.mean(a_o))))
    print(f"[self-labeled] {cell['name']}: {len(have)} conditions", flush=True)
    return rows


def run_self_labeled():
    with Pool(7) as pool:
        out = pool.map(_self_labeled_cell, _correctness_cells())
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame([r for o in out for r in o]).dropna(subset=["auroc"]).to_csv(f"{OUT}/self_labeled.csv", index=False)


def _negation_cell(cell):
    S = data.load_cell(cell, wordings=config.FIVE, rows=400)
    y, q, n, H, marg, comp, cross = (S[k] for k in ("y", "q", "n", "H", "marg", "comp", "cross"))
    own = {p: np.where(comp[("yes_no", p)], (marg[("yes_no", p)] > 0).astype(int), -1) for p in config.FIVE}
    negv = {p: np.where(comp[("negated", p)], (marg[("negated", p)] > 0).astype(int), -1) for p in config.FIVE}
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; estimators.pool(SC[key], te)

    for tr, te in S["folds"]:
        for p in config.FIVE:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr])
            ok = own[p][tr] >= 0
            ws = estimators.class_mean(X[ok], own[p][tr][ok])
            Bm, _ = estimators.expression_subspace(H, tr, p, q, cross)
            Bq = np.linalg.qr((Bm / ss.scale_).T)[0].T
            wsp = ws - Bq.T @ (Bq @ ws)
            put(p, te, ss.transform(H[("negated", p)][te]) @ wsp)
    a_rows = []
    for p in config.FIVE:
        both = (own[p] >= 0) & (negv[p] >= 0); dis = both & (own[p] != negv[p]); s = SC[p]
        a_rows.append(dict(cell=cell["name"], wording=p,
                           verdicts_agree=float((own[p][both] == negv[p][both]).mean()),
                           vs_yes_no_verdict=estimators.auroc(own[p][both], s[both]),
                           vs_negated_verdict=estimators.auroc(negv[p][both], s[both]),
                           vs_yes_no_verdict_disagreeing=estimators.auroc(own[p][dis], s[dis]) if dis.sum() > 20 and len(np.unique(own[p][dis])) > 1 else np.nan))
    b5_rows = []
    for t in config.HELD_OUT:
        ps = [p for p in config.FIVE if (t, p) in H and (t, p) in marg]
        if not ps:
            continue
        rate = float(np.mean([(comp[(t, p)] & ((marg[(t, p)] > 0).astype(int) == y)).mean() for p in ps]))
        b5_rows.append(dict(cell=cell["name"], condition=t, correct_token_rate=rate))

    Sall = data.load_cell(cell, wordings=config.THREE, rows=None)
    y3, n3, H3, marg3, comp3 = (Sall[k] for k in ("y", "n", "H", "marg", "comp"))
    SC3 = {}

    def put3(key, te, v):
        SC3.setdefault(key, np.full(n3, np.nan)); SC3[key][te] = v; estimators.pool(SC3[key], te)

    for tr, te in Sall["folds"]:
        for p in config.THREE:
            ss = StandardScaler().fit(H3[("yes_no", p)][tr]); X = ss.transform(H3[("yes_no", p)][tr]); ytr = y3[tr]
            w = estimators.class_mean(X, ytr)
            for t in ["yes_no"] + HELD3:
                if (t, p) in H3:
                    put3((t, p), te, ss.transform(H3[(t, p)][te]) @ w)
    yn_auc = float(np.mean([estimators.auroc(y3, SC3[("yes_no", p)]) for p in config.THREE]))
    b3_rows = []
    for t in HELD3:
        ps = [p for p in config.THREE if (t, p) in SC3 and (t, p) in marg3]
        if not ps:
            continue
        auc = float(np.mean([estimators.auroc(y3, SC3[(t, p)]) for p in ps]))
        rate = float(np.mean([(comp3[(t, p)] & ((marg3[(t, p)] > 0).astype(int) == y3)).mean() for p in ps]))
        b3_rows.append(dict(cell=cell["name"], condition=t, readout_drop=yn_auc - auc, correct_token_rate=rate))
    print(f"[negation] {cell['name']}", flush=True)
    return a_rows, b5_rows, b3_rows


def run_negation():
    with Pool(10) as pool:
        out = pool.map(_negation_cell, _correctness_cells())
    os.makedirs(OUT, exist_ok=True)
    A = pd.DataFrame([r for o in out for r in o[0]]); A = A[A.cell != MISTRAL_ARC]
    A.to_csv(f"{OUT}/negated_verdicts.csv", index=False)
    pd.DataFrame([r for o in out for r in o[1]]).to_csv(f"{OUT}/correct_token_rate.csv", index=False)
    pd.DataFrame([r for o in out for r in o[2]]).to_csv(f"{OUT}/drop_vs_rate.csv", index=False)


COMMANDS = {"self-labeled": run_self_labeled, "negation": run_negation}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("command", choices=list(COMMANDS) + ["all"])
    a = ap.parse_args()
    for c in (COMMANDS if a.command == "all" else [a.command]):
        COMMANDS[c]()
