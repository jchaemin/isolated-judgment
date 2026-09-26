"""The readout: response formats, prompt position, steering and regularization (Section 4, Appendix B). Usage: python analysis/readout.py <command> (or all)."""
import glob
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import config, data, estimators as est, storage


P3 = ["P01", "P04", "P07"]


def _cm_auroc(H, y, fld):
    S = np.zeros(len(y))
    for tr, te in fld:
        ss = StandardScaler().fit(H[tr]); X = ss.transform(H[tr]); w = est.class_mean(X, y[tr])
        S[te] = ss.transform(H[te]) @ w; est.pool(S, te)
    return float(est.rank_auroc(y, S))


def _position_cell(cell):
    model, task = cell["model"], cell["task"]
    S = data.load_cell(cell, config.TEN)
    y, H, fld, n = S["y"], S["H"], S["folds"], S["n"]
    pids = [p for p in config.TEN if ("yes_no", p) in H]
    Z = {p: np.load(storage.path("position", model=model, task=task, wording=p), allow_pickle=True) for p in P3}
    pre = Z["P01"]["H_yn"][:, 1, :].astype(np.float64)
    stem = Z["P01"]["H_yn"][:, 2, :].astype(np.float64)
    a_pre = _cm_auroc(pre, y, fld)
    a_stem = _cm_auroc(stem, y, fld)
    a_post = float(np.mean([_cm_auroc(H[("yes_no", p)], y, fld) for p in pids]))
    fitted = dict(cell=cell["name"], pre=a_pre, stem=a_stem, post=a_post, post_minus_pre=a_post - a_pre)

    sem = {p: Z[p]["H_yn"][:, 2, :].astype(np.float64) for p in P3}
    S_pre = np.zeros(n); S_sem = {p: np.zeros(n) for p in P3}; S_post = {p: np.zeros(n) for p in pids}
    for tr, te in fld:
        ss = StandardScaler().fit(pre[tr]); X = ss.transform(pre[tr]); w = est.class_mean(X, y[tr])
        S_pre[te] = ss.transform(pre[te]) @ w; est.pool(S_pre, te)
        for p in P3:
            S_sem[p][te] = ss.transform(sem[p][te]) @ w; est.pool(S_sem[p], te)
        for p in pids:
            S_post[p][te] = ss.transform(H[("yes_no", p)][te]) @ w; est.pool(S_post[p], te)
    A_pre = float(est.rank_auroc(y, S_pre))
    A_sem = float(np.mean([est.rank_auroc(y, S_sem[p]) for p in P3]))
    A_post = float(np.mean([est.rank_auroc(y, S_post[p]) for p in pids]))
    frozen = dict(cell=cell["name"], pre=A_pre, stem=A_sem, post=A_post)
    print(f"[position] {cell['name']} done", flush=True)
    return fitted, frozen


def run_position():
    cells = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]
    with Pool(6) as pool:
        out = pool.map(_position_cell, cells)
    os.makedirs("results/readout", exist_ok=True)
    fitted = pd.DataFrame([o[0] for o in out])
    fitted.to_csv("results/readout/position.csv", index=False)
    v = fitted.post_minus_pre.values
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10000)])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    json.dump(dict(post_minus_pre_interval=ci), open("results/readout/position_interval.json", "w"), indent=1)
    pd.DataFrame([o[1] for o in out]).to_csv("results/readout/position_frozen.csv", index=False)


POS3 = {"PRE": 1, "STEM": 2, "POST": 3}


FORMATS3 = ["yes_no", "a_b", "b_a"]


NEED = [("yes_no", "STEM"), ("yes_no", "POST"), ("a_b", "PRE"), ("a_b", "STEM"), ("b_a", "PRE"), ("b_a", "STEM")]


def _format_first_cell(cell):
    model, task = cell["model"], cell["task"]
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, fld, n = S["y"], S["folds"], S["n"]
    FF = {}
    for p in config.FIVE:
        got = {}
        for f in FORMATS3:
            fp = storage.path("format_first", model=model, task=task, condition=f, wording=p)
            if os.path.exists(fp):
                got[f] = np.load(fp, allow_pickle=True)
        if len(got) == len(FORMATS3):
            FF[p] = got
    pids = list(FF)
    Sc = {(pos, p): np.zeros(n) for pos in ("STEM", "POST") for p in pids}
    ident = {(pos, p): np.zeros(n) for pos in ("PRE", "STEM") for p in pids}
    for tr, te in fld:
        for p in pids:
            HH = {(f, pos): FF[p][f]["H"][:, POS3[pos], :].astype(np.float64) for f, pos in NEED}
            for pos in ("STEM", "POST"):
                ss = StandardScaler().fit(HH[("yes_no", pos)][tr]); X = ss.transform(HH[("yes_no", pos)][tr]); w = est.class_mean(X, y[tr])
                Sc[(pos, p)][te] = ss.transform(HH[("yes_no", pos)][te]) @ w; est.pool(Sc[(pos, p)], te)
            for pos in ("PRE", "STEM"):
                Xa, Xb = HH[("a_b", pos)], HH[("b_a", pos)]
                Xs = np.r_[Xa[tr], Xb[tr]]; lab = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
                ss = StandardScaler().fit(Xs); Zc = ss.transform(Xs); wf = est.class_mean(Zc, lab)
                sa, sb = ss.transform(Xa[te]) @ wf, ss.transform(Xb[te]) @ wf
                ident[(pos, p)][te] = float(est.rank_auroc(np.r_[np.ones(len(te)), np.zeros(len(te))], np.r_[sa, sb]))
    stem = float(np.mean([est.rank_auroc(y, Sc[("STEM", p)]) for p in pids]))
    post = float(np.mean([est.rank_auroc(y, Sc[("POST", p)]) for p in pids]))
    pre_ident = float(np.mean([float(np.mean(ident[("PRE", p)])) for p in pids]))
    stem_ident = float(np.mean([float(np.mean(ident[("STEM", p)])) for p in pids]))
    print(f"[format_first] {cell['name']} done", flush=True)
    return dict(cell=cell["name"], primary=cell["primary"], stem=stem, response_position=post, end_of_candidate=pre_ident, after_instruction=stem_ident)


def run_format_first():
    cells = [c for c in config.cells() if c["task"] in config.TASKS]
    with Pool(16) as pool:
        rows = pool.map(_format_first_cell, cells)
    os.makedirs("results/readout", exist_ok=True)
    pd.DataFrame(rows)[["cell", "stem", "response_position"]].to_csv("results/readout/format_first.csv", index=False)
    prim = [r for r in rows if r["primary"]]
    pd.DataFrame(prim)[["cell", "end_of_candidate", "after_instruction"]].to_csv("results/readout/condition_identity.csv", index=False)


FORDER = ["Yes/No", "A/B", "B/A"]


KEEP = [(0.5, ["A/B", "B/A"]), (0.75, FORDER), (0.9, ["B/A"])]


FMT_NAME = {"Yes/No": "yes_no", "A/B": "a_b", "B/A": "b_a"}


STEER_COLS = ["model", "dataset", "depth", "format", "direction", "row", "wording", "alpha", "valid", "verdict", "margin"]


def _slope(rd):
    piv = rd[rd.alpha.abs() <= 1].pivot_table(index=["row", "wording"], columns="alpha", values="margin").dropna()
    al = np.array(piv.columns, float); Xc = al - al.mean()
    return float(((piv.values @ Xc) / (Xc @ Xc)).mean())


def run_steering():
    files = sorted(glob.glob(storage.path("steering", model="*", task="*", depth="*")))
    R = pd.concat([pd.read_csv(f, usecols=STEER_COLS) for f in files], ignore_index=True)
    cells = [c for c in config.cells() if c["task"] in config.TASKS]
    meta = pd.DataFrame([dict(model=c["model"], dataset=c["task"], cell=c["name"], primary=c["primary"]) for c in cells])
    R = R.merge(meta, on=["model", "dataset"], how="inner")
    R = R[R.primary & R.format.isin(FORDER)].copy()

    sem, ctrl_s, ctrl_r = {}, {}, {}
    for blk, gb in R.groupby("cell"):
        for dep, fmts in KEEP:
            gd = gb[gb.depth == dep]
            for fmt in fmts:
                g = gd[gd.format == fmt]
                sem[(blk, dep, fmt)] = _slope(g[g.direction == "readout"])
                if dep == 0.75:
                    for kd, gc in g[g.direction.str.startswith(("shuffled", "random"))].groupby("direction"):
                        d = gc.groupby("alpha").margin.mean()
                        if 1.0 in d and -1.0 in d:
                            (ctrl_s if kd.startswith("shuffled") else ctrl_r).setdefault((dep, fmt), []).append(float((d[1.0] - d[-1.0]) / 2))
    blocks = sorted(meta[meta.primary].cell)

    os.makedirs("results/readout", exist_ok=True)
    pd.DataFrame([dict(cell=b, depth=0.75, format=FMT_NAME[f], slope=sem[(b, 0.75, f)]) for b in blocks for f in FORDER]
                 ).to_csv("results/readout/steering_cells.csv", index=False)

    rows = []
    for dep, fmts in KEEP:
        for f in fmts:
            vals = [sem[(b, dep, f)] for b in blocks]
            rec = dict(depth=dep, format=FMT_NAME[f], slope=float(np.mean(vals)), cells=np.nan, cells_positive=np.nan, shuffled_slope=np.nan, random_slope=np.nan)
            if dep == 0.75:
                rec["cells"] = float(len(vals)); rec["cells_positive"] = float(sum(v > 0 for v in vals))
                rec["shuffled_slope"] = float(np.mean(ctrl_s[(dep, f)])) if ctrl_s.get((dep, f)) else np.nan
                rec["random_slope"] = float(np.mean(ctrl_r[(dep, f)])) if ctrl_r.get((dep, f)) else np.nan
            rows.append(rec)
    pd.DataFrame(rows).to_csv("results/readout/steering.csv", index=False)

    R75 = R[(R.depth == 0.75) & (R.direction == "readout")]
    comp = []
    for blk, gb in R75.groupby("cell"):
        for fmt in FORDER:
            d = gb[(gb.format == fmt) & (gb.alpha.abs() <= 1)]
            ok = d.groupby(["row", "wording"]).agg(v=("valid", "all"), k=("alpha", "nunique"))
            idx = ok[ok.v & (ok.k == 7)].index
            dcx = d.set_index(["row", "wording"]).loc[idx].reset_index()
            rate = lambda x, a: float((x[(x.alpha == a) & (x.verdict >= 0)].verdict == 1).mean())
            comp.append(dict(cell=blk, format=FMT_NAME[fmt], slope=_slope(dcx), correct_rate_change=rate(dcx, 1.0) - rate(dcx, -1.0)))
    pd.DataFrame(comp).to_csv("results/readout/steering_compliant.csv", index=False)
    print("[steering] wrote steering.csv, steering_cells.csv, steering_compliant.csv", flush=True)


def _reg_cell(cell):
    S = data.load_cell(cell, config.TEN)
    y, H, fld = S["y"], S["H"], S["folds"]
    out = {}
    for C in est.GRID:
        diag = np.zeros((len(config.TEN), len(y))); ab = np.zeros_like(diag); ba = np.zeros_like(diag)
        for tr, te in fld:
            for pi, p in enumerate(config.TEN):
                Xs = H[("yes_no", p)]; ss = StandardScaler().fit(Xs[tr])
                m = LogisticRegression(C=C, max_iter=3000).fit(ss.transform(Xs[tr]), y[tr])
                diag[pi, te] = m.decision_function(ss.transform(Xs[te])); est.pool(diag[pi], te)
                ab[pi, te] = m.decision_function(ss.transform(H[("a_b", p)][te])); est.pool(ab[pi], te)
                ba[pi, te] = m.decision_function(ss.transform(H[("b_a", p)][te])); est.pool(ba[pi], te)
        ad = np.array([est.rank_auroc(y, diag[i]) for i in range(len(config.TEN))])
        aab = np.array([est.rank_auroc(y, ab[i]) for i in range(len(config.TEN))])
        aba = np.array([est.rank_auroc(y, ba[i]) for i in range(len(config.TEN))])
        out[C] = dict(source_auroc=float(ad.mean()), a_b_drop=float(ad.mean() - aab.mean()), b_a_drop=float(ad.mean() - aba.mean()))
    print(f"[regularization] {cell['name']} done", flush=True)
    return out


def run_regularization():
    cells = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]
    with Pool(len(cells)) as pool:
        out = pool.map(_reg_cell, cells)
    order = sorted(est.GRID)
    rows = [dict(C=C, source_auroc=float(np.mean([o[C]["source_auroc"] for o in out])),
                 a_b_drop=float(np.mean([o[C]["a_b_drop"] for o in out])),
                 b_a_drop=float(np.mean([o[C]["b_a_drop"] for o in out]))) for C in order]
    os.makedirs("results/readout", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/readout/regularization_path.csv", index=False)


COMMANDS = {"position": run_position, "format-first": run_format_first, "steering": run_steering, "regularization": run_regularization}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("command", choices=list(COMMANDS) + ["all"])
    a = ap.parse_args()
    for c in (COMMANDS if a.command == "all" else [a.command]):
        COMMANDS[c]()
