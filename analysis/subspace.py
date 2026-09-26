"""The expression subspace and the held-out readouts (Sections 5 and 7, Appendix C.1 to C.5). Usage: python analysis/subspace.py <command> (or all)."""
import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import config, data, estimators as est, storage


def cell_run(cell):
    name = cell["name"]; S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, marg, cross, fld = (S[k] for k in ("y", "q", "n", "H", "marg", "cross", "folds"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]
    have = [t for t in config.CONDITIONS if t != "yes_no" and all((t, p) in H for p in pids)]
    SC = {}; units = []

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ytr = y[tr]; Hy = H[("yes_no", p)]; ss = StandardScaler().fit(Hy[tr]); X = ss.transform(Hy[tr])
            other = [ss.transform(H[("yes_no", p2)][tr]) for p2 in pids if p2 != p]
            Bm, k_B = est.expression_subspace(H, tr, p, q, cross, rule="split_half")
            Bs, _ = np.linalg.qr((Bm / ss.scale_).T); Bs = Bs.T
            wcm = est.class_mean(X, ytr); w_fijr = wcm - Bs.T @ (Bs @ wcm)
            wlr, _, _, _ = est.tuned_probe(X, ytr, q[tr], other)
            Xp = np.r_[ss.transform(H[("a_b", p)][tr]), ss.transform(H[("b_a", p)][tr])]; yp = np.r_[ytr, ytr]
            w_pool = est.class_mean(Xp, yp)
            for nm, w_ in {"readout": wcm, "isolated": w_fijr, "tuned_probe": wlr, "pooled": w_pool}.items():
                for t in ["yes_no"] + have:
                    put((nm, t, p), te, ss.transform(H[(t, p)][te]) @ w_)
            for t in ["yes_no"] + have:
                st = StandardScaler().fit(H[(t, p)][tr]); Xt = st.transform(H[(t, p)][tr]); wt = est.class_mean(Xt, ytr)
                put(("fitted", t, p), te, st.transform(H[(t, p)][te]) @ wt)
            units.append(dict(cell=name, fold=fi, wording=p, rank=k_B))

    def A(nm, t):
        vals = [est.auroc(y, SC[(nm, t, p)]) for p in pids if (nm, t, p) in SC]
        return float(np.mean(vals)) if vals else np.nan

    rows = []
    for nm in ("readout", "isolated", "tuned_probe", "pooled", "fitted"):
        for t in ["yes_no"] + have:
            native = [est.auroc(y, marg[(t, p)]) for p in pids if (t, p) in marg] if t != "yes_no" else [est.auroc(y, marg[("yes_no", p)]) for p in pids]
            rows.append(dict(cell=name, model=config.model_name(cell["model"]), method=nm, condition=t,
                              auroc=A(nm, t), verdict_auroc=float(np.mean(native)) if native else np.nan))
    print(f"[heldout] {name}", flush=True)
    return rows, units


def run_heldout():
    cells = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]
    with Pool(7) as pool:
        out = pool.map(cell_run, cells)
    os.makedirs("results/subspace", exist_ok=True)
    pd.DataFrame([r for o in out for r in o[0]]).to_csv("results/subspace/heldout.csv", index=False)
    pd.DataFrame([u for o in out for u in o[1]]).to_csv("results/subspace/ranks.csv", index=False)


def run_intervals():
    H = pd.read_csv("results/subspace/heldout.csv")
    piv = H.pivot_table(index=["cell", "condition"], columns="method", values="auroc")
    gap = (piv["isolated"] - piv["fitted"]).rename("gap").reset_index()
    rng = np.random.default_rng(0); out = []
    for t in config.HELD_OUT:
        cell = gap[gap.condition == t].gap.values
        bc = rng.integers(0, len(cell), (10000, len(cell))); cm = cell[bc].mean(1)
        out.append(dict(condition=t, interval=f"[{np.percentile(cm, 2.5):+.3f}, {np.percentile(cm, 97.5):+.3f}]"))
    os.makedirs("results/subspace", exist_ok=True)
    pd.DataFrame(out).to_csv("results/subspace/heldout_intervals.csv", index=False)


def _cells():
    return [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]


def _per_wording_cell(cell):
    name = cell["name"]; S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    pids = [p for p in config.FIVE if ("yes_no", p) in H]
    have = [t for t in config.HELD_OUT if all((t, p) in H for p in pids)]
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for fi, (tr, te) in enumerate(fld):
        for p in pids:
            ytr = y[tr]; Hy = H[("yes_no", p)]; ss = StandardScaler().fit(Hy[tr]); X = ss.transform(Hy[tr])
            Bm, _ = est.expression_subspace(H, tr, p, q, cross, rule="split_half")
            Bs, _ = np.linalg.qr((Bm / ss.scale_).T); Bs = Bs.T
            wcm = est.class_mean(X, ytr); w_fijr = wcm - Bs.T @ (Bs @ wcm)
            for t in have:
                put(("isolated", t, p), te, ss.transform(H[(t, p)][te]) @ w_fijr)
                st = StandardScaler().fit(H[(t, p)][tr]); Xt = st.transform(H[(t, p)][tr]); wt = est.class_mean(Xt, ytr)
                put(("fitted", t, p), te, st.transform(H[(t, p)][te]) @ wt)
    rows = [dict(cell=name, wording=p, condition=t, method=nm, auroc=float(est.auroc(y, s)))
            for (nm, t, p), s in SC.items()]
    print(f"[per-wording] {name}", flush=True)
    return rows


def run_per_wording():
    with Pool(14) as pool:
        out = pool.map(_per_wording_cell, _cells())
    os.makedirs("results/subspace", exist_ok=True)
    pd.DataFrame([r for o in out for r in o]).to_csv("results/subspace/heldout_per_wording.csv", index=False)


def _share_cell(cell):
    name = cell["name"]; S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, H, cross, fld = S["y"], S["q"], S["H"], S["cross"], S["folds"]
    rows = []
    for fi, (tr, te) in enumerate(fld):
        for p in config.FIVE:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); w = est.class_mean(X, y[tr])
            Bm, _ = est.expression_subspace(H, tr, p, q, cross, rule="split_half")
            Qb = np.linalg.qr((Bm / ss.scale_).T)[0]; w_in = Qb @ (Qb.T @ w)
            rows.append(dict(cell=name, fold=fi, wording=p, share=float(w_in @ w_in / (w @ w))))
    print(f"[share] {name}", flush=True)
    return rows


def run_share():
    with Pool(14) as pool:
        out = pool.map(_share_cell, _cells())
    os.makedirs("results/subspace", exist_ok=True)
    pd.DataFrame([r for o in out for r in o]).to_csv("results/subspace/readout_share.csv", index=False)


_PAIR_METHOD = {"readout": "readout", "minus_both": "isolated", "minus_formats": "isolated_format_pairs", "minus_negation": "isolated_negation_pairs", "fitted": "fitted"}


_PAIR_RANK = {"formats": "format_pairs", "negation": "negation_pairs"}


def _pair_subspaces(H, tr, p, q, cross):
    fmt = [H[(g, p)][tr] - H[(f, p)][tr] for i, f in enumerate(config.FORMATS) for g in config.FORMATS[i + 1:] if (f, p) in H and (g, p) in H]
    xb = cross[np.random.default_rng(1).choice(len(cross), min(len(cross), 4 * len(tr)), replace=False)] if cross is not None else None
    out = {}
    ka = est.split_half_rank(fmt, [q[tr]] * len(fmt)); out["formats"] = (est.top_directions(np.concatenate(fmt), ka)[0], ka)
    if xb is not None:
        kb = est.split_half_rank([xb], [np.arange(len(xb))]); out["negation"] = (est.top_directions(xb, kb)[0], kb)
        kr = est.split_half_rank(fmt + [xb], [q[tr]] * len(fmt) + [np.arange(len(xb))])
        out["both"] = (est.top_directions(np.concatenate(fmt + [xb]), kr)[0], kr)
    return out


def _pair_sources_cell(cell):
    name = cell["name"]; S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    SC = {}; shares = []

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for fi, (tr, te) in enumerate(fld):
        for p in config.FIVE:
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); w = est.class_mean(X, y[tr])
            Bs = _pair_subspaces(H, tr, p, q, cross); Wd = {"readout": w}
            for nm, (Bm, k) in Bs.items():
                Qb = np.linalg.qr((Bm / ss.scale_).T)[0]; pw = Qb @ (Qb.T @ w); Wd[f"minus_{nm}"] = w - pw
                shares.append(dict(fold=fi, wording=p, subspace=nm, k=int(k)))
            targets = set(config.HELD_OUT) if p in config.FIVE else set()
            for t in targets:
                if (t, p) not in H:
                    continue
                Zt = ss.transform(H[(t, p)][te])
                for nm, ww in Wd.items():
                    put((nm, t, p), te, Zt @ ww)
                st = StandardScaler().fit(H[(t, p)][tr]); Xt = st.transform(H[(t, p)][tr]); wt = est.class_mean(Xt, y[tr])
                put(("fitted", t, p), te, st.transform(H[(t, p)][te]) @ wt)
    rows = []
    for t in config.HELD_OUT:
        for nm in [m for m in ("readout", "minus_both", "minus_formats", "minus_negation", "fitted") if any(k[0] == m for k in SC)]:
            ps = [p for p in config.FIVE if (nm, t, p) in SC]
            if len(ps) < len(config.FIVE):
                continue
            rows.append(dict(cell=name, quantity="auroc", condition=t, method=_PAIR_METHOD[nm],
                              value=float(np.mean([est.auroc(y, SC[(nm, t, p)]) for p in ps]))))
    Sh = pd.DataFrame(shares)
    for (sub,), g in Sh[Sh.subspace.isin(("formats", "negation"))].groupby(["subspace"]):
        rows.append(dict(cell=name, quantity="mean_rank", method=_PAIR_RANK[sub], value=float(g.k.mean())))
    print(f"[pair-sources] {name}", flush=True)
    return rows


def run_pair_sources():
    with Pool(10) as pool:
        out = pool.map(_pair_sources_cell, _cells())
    os.makedirs("results/subspace", exist_ok=True)
    pd.DataFrame([r for o in out for r in o]).to_csv("results/subspace/pair_sources.csv", index=False)


def _negation_directions(cross):
    V3, _, _ = est.top_directions(cross, 3)
    return dict(rank_1=V3[:1], rank_3=V3[:3])


def _negation_pair_variants(model):
    out = {"whole": [], "agree": [], "balanced": []}
    for tk in config.CROSS_TASKS:
        Zn = np.load(storage.path("states", model=model, task=tk, condition="negated"), allow_pickle=True)
        Za = np.load(storage.path("states", model=model, task=tk, condition="yes_no"), allow_pickle=True)
        nmA = [str(x) for x in Za["token_names"]]; nmN = [str(x) for x in Zn["token_names"]]
        idA = {nm: int(i) for nm, i in zip(nmA, Za["token_ids"])}; idN = {nm: int(i) for nm, i in zip(nmN, Zn["token_ids"])}
        for p in config.FIVE:
            if f"G_{p}" not in Za or f"G_{p}" not in Zn:
                continue
            D = Zn[f"G_{p}"].astype(np.float64) - Za[f"G_{p}"].astype(np.float64)
            za = Za[f"Z_{p}"]; zn = Zn[f"Z_{p}"]
            va = np.where(np.isin(Za[f"t20i_{p}"][:, 0], [idA["Yes"], idA["No"]]), (za[:, nmA.index("Yes")] > za[:, nmA.index("No")]).astype(int), -1)
            vn = np.where(np.isin(Zn[f"t20i_{p}"][:, 0], [idN["Yes"], idN["No"]]), (zn[:, nmN.index("No")] > zn[:, nmN.index("Yes")]).astype(int), -1)
            out["whole"].append(D); ok = (va >= 0) & (vn >= 0) & (va == vn); out["agree"].append(D[ok])
            m = va >= 0; s = 2.0 * va[m] - 1; sc = s - s.mean(); Dm = D[m]
            beta = (sc[:, None] * (Dm - Dm.mean(0))).sum(0) / (sc @ sc) if sc @ sc > 0 else np.zeros(D.shape[1])
            out["balanced"].append(Dm - sc[:, None] * beta[None, :])
    return {k: np.concatenate(v) for k, v in out.items()}


def _negation_pairs_at(model, wordings):
    X = []
    for tk in config.CROSS_TASKS:
        Za = np.load(storage.path("states", model=model, task=tk, condition="negated"), allow_pickle=True)
        Zb = np.load(storage.path("states", model=model, task=tk, condition="yes_no"), allow_pickle=True)
        for p in wordings:
            if f"G_{p}" in Za and f"G_{p}" in Zb:
                X.append(Za[f"G_{p}"].astype(np.float64) - Zb[f"G_{p}"].astype(np.float64))
    return np.concatenate(X) if X else None


def _verdicts(S, p):
    V = {}
    for f in ("yes_no", "true_false", "a_b", "b_a"):
        if (f, p) in S["marg"]:
            V[f] = np.where(S["comp"][(f, p)], (S["marg"][(f, p)] > 0).astype(int), -1)
    return V


def _directions_cell(cell):
    name = cell["name"]; S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    Pz = _negation_directions(cross); SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for tr, te in fld:
        for p in config.FIVE:
            ytr = y[tr]; ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr])
            w = est.class_mean(X, ytr)
            Bm, _ = est.expression_subspace(H, tr, p, q, cross, rule="split_half")
            Q8 = np.linalg.qr((Bm / ss.scale_).T)[0]; Q1 = np.linalg.qr((Pz["rank_1"] / ss.scale_).T)[0]; Q3 = np.linalg.qr((Pz["rank_3"] / ss.scale_).T)[0]
            ws = {"readout": w, "negation_rank_1": w - Q1 @ (Q1.T @ w), "negation_rank_3": w - Q3 @ (Q3.T @ w), "subspace_rank_8": w - Q8 @ (Q8.T @ w)}
            for t in config.CONDITIONS:
                if (t, p) not in H:
                    continue
                Zt = ss.transform(H[(t, p)][te])
                for nm, ww in ws.items():
                    put((nm, t, p), te, Zt @ ww)
                st = StandardScaler().fit(H[(t, p)][tr]); Xt = st.transform(H[(t, p)][tr]); wt = est.class_mean(Xt, ytr)
                put(("fitted", t, p), te, st.transform(H[(t, p)][te]) @ wt)

    rows = []
    for key in sorted({k[:2] for k in SC}):
        nm, t = key; ps = [p for p in config.FIVE if (nm, t, p) in SC]
        rows.append(dict(cell=name, method=nm, condition=t, auroc=float(np.mean([est.auroc(y, SC[(nm, t, p)]) for p in ps]))))
    print(f"[directions] {name}", flush=True)
    return rows


def run_directions():
    with Pool(8) as pool:
        out = pool.map(_directions_cell, _cells())
    os.makedirs("results/subspace", exist_ok=True)
    df = pd.DataFrame([r for o in out for r in o])[["cell", "method", "condition", "auroc"]]
    df.to_csv("results/subspace/directions_removed.csv", index=False)


def _components_cell(cell):
    name = cell["name"]; TH = 0.95
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, H, cross, fld = (S[k] for k in ("y", "H", "cross", "folds"))
    q = S["q"]; pids = [p for p in config.FIVE if ("yes_no", p) in H]
    units = []; SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(S["n"], np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for tr, te in fld:
        for p in pids:
            ytr = y[tr]; ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr])
            w = est.class_mean(X, ytr)
            Bm, k = est.expression_subspace(H, tr, p, q, cross, rule="split_half"); Qb = np.linalg.qr((Bm / ss.scale_).T)[0]
            _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False); Qp = Vt[:k].T
            _, cosv, _ = np.linalg.svd(Qb.T @ Qp); cosv = np.clip(cosv, 0, 1)
            nw = float(w @ w); pb = Qb @ (Qb.T @ w); pp = Qp @ (Qp.T @ w)
            units.append(dict(mean_abs_cos=float(cosv.mean()), n_overlap_dirs=int((cosv >= TH).sum()), share_in_subspace=float(pb @ pb / nw), share_in_components=float(pp @ pp / nw)))
            W_ = {"readout": w, "minus_subspace": w - pb, "minus_components": w - pp}
            if ("negated", p) in H:
                Zt = ss.transform(H[("negated", p)][te])
                for nm, ww in W_.items():
                    put((nm, "negated", p), te, Zt @ ww)

    ps = [p for p in pids if ("readout", "negated", p) in SC]; base = no_sub = no_pc = float("nan")
    if ps:
        base = float(np.mean([est.auroc(y, SC[("readout", "negated", p)]) for p in ps]))
        no_sub = float(np.mean([est.auroc(y, SC[("minus_subspace", "negated", p)]) for p in ps]))
        no_pc = float(np.mean([est.auroc(y, SC[("minus_components", "negated", p)]) for p in ps]))
    g = pd.DataFrame(units).mean(numeric_only=True)
    print(f"[components] {name}", flush=True)
    return dict(cell=name, mean_abs_cos=float(g.mean_abs_cos), overlap_directions=float(g.n_overlap_dirs),
                share_in_subspace=float(g.share_in_subspace), share_in_components=float(g.share_in_components),
                negated_change_components=no_pc - base, negated_change_subspace=no_sub - base)


def run_components():
    with Pool(7) as pool:
        rows = pool.map(_components_cell, _cells())
    os.makedirs("results/subspace", exist_ok=True)
    cols = ["cell", "mean_abs_cos", "overlap_directions", "share_in_subspace", "share_in_components", "negated_change_components", "negated_change_subspace"]
    pd.DataFrame(rows)[cols].to_csv("results/subspace/principal_components.csv", index=False)


def _erasure_cell(cell):
    name = cell["name"]
    conv_order = ["yes_no", "true_false", "a_b", "b_a", "sentence", "negated", "one_zero", "correct_incorrect", "zero_means_correct", "false_means_correct", "no_means_correct"]
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, cross, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    ws = [p for p in config.FIVE if ("yes_no", p) in H]
    have = [c for c in conv_order if all((c, p) in H for p in ws)]
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for tr, te in fld:
        for p in ws:
            Xst = np.concatenate([H[(c, p)][tr] for c in have]); zst = np.concatenate([[i] * len(tr) for i in range(len(have))])
            Z = np.eye(len(have))[zst]
            mu, A = est.leace(Xst, Z); era = lambda X: X - (X - mu) @ A
            ssp = StandardScaler().fit(Xst); R = est.inlp(ssp.transform(Xst), zst, iters=8); ein = lambda X: ssp.transform(X) @ R
            ytr = y[tr]; ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); w = est.class_mean(X, ytr)
            Bm, _ = est.expression_subspace(H, tr, p, q, cross, rule="split_half"); Qb = np.linalg.qr((Bm / ss.scale_).T)[0]; wp = w - Qb @ (Qb.T @ w)
            for nm, f_ in (("leace", era), ("inlp", ein)):
                Etr = f_(H[("yes_no", p)][tr]); se = StandardScaler().fit(Etr); Xe = se.transform(Etr); we = est.class_mean(Xe, ytr)
                for c in config.HELD_OUT:
                    if c in have:
                        put((nm, c, p), te, se.transform(f_(H[(c, p)][te])) @ we)
            for c in config.HELD_OUT:
                if c in have:
                    Zt = ss.transform(H[(c, p)][te]); put(("readout", c, p), te, Zt @ w); put(("isolated", c, p), te, Zt @ wp)

    rows = []
    for key in sorted({k[:2] for k in SC}):
        nm, c = key; ps = [p for p in ws if (nm, c, p) in SC]
        rows.append(dict(cell=name, method=nm, condition=c, auroc=float(np.mean([est.auroc(y, SC[(nm, c, p)]) for p in ps]))))
    print(f"[erasure] {name}", flush=True)
    return rows


def run_erasure():
    with Pool(14) as pool:
        out = pool.map(_erasure_cell, _cells())
    os.makedirs("results/subspace", exist_ok=True)
    df = pd.DataFrame([r for o in out for r in o])[["cell", "method", "condition", "auroc"]]
    df.to_csv("results/subspace/erasure.csv", index=False)


def _verdict_free_cell(cell):
    name = cell["name"]; model, task = cell["model"], cell["task"]
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, fld = (S[k] for k in ("y", "q", "n", "H", "folds"))
    Mt = np.load(storage.path("states", model=model, task=task, condition="sentence"), allow_pickle=True)
    CX = _negation_pair_variants(model); SC = {}
    names = {"plain": "none", "agree": "identical_verdict_pairs", "balanced": "verdict_regressed_out", "whole": "all_pairs"}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for tr, te in fld:
        for p in config.FIVE:
            V = _verdicts(S, p); zd = Mt[f"Zdiv_{p}"]; V["sentence"] = (zd[:, 0] > zd[:, 1]).astype(int)
            own = V["yes_no"]; ytr = y[tr]
            ss = StandardScaler().fit(H[("yes_no", p)][tr]); X = ss.transform(H[("yes_no", p)][tr]); w = est.class_mean(X, ytr)
            blocks = {"whole": [], "agree": [], "balanced": []}; groups = {"whole": [], "agree": [], "balanced": []}
            for i, f in enumerate(config.FORMATS):
                for g in config.FORMATS[i + 1:]:
                    D = H[(g, p)][tr] - H[(f, p)][tr]; blocks["whole"].append(D); groups["whole"].append(q[tr])
                    ok = (V[f][tr] >= 0) & (V[g][tr] >= 0) & (V[f][tr] == V[g][tr]); blocks["agree"].append(D[ok]); groups["agree"].append(q[tr][ok])
                    m = own[tr] >= 0; s = 2.0 * own[tr][m] - 1; sc = s - s.mean(); Dm = D[m]
                    beta = (sc[:, None] * (Dm - Dm.mean(0))).sum(0) / (sc @ sc) if sc @ sc > 0 else np.zeros(D.shape[1])
                    blocks["balanced"].append(Dm - sc[:, None] * beta[None, :]); groups["balanced"].append(q[tr][m])
            for v in ("whole", "agree", "balanced"):
                xb = CX[v][np.random.default_rng(1).choice(len(CX[v]), min(len(CX[v]), 4 * len(tr)), replace=False)]
                bl = blocks[v] + [xb]; gr = groups[v] + [np.arange(len(xb))]
                k_rule = est.split_half_rank(bl, gr)
                Bm, _, _ = est.top_directions(np.concatenate(bl), k_rule); Qb = np.linalg.qr((Bm / ss.scale_).T)[0]; wp = w - Qb @ (Qb.T @ w)
                for t in config.HELD_OUT:
                    if (t, p) in H:
                        put((v, t, p), te, ss.transform(H[(t, p)][te]) @ wp)
            for t in config.HELD_OUT:
                if (t, p) in H:
                    put(("plain", t, p), te, ss.transform(H[(t, p)][te]) @ w)

    rows = []
    for key in sorted({k[:2] for k in SC}):
        v, t = key; ps = [p for p in config.FIVE if (v, t, p) in SC]
        rows.append(dict(cell=name, model=config.model_name(model), subspace=names[v], condition=t,
                          auroc=float(np.mean([est.auroc(y, SC[(v, t, p)]) for p in ps])), kept_pairs=np.nan))
    if task == "gsm8k":
        rows.append(dict(cell=name, model=config.model_name(model), subspace=names["agree"], condition=np.nan, auroc=np.nan, kept_pairs=int(len(CX["agree"]))))
    print(f"[verdict-free] {name}", flush=True)
    return rows


def run_verdict_free():
    with Pool(10) as pool:
        out = pool.map(_verdict_free_cell, _cells())
    os.makedirs("results/subspace", exist_ok=True)
    df = pd.DataFrame([r for o in out for r in o])[["cell", "model", "subspace", "condition", "auroc", "kept_pairs"]]
    df.to_csv("results/subspace/verdict_free.csv", index=False)


def _wordings_cell(cell):
    name = cell["name"]; model = cell["model"]
    splits = [("P01,P03,P05 -> P07,P09", ["P01", "P03", "P05"], ["P07", "P09"]), ("P07,P09 -> P01,P03,P05", ["P07", "P09"], ["P01", "P03", "P05"])]
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H, cross_all, fld = (S[k] for k in ("y", "q", "n", "H", "cross", "folds"))
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for sname, src, ev in splits:
        xs = _negation_pairs_at(model, src)
        for tr, te in fld:
            pairs = [H[(g, p)][tr] - H[(f, p)][tr] for p in src for i, f in enumerate(config.FORMATS) for g in config.FORMATS[i + 1:] if (f, p) in H and (g, p) in H]
            if xs is not None:
                pairs.append(xs[np.random.default_rng(1).choice(len(xs), min(len(xs), 4 * len(tr)), replace=False)])
            Bsrc, _, _ = est.top_directions(np.concatenate(pairs), 8)
            for e in ev:
                ytr = y[tr]; ss = StandardScaler().fit(H[("yes_no", e)][tr]); X_ = ss.transform(H[("yes_no", e)][tr]); w = est.class_mean(X_, ytr)
                Qs = np.linalg.qr((Bsrc / ss.scale_).T)[0]
                Bin, _ = est.expression_subspace(H, tr, e, q, cross_all, rule="split_half"); Qi = np.linalg.qr((Bin / ss.scale_).T)[0]
                Wd = {"readout": w, "isolated_other_wordings": w - Qs @ (Qs.T @ w), "isolated_same_wording": w - Qi @ (Qi.T @ w)}
                for t in config.HELD_OUT:
                    if (t, e) not in H:
                        continue
                    Zt = ss.transform(H[(t, e)][te])
                    for nm, ww in Wd.items():
                        put((sname, nm, t, e), te, Zt @ ww)
                    st = StandardScaler().fit(H[(t, e)][tr]); Xt = st.transform(H[(t, e)][tr]); wt = est.class_mean(Xt, ytr)
                    put((sname, "fitted", t, e), te, st.transform(H[(t, e)][te]) @ wt)

    rows = []
    for sname, src, ev in splits:
        for t in config.HELD_OUT:
            es = [e for e in ev if (sname, "readout", t, e) in SC]
            if not es:
                continue
            rec = dict(cell=name, split=sname, condition=t)
            for nm in ("readout", "isolated_other_wordings", "isolated_same_wording", "fitted"):
                rec[nm] = float(np.mean([est.auroc(y, SC[(sname, nm, t, e)]) for e in es]))
            rows.append(rec)
    print(f"[wordings] {name}", flush=True)
    return rows


def run_wordings():
    with Pool(10) as pool:
        out = pool.map(_wordings_cell, _cells())
    os.makedirs("results/subspace", exist_ok=True)
    cols = ["cell", "split", "condition", "readout", "isolated_other_wordings", "isolated_same_wording", "fitted"]
    pd.DataFrame([r for o in out for r in o])[cols].to_csv("results/subspace/heldout_wordings.csv", index=False)


def _controls_cell(cell):
    S = data.load_cell(cell, config.FIVE, rows=None)
    y, q, n, H = S["y"], S["q"], S["n"], S["H"]
    pids = [p for p in config.FIVE if all((f, p) in H for f in config.FORMATS) and ("negated", p) in H]
    SC = {}

    def put(key, te, v):
        SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

    for fi, (tr, te) in enumerate(S["folds"]):
        for p in pids:
            ytr = y[tr]; Hy = H[("yes_no", p)]; ss = StandardScaler().fit(Hy[tr]); X = ss.transform(Hy[tr])
            Hneg = H[("negated", p)]
            pairs = [H[(g, p)][tr] - H[(f, p)][tr] for i, f in enumerate(config.FORMATS) for g in config.FORMATS[i + 1:] if (f, p) in H and (g, p) in H]
            pairs.append(Hneg[tr] - Hy[tr])
            k, _ = est.choose_k(pairs, q[tr])
            wcm = est.class_mean(X, ytr)
            rng = np.random.default_rng(1000 + 10 * fi + config.FIVE.index(p))
            Rr = np.linalg.qr(rng.standard_normal((X.shape[1], k)))[0].T
            Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False); Pk = Vt[:k]
            R = {"readout": wcm, "random_removed": wcm - Rr.T @ (Rr @ wcm), "components_removed": wcm - Pk.T @ (Pk @ wcm)}
            for nm in ("readout", "random_removed"):
                put(("source", nm, p), te, ss.transform(H[("yes_no", p)][te]) @ R[nm])
            for nm, w in R.items():
                put(("negated", nm, p), te, ss.transform(Hneg[te]) @ w)

    def A(t, nm):
        return float(np.mean([est.auroc(y, SC[(t, nm, p)]) for p in pids]))

    print(f"[controls] {cell['name']}", flush=True)
    return dict(cell=cell["name"], yes_no_readout=A("source", "readout"), yes_no_random_removed=A("source", "random_removed"),
                negated_readout=A("negated", "readout"), negated_random_removed=A("negated", "random_removed"),
                negated_components_removed=A("negated", "components_removed"))


def run_controls():
    cells = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]
    with Pool(14) as pool:
        recs = pool.map(_controls_cell, cells)
    os.makedirs("results/subspace", exist_ok=True)
    cols = ["cell", "yes_no_readout", "yes_no_random_removed", "negated_readout", "negated_random_removed", "negated_components_removed"]
    pd.DataFrame(recs)[cols].to_csv("results/subspace/controls.csv", index=False)


_DEPTH_CONDITIONS = ["yes_no", "true_false", "a_b", "b_a", "sentence", "one_zero", "correct_incorrect", "negated", "no_means_correct"]


_DEPTHS = [0.25, 0.5, 0.75, 0.9, 1.0]


_DEPTH_TARGETS = ["yes_no", "b_a", "no_means_correct", "negated"]


def _depth_cell(cell):
    S = data.load_cell(cell, config.FIVE, rows=400)
    y, q, n, H0, cross0, fld = S["y"], S["q"], S["n"], S["H"], S["cross"], S["folds"]
    f = storage.path("depth", model=cell["model"], task=cell["task"])
    if not os.path.exists(f):
        return []
    Z = np.load(f, allow_pickle=True)
    rows = []
    for j, dpt in enumerate(_DEPTHS):
        X = []
        for p in config.FIVE:
            for tk in config.CROSS_TASKS:
                fa = storage.path("task_depth", model=cell["model"], task=tk, condition="yes_no", wording=p)
                fn = storage.path("task_depth", model=cell["model"], task=tk, condition="negated", wording=p)
                if dpt < 1.0 and os.path.exists(fa) and os.path.exists(fn):
                    X.append(np.load(fn, allow_pickle=True)["H"][:, j, :].astype(np.float64) - np.load(fa, allow_pickle=True)["H"][:, j, :].astype(np.float64))
        cross = (np.concatenate(X) if X else None) if dpt < 1.0 else cross0
        SC = {}

        def put(key, te, v):
            SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)

        for p in config.FIVE:
            if dpt < 1.0:
                Hd = {name: Z[storage.key("H", name, p)][:, j, :].astype(np.float64) for name in _DEPTH_CONDITIONS if storage.key("H", name, p) in Z.files}
            else:
                Hd = {name: H0[(name, p)] for name in _DEPTH_CONDITIONS if (name, p) in H0}
            for tr, te in fld:
                ss = StandardScaler().fit(Hd["yes_no"][tr]); w = est.class_mean(ss.transform(Hd["yes_no"][tr]), y[tr])
                Hp = {(c, p): v for c, v in Hd.items()}
                Bm, _ = est.expression_subspace(Hp, tr, p, q, cross, rule="split_half")
                Bs, _ = np.linalg.qr((Bm / ss.scale_).T); Bs = Bs.T
                u_out = w - Bs.T @ (Bs @ w)
                for t in _DEPTH_TARGETS:
                    if t not in Hd:
                        continue
                    Zt = ss.transform(Hd[t][te])
                    for nm, ww in (("readout", w), ("isolated", u_out)):
                        put((nm, t, p), te, Zt @ ww)
        for nm in ("readout", "isolated"):
            for t in _DEPTH_TARGETS:
                vals = [est.auroc(y, SC[(nm, t, p)]) for p in config.FIVE if (nm, t, p) in SC]
                if vals:
                    rows.append(dict(cell=cell["name"], depth=dpt, condition=t, method=nm, auroc=float(np.mean(vals))))
    print(f"[depth] {cell['name']}", flush=True)
    return rows


def run_depth():
    cells = [c for c in config.cells() if c["task"] in config.TASKS and c["primary"]]
    with Pool(7) as pool:
        out = pool.map(_depth_cell, cells)
    os.makedirs("results/subspace", exist_ok=True)
    pd.DataFrame([r for o in out for r in o]).to_csv("results/subspace/depth.csv", index=False)


COMMANDS = {"heldout": run_heldout, "intervals": run_intervals, "per-wording": run_per_wording, "share": run_share, "pair-sources": run_pair_sources, "directions": run_directions, "components": run_components, "erasure": run_erasure, "verdict-free": run_verdict_free, "wordings": run_wordings, "controls": run_controls, "depth": run_depth}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("command", choices=list(COMMANDS) + ["all"])
    a = ap.parse_args()
    for c in (COMMANDS if a.command == "all" else [a.command]):
        COMMANDS[c]()
