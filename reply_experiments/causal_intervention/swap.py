"""GPU: swapping the expression component between ordinary and reversed prompts. Usage: python reply_experiments/causal_intervention/swap.py --model M --task T --rows 200"""
import argparse, json, os, sys, time
CUDA = os.environ.get("CUDA_VISIBLE_DEVICES", "")
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "6"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from estimators import stack, svd_top, rank_from, masks_coord, config, data, est, storage, KMAX
from readout import models, prompts
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
WORD = ["P01", "P05"]; REV = ["no_means_correct", "negated"]; RUN = ["negated"]; J = -1


def manuscript_basis(H, tr, p, q, cr=None, gid=None, ct=None, sid=None, fmts=config.FORMATS):
    pairs = [H[(g, p)][tr] - H[(f, p)][tr] for i, f in enumerate(fmts) for g in fmts[i + 1:] if (f, p) in H and (g, p) in H]; nf = len(pairs)
    bl = pairs + ([cr] if cr is not None else []); ids = [None] * nf + ([sid] if cr is not None else [])
    ms = masks_coord(q[tr], nf, gid, ct, 0)
    V1 = svd_top(stack([b[m] for b, m in zip(bl, ms)], [None if s is None else s[m] for s, m in zip(ids, ms)], "block"), KMAX)
    V2 = svd_top(stack([b[~m] for b, m in zip(bl, ms)], [None if s is None else s[~m] for s, m in zip(ids, ms)], "block"), KMAX)
    k = rank_from(V1, V2, "pa_largest"); return svd_top(stack(bl, ids, "block"), k)


def depth_cross(T, j):
    X, ti, qd, wd = [], [], [], []
    for p in config.FIVE:
        for it, tk in enumerate(config.CROSS_TASKS):
            a = np.load(storage.path("task_depth", model=T, task=tk, condition="yes_no", wording=p), allow_pickle=True)["H"]
            b = np.load(storage.path("task_depth", model=T, task=tk, condition="negated", wording=p), allow_pickle=True)["H"]
            X.append(b[:, j, :].astype(np.float64) - a[:, j, :].astype(np.float64)); qid = np.load(storage.path("states", model=T, task=tk, condition="yes_no"), allow_pickle=True)["qid"].astype(np.int64)
            assert len(qid) == len(a); ti.append(np.full(len(a), it)); qd.append(qid); wd.append(np.full(len(a), int(p[1:])))
    return np.concatenate(X), np.concatenate(ti), np.concatenate(qd), np.concatenate(wd)


def sub_cross(cr, ti, qd, wd, ntr):
    idx = np.random.default_rng(1).choice(len(cr), min(len(cr), 4 * ntr), replace=False)
    return cr[idx], ti[idx] * 10**8 + qd[idx], ti[idx], ti[idx] * 100 + wd[idx]


ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--task", required=True); ap.add_argument("--rows", type=int, default=200); ap.add_argument("--batch", type=int, default=8); a = ap.parse_args()
T = a.model.split("/")[-1]; tag = f"{T}_{a.task}"; OUT = os.path.join(HERE, "cells"); os.makedirs(OUT, exist_ok=True)
if os.path.exists(f"{OUT}/swap_{tag}.parquet"):
    print("SKIP", tag); raise SystemExit(0)
t0 = time.time(); cell = [c for c in config.cells() if c["model"] == T and c["task"] == a.task and c["family"] == "correctness"][0]
S = data.load_cell(cell, config.FIVE, rows=400); y, q, H, fld = S["y"], S["q"], S["H"], S["folds"]; n = S["n"]
Dz = np.load(storage.path("depth", model=T, task=a.task), allow_pickle=True); assert np.array_equal(Dz["qid"].astype(int)[:n], q[:n].astype(int)); LI = [int(x) for x in Dz["layers"]]; NL = int(Dz["n_layers"]); L = LI[J]
fid = np.full(n, -1)
for k, (_, te) in enumerate(fld): fid[te] = k
assert (fid >= 0).all(); sel = np.arange(min(a.rows, n))
BAS = {}; RB = np.load(os.path.join(HERE, "random_bases", f"{tag}.npz"))
cr, ti, qd, wd = depth_cross(T, J)
Hd = {(c, p): Dz[storage.key("H", c, p)][:, J, :].astype(np.float64) for p in WORD for c in config.FORMATS}
for fi, (tr, _) in enumerate(fld):
    for p in WORD:
        c_, g_, ct_, s_ = sub_cross(cr, ti, qd, wd, len(tr))
        B = dict(manuscript=manuscript_basis(Hd, tr, p, q, c_, g_, ct_, s_))
        B["manuscript"], _r = np.linalg.qr(B["manuscript"].T); B["manuscript"] = B["manuscript"].T; assert np.isfinite(B["manuscript"]).all()
        for s_i in range(3):
            B[f"haar{s_i}_manuscript"] = RB[f"f{fi}_{p}_haar{s_i}_manuscript"]; assert B[f"haar{s_i}_manuscript"].shape == B["manuscript"].shape
        BAS[(fi, p)] = {k: torch.tensor(v, dtype=torch.float32) for k, v in B.items()}
RD = {}
for fi, (tr, _) in enumerate(fld):
    for p in WORD:
        ss = StandardScaler().fit(H[("yes_no", p)][tr]); w = est.class_mean(ss.transform(H[("yes_no", p)][tr]), y[tr]); r = {"plain": (ss.mean_, ss.scale_, w)}
        for c in REV:
            s2 = StandardScaler().fit(H[(c, p)][tr]); r["own_" + c] = (s2.mean_, s2.scale_, est.class_mean(s2.transform(H[(c, p)][tr]), y[tr]))
        RD[(fi, p)] = r
print(f"[fit] {time.time()-t0:.0f}s", flush=True)

PT = pd.read_csv(storage.path("wordings")).set_index("wording"); INSTR = prompts.instructions(PT, config.FIVE); rows, _ = prompts.rows_of(T, a.task, 0); assert (np.array([r["qid"] for r in rows])[:n] == q[:n]).all()
model, tok = models.chat_model(a.model); Wu = model.get_output_embeddings().weight; nrm = models.final_norm(model); dev = model.device
_, ids = config.response_tokens(cell)
layers = None
for m_ in (model, getattr(model, "model", None)):
    if m_ is not None and hasattr(m_, "layers"): layers = m_.layers; break
    if m_ is not None and hasattr(m_, "model") and hasattr(m_.model, "layers"): layers = m_.model.layers; break
assert len(layers) == NL
ST = {"cap": None, "lay": None, "delta": None}


def mk(li):
    def hook(_m, _i, outp):
        hs = outp[0] if isinstance(outp, tuple) else outp
        if ST["cap"] is not None: ST["cap"][li] = hs[:, -1, :].detach().float()
        if ST["lay"] != li or ST["delta"] is None: return outp
        hs = hs.clone(); o_ = hs[:, -1, :].float(); dl_ = ST["delta"].to(hs.device); hs[:, -1, :] = (o_ + dl_).to(hs.dtype)
        return (hs,) + tuple(outp[1:]) if isinstance(outp, tuple) else hs
    return hook


layers[L - 1].register_forward_hook(mk(L))
NORM = [None]


def enc_prefix(texts):
    e = tok(texts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        o = model(input_ids=e.input_ids[:, :-1], attention_mask=e.attention_mask[:, :-1], use_cache=True, logits_to_keep=1)
    return dict(last=e.input_ids[:, -1:], mask=e.attention_mask, cache=o.past_key_values, npre=e.input_ids.shape[1] - 1)


def last(pf, lay=None, delta=None, cap=False):
    ST.update(cap={} if cap else None, lay=lay, delta=delta)
    pf["cache"].crop(pf["npre"])
    with torch.no_grad():
        o = model(input_ids=pf["last"], attention_mask=pf["mask"], past_key_values=pf["cache"], use_cache=True, output_hidden_states=True, logits_to_keep=1)
        lg = o.logits[:, -1, :].float(); h = o.hidden_states[-1][:, -1, :]
        if NORM[0] is None: NORM[0] = models.decide_norm(h, lg, Wu, nrm)
        g = models.apply_state(h, NORM[0], Wu, nrm).cpu().numpy().astype(np.float64)
    c = ST["cap"]; ST.update(cap=None, lay=None, delta=None); del o
    return lg, g, c


def arms(d, Bs):
    P = lambda B, v: (v @ B.T) @ B
    out = {"noop": 0 * d, "full": d}
    Pd = P(Bs["manuscript"], d); N = Pd.norm(dim=1, keepdim=True)
    out["E_manuscript_a1.0"] = Pd
    for s in range(3):
        v = P(Bs[f"haar{s}_manuscript"], d); out[f"haar{s}_manuscript"] = v * N / (v.norm(dim=1, keepdim=True) + 1e-9)
    cp = d - Pd; out["comp_scaled_manuscript"] = cp * N / (cp.norm(dim=1, keepdim=True) + 1e-9)
    return out


ROWS = []; np.random.seed(0)


def frame():
    names_ = sorted(RD[(0, WORD[0])])
    D_ = pd.DataFrame(ROWS, columns=["row", "fold", "y", "wording", "rev", "recipient", "arm", "top"] + ["s_" + n_ for n_ in names_]); D_["cell"] = cell["name"]
    return D_


for fi in range(len(fld)):
    idx_f = [int(i) for i in sel if fid[i] == fi]
    for b0 in range(0, len(idx_f), a.batch):
        idx = idx_f[b0:b0 + a.batch]
        for p in WORD:
            txt = {c: [models.wrap(tok, prompts.TEMPLATE.format(q=rows[i]["question"], o=("\n\n" + rows[i]["options"]) if rows[i]["options"] else "", a=rows[i]["answer"], instr=INSTR[(c, p)])) for i in idx] for c in ["yes_no"] + RUN}
            PF = {c: enc_prefix(txt[c]) for c in txt}; base = {c: last(PF[c], cap=True) for c in PF}
            for rev in RUN:
                for R, Dn in (("yes_no", rev), (rev, "yes_no")):
                    li = L; hR, hD = base[R][2][li], base[Dn][2][li]; d = hD - hR
                    Bs = {k: v.to(d.device) for k, v in BAS[(fi, p)].items()}; A = arms(d, Bs)
                    if R == "yes_no": A = {k: A[k] for k in ("noop", "E_manuscript_a1.0")}
                    for arm, delta in A.items():
                        lg, g, _ = last(PF[R], li, delta)
                        top = lg.argmax(-1).cpu().numpy()
                        r = RD[(fi, p)]; sc = {nm: ((g - v[0]) / v[1]) @ v[2] for nm, v in r.items()}
                        for i_, row in enumerate(idx):
                            ROWS.append((row, fi, int(y[row]), p, rev, R, arm, int(top[i_])) + tuple(float(sc[nm][i_]) for nm in sorted(sc)))
            del PF, base
    print(f"[run] fold {fi} {len(ROWS)} recs {time.time()-t0:.0f}s", flush=True)

frame().to_parquet(f"{OUT}/swap_{tag}.parquet")
