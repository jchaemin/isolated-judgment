"""GPU: steering along the readout direction and along control directions, per-row records for the steering analysis."""
import argparse
import hashlib
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import config, data, models, prompts, storage

BATCH_SIZE = 48
FORMATS = ["yes_no", "a_b", "b_a"]
LABEL = {"yes_no": "Yes/No", "a_b": "A/B", "b_a": "B/A"}
STEERED = {0.5: ["a_b", "b_a"], 0.75: ["yes_no", "a_b", "b_a"], 0.9: ["b_a"]}
CONTROL_DEPTH, CONTROLS, CONTROL_WORDINGS, CONTROL_ALPHAS = 0.75, 5, config.FIVE[:2], [-1.0, 1.0]
ALPHAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]


def out_path(out, **fields):
    return storage.path("steering", **fields) if out is None else os.path.join(out, storage.LAYOUT["steering"].format(**fields))


def blocks(model):
    for m_ in (model, getattr(model, "model", None)):
        if m_ is not None and hasattr(m_, "layers"):
            return m_.layers
        if m_ is not None and hasattr(m_, "model") and hasattr(m_.model, "layers"):
            return m_.model.layers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True); ap.add_argument("--task", required=True); ap.add_argument("--rows", type=int, default=300, help="steered rows (the first ones)")
    ap.add_argument("--out", default=None, help="root of the written records (default data/)")
    a = ap.parse_args()
    T, task = a.model.split("/")[-1], a.task
    outs = {f: out_path(a.out, model=T, task=task, depth=int(round(f * 100))) for f in STEERED}
    FR = [f for f in STEERED if not os.path.exists(outs[f])]
    if not FR:
        print("[steering] SKIP all depths exist", flush=True); return
    cell = dict(model=T, task=task, family="correctness")
    y, q = data.labels(cell, {"yes_no": data.states(T, task, "yes_no")}); y = np.asarray(y).astype(int); q = np.asarray(q); n_all = len(y)
    fds = list(data.folds(y, q)); fid = np.full(n_all, -1)
    for k, (tr, te) in enumerate(fds):
        fid[te] = k
    PT = pd.read_csv(storage.path("wordings")).set_index("wording")
    INSTR = {f: {p: str(PT.loc[p, f]) for p in config.FIVE} for f in FORMATS}
    assert all(INSTR["yes_no"][p] == prompts.WORDINGS[p] for p in config.FIVE)
    rows, _ = prompts.answer_rows(T, task); assert (np.array([r["qid"] for r in rows]) == q).all()
    ex = pd.read_csv(storage.path("answer_hashes", model=T, task=task))
    assert all(hashlib.sha256(r["answer"].encode()).hexdigest() == h for r, h in zip(rows, ex.answer_sha256))
    sel = np.arange(min(a.rows, n_all))

    model, tok = models.chat_model(a.model); assert models.uses_chat_template(tok)
    dmodel = models.final_norm(model).weight.shape[0]; layers = blocks(model)
    L = len(layers); LIDX = {f: int(round(f * L)) for f in FR}
    des, ids = config.response_tokens(cell); pair = {f: (int(ids[des[f][0]]), int(ids[des[f][1]])) for f in FORMATS}
    STATE = {"delta": None, "layer": None}; CAP = {}

    def mk_hook(li):
        def hook(_mod, _inp, outp):
            hs = outp[0] if isinstance(outp, tuple) else outp
            if CAP.get("on"):
                CAP[li] = hs[:, -1, :].detach().float().cpu().numpy()
            if STATE["delta"] is None or STATE["layer"] != li:
                return outp
            hs = hs.clone(); hs[:, -1, :] = hs[:, -1, :] + STATE["delta"].to(hs.dtype).to(hs.device)
            return (hs,) + tuple(outp[1:]) if isinstance(outp, tuple) else hs
        return hook

    handles = [layers[LIDX[f] - 1].register_forward_hook(mk_hook(LIDX[f])) for f in FR]

    def texts_for(f, p, idx):
        return [models.wrap(tok, prompts.TEMPLATE.format(q=rows[i]["question"], o=("\n\n" + rows[i]["options"]) if rows[i]["options"] else "", a=rows[i]["answer"], instr=INSTR[f][p])) for i in idx]

    t0 = time.time(); RES = {f: {} for f in FR}; CAP["on"] = True
    for p in config.FIVE:
        R = {f: np.zeros((n_all, dmodel), np.float32) for f in FR}
        for i in range(0, n_all, BATCH_SIZE):
            idx = list(range(i, min(i + BATCH_SIZE, n_all))); enc = tok(texts_for("yes_no", p, idx), return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                model(**enc, logits_to_keep=1)
            for f in FR:
                R[f][idx] = CAP[LIDX[f]]
        for f in FR:
            RES[f][p] = R[f]
    CAP["on"] = False
    print(f"[steering] {T} {task} L={L} depths {LIDX}: states {time.time()-t0:.0f}s", flush=True)

    DIRS, SIG = {}, {}; seed = zlib.crc32(f"{T}_{task}".encode()) % 2**31
    for f in FR:
        li = LIDX[f]; SIGN = []
        for p in config.FIVE:
            Gs = RES[f][p].astype(np.float64)
            for fi, (tr, te) in enumerate(fds):
                ss = StandardScaler().fit(Gs[tr]); X = ss.transform(Gs[tr]); ytr = y[tr]
                w = X[ytr == 1].mean(0) - X[ytr == 0].mean(0); v = w / ss.scale_; vr = v / np.linalg.norm(v); DIRS[(f, "readout", p, fi)] = vr
                fsc = lambda H: ((H - ss.mean_) / ss.scale_) @ w; SIGN.append(bool((fsc(Gs[tr[:5]] + 1e-2 * vr) - fsc(Gs[tr[:5]]) > 0).all()))
                if f == CONTROL_DEPTH and p in CONTROL_WORDINGS:
                    rng = np.random.default_rng(seed + 1000 * fi + li + 7919 * int(p[1:]))
                    for c in range(CONTROLS):
                        yp = rng.permutation(ytr); wp = X[yp == 1].mean(0) - X[yp == 0].mean(0); vq = wp / ss.scale_; DIRS[(f, f"shuffled_{c}", p, fi)] = vq / np.linalg.norm(vq)
                        vq = rng.standard_normal(dmodel); DIRS[(f, f"random_{c}", p, fi)] = vq / np.linalg.norm(vq)
        assert all(SIGN), "sign convention violated"
        for k, vr in DIRS.items():
            if k[0] == f:
                SIG[k] = float((RES[f][k[2]][fds[k[3]][0]] @ vr).std())
        print(f"[steering] {T} {task} depth {f} -> block {li}/{L}: sigma mean {np.mean([SIG[k] for k in SIG if k[0] == f and k[1] == 'readout']):.3f}", flush=True)

    jobs = [(f, "readout", p, fm, al) for f in FR for p in config.FIVE for fm in STEERED[f] for al in ALPHAS]
    if CONTROL_DEPTH in FR:
        jobs += [(CONTROL_DEPTH, kd, p, fm, al) for p in CONTROL_WORDINGS for c in range(CONTROLS) for kd in (f"shuffled_{c}", f"random_{c}") for fm in STEERED[CONTROL_DEPTH] for al in CONTROL_ALPHAS]
    groups = {}
    for f, kd, p, fm, al in jobs:
        groups.setdefault((p, fm), []).append((f, kd, al))
    recs = {f: [] for f in FR}; EQ = []
    for gi, ((p, fm), jl) in enumerate(groups.items()):
        pos, neg = pair[fm]; res = {j: (np.zeros(len(sel), int), np.zeros(len(sel))) for j in jl}
        for i in range(0, len(sel), BATCH_SIZE):
            idx = sel[i:i + BATCH_SIZE]; enc = tok(texts_for(fm, p, idx), return_tensors="pt", padding=True).to(model.device); ids_, am_ = enc.input_ids, enc.attention_mask
            npre = ids_.shape[1] - 1
            with torch.no_grad():
                STATE["delta"] = None; o = model(input_ids=ids_[:, :-1], attention_mask=am_[:, :-1], use_cache=True, logits_to_keep=1); cache = o.past_key_values
                if i == 0:
                    ref = model(input_ids=ids_, attention_mask=am_, logits_to_keep=1).logits[:, -1, :].float()
                for (f, kd, al) in jl:
                    cache.crop(npre)
                    STATE["delta"] = None if al == 0 else torch.tensor(np.stack([al * SIG[(f, kd, p, fid[r])] * DIRS[(f, kd, p, fid[r])] for r in idx]), dtype=torch.float32, device=model.device); STATE["layer"] = LIDX[f]
                    lg = model(input_ids=ids_[:, -1:], attention_mask=am_, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :].float(); STATE["delta"] = None
                    if i == 0 and al == 0 and kd == "readout":
                        EQ.append(float((lg - ref).abs().max()) / max(1e-6, float(ref.abs().max())))
                    top = lg.argmax(-1).cpu().numpy(); tops, ms = res[(f, kd, al)]; tops[i:i + len(idx)] = top; ms[i:i + len(idx)] = (lg[:, pos] - lg[:, neg]).cpu().numpy()
            del cache, o
        for (f, kd, al) in jl:
            tops, ms = res[(f, kd, al)]; valid = np.isin(tops, [pos, neg]); vd = np.where(valid, (tops == pos).astype(int), -1)
            for j, r in enumerate(sel):
                recs[f].append(dict(model=T, dataset=task, depth=f, format=LABEL[fm], direction=kd, row=int(r), wording=p, alpha=al, valid=bool(valid[j]), verdict=int(vd[j]), margin=float(ms[j])))
        print(f"[steering] {T} {task} {p} {LABEL[fm]} group {gi + 1}/{len(groups)} ({len(jl)} jobs)  {time.time()-t0:.0f}s", flush=True)
    print(f"[steering] {T} {task} cached path vs plain forward, relative max |dlogit| at alpha 0: {max(EQ) if EQ else float('nan'):.2e}", flush=True)
    for f in FR:
        os.makedirs(os.path.dirname(outs[f]), exist_ok=True); pd.DataFrame(recs[f]).to_csv(outs[f], index=False); print("[steering] wrote", outs[f], flush=True)
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
