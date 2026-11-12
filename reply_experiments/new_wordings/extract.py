"""GPU: states for the 20 new wordings. Usage: python reply_experiments/new_wordings/extract.py --model M --task T"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from readout import config, data, models, prompts, storage
from preprocessing.extract_variants import load, cell, text, first_token, BATCH_SIZE, ROWS
HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = ["yes_no", "negated"]
NW = pd.read_csv(os.path.join(HERE, "new_wordings.csv"), dtype=str, keep_default_na=False).set_index("wording"); assert len(NW) == 20 and list(NW.index) == [f"N{i:02d}" for i in range(1, 21)]

def read(model, tok, Wu, nrm, tx, ids2, norm, name, extra=None):
    n = len(tx); G = np.zeros((n, model.config.hidden_size), np.float16); Z = np.zeros((n, 2), np.float32); top = np.zeros(n, np.int64); t0 = time.time()
    for b0 in range(0, n, BATCH_SIZE):
        idx = list(range(b0, min(b0 + BATCH_SIZE, n))); ids_, am_ = models.encode_full(tok, [tx[i] for i in idx], extra, model.device)
        with torch.no_grad():
            o = model(input_ids=ids_[:, :-1], attention_mask=am_[:, :-1], use_cache=True, logits_to_keep=1); cache = o.past_key_values; del o
            o = model(input_ids=ids_[:, -1:], attention_mask=am_, past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=1)
            lg = o.logits[:, -1, :].float(); h = o.hidden_states[-1][:, -1, :]
            if norm[0] is None: norm[0] = models.decide_norm(h, lg, Wu, nrm)
            G[idx] = models.apply_state(h, norm[0], Wu, nrm).cpu().numpy().astype(np.float16)
            Z[idx] = lg[:, ids2].cpu().numpy(); top[idx] = lg.argmax(-1).cpu().numpy()
        del o, cache
        if (b0 // BATCH_SIZE) % 10 == 0: print(f"[{name}] rows {idx[-1] + 1}/{n} {time.time()-t0:.0f}s", flush=True)
    return G, Z, top

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--wordings", default=",".join(NW.index)); a = ap.parse_args()
    T = a.model.split("/")[-1]; d = os.path.join(HERE, "out", T, a.task); path = os.path.join(d, "nwords.npz")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(path): print("exists", path); return
    D1 = data.states(T, a.task, "yes_no"); n = ROWS; q = D1["qid"].astype(int)[:n]; y = D1["y"].astype(int)
    if a.task == "triviaqa":
        lab = pd.read_csv(storage.path("triviaqa_labels", model=T)); assert (lab.qid.values.astype(int)[:n] == q).all() and (lab.original.values.astype(int)[:n] == y[:n]).all(); y = lab.regraded.values.astype(int)
    rows, _ = prompts.rows_of(T, a.task, n); assert (np.array([r["qid"] for r in rows]) == q).all()
    W = pd.read_csv(storage.path("wordings")).set_index("wording"); I = prompts.instructions(W); P01 = I[("yes_no", "P01")]
    ids = {k: int(v) for k, v in json.load(open(storage.path("response_tokens")))["tokens"][T].items()}; yn = [ids["Yes"], ids["No"]]
    model, tok, Wu, nrm = load(a.model); norm = [None]
    ft_yes = first_token(tok, "Yes"); assert ft_yes == (yn[0], []), (ft_yes, yn); ft_no = first_token(tok, "No"); assert ft_no == (yn[1], []), (ft_no, yn)
    tmpl = lambda form, p: str(NW.loc[p, form])
    info = dict(model=a.model, task=a.task, n=n, yes_no_ids=yn)
    read(model, tok, Wu, nrm, [text(tok, r, P01) for r in rows[:16]], yn, norm, "norm")
    out = dict(qid=q, y=y[:n], yes_no_ids=np.array(yn)); stats = {}; t0 = time.time(); ps = a.wordings.split(",")
    for form in FORMS:
        for p in ps:
            tx = [text(tok, r, tmpl(form, p)) for r in rows]; k = f"{form}_{p}"
            G, Z, top = read(model, tok, Wu, nrm, tx, yn, norm, k); C = np.isin(top, yn)
            out.update({f"G_{k}": G, f"Z_{k}": Z, f"top_{k}": top, f"C_{k}": C})
            stats[k] = dict(compliance=float(C.mean()), frac_yes=float((Z[:, 0] > Z[:, 1]).mean()), mean_margin=float((Z[:, 0] - Z[:, 1]).mean()), sec=time.time() - t0); print(k, stats[k], flush=True)
    tmp = path[:-4] + ".tmp.npz"; np.savez(tmp, **out); os.replace(tmp, path)
    json.dump(dict(info, norm=str(norm[0]), stats=stats), open(os.path.join(d, "nwords.json"), "w"), indent=1)
    print("wrote", d, "elapsed", time.time() - t0, flush=True)

main()
