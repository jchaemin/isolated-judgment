"""GPU: free-text critiques and the model's Yes/No/Unclear reading of them. Usage: python reply_experiments/critiques/extract.py --stage gen --model M, then --stage judge --judge M"""
import os, sys
import argparse, json, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from readout import config, models, prompts, storage
from preprocessing.extract_variants import load, first_token
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); BATCH = 16
TASKS = ["gsm8k", "triviaqa"]; LO, HI = 400, 800; MAXNEW = 256
CRIT = "Question: {q}{o}\n\nProposed answer: {a}\n\nReview this answer and explain whether it is right."
JUDGE = ("Below are a question, a proposed answer, and a review of that answer.\n\nQuestion: {q}{o}\n\nProposed answer: {a}\n\nReview:\n{c}\n\n"
         "Does the review say that the proposed answer is correct? Reply with exactly one word: Yes, No, or Unclear.")

def read(model, tok, Wu, nrm, tx, ids2, norm, name, bs=BATCH):
    n = len(tx); G = np.zeros((n, model.config.hidden_size), np.float16); Z = np.zeros((n, len(ids2)), np.float32); top = np.zeros(n, np.int64); t0 = time.time()
    for b0 in range(0, n, bs):
        idx = list(range(b0, min(b0 + bs, n))); ids_, am_ = models.encode_full(tok, [tx[i] for i in idx], None, model.device)
        with torch.no_grad():
            o = model(input_ids=ids_[:, :-1], attention_mask=am_[:, :-1], use_cache=True, logits_to_keep=1); cache = o.past_key_values; del o
            o = model(input_ids=ids_[:, -1:], attention_mask=am_, past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=1)
            lg = o.logits[:, -1, :].float(); h = o.hidden_states[-1][:, -1, :]
            if norm[0] is None: norm[0] = models.decide_norm(h, lg, Wu, nrm)
            G[idx] = models.apply_state(h, norm[0], Wu, nrm).cpu().numpy().astype(np.float16)
            Z[idx] = lg[:, ids2].cpu().numpy(); top[idx] = lg.argmax(-1).cpu().numpy()
        del o, cache
        if (b0 // bs) % 25 == 0: print(f"[{name}] rows {idx[-1] + 1}/{n} {time.time()-t0:.0f}s", flush=True)
    return G, Z, top, time.time() - t0

def rows(T, task):
    R, _ = prompts.rows_of(T, task, HI); R = R[LO:HI]; assert len(R) == HI - LO, len(R)
    return R

def opts(r): return ("\n\n" + r["options"]) if r["options"] else ""

def chat(tok, user, asst=None):
    m = [{"role": "user", "content": user}]
    t = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    return t if asst is None else t + asst

def gen(model, tok, tx, name):
    out, nnew, trunc = [], [], []; eos = set([tok.eos_token_id] + list(np.atleast_1d(model.generation_config.eos_token_id or [])))
    assert tok.padding_side == "left", tok.padding_side
    for b0 in range(0, len(tx), BATCH):
        enc = tok(tx[b0:b0 + BATCH], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for s in o[:, enc.input_ids.shape[1]:].tolist():
            k = next((i for i, t in enumerate(s) if t in eos), None)
            body = s if k is None else s[:k]; out.append(tok.decode(body, skip_special_tokens=True).strip()); nnew.append(len(body)); trunc.append(k is None)
        if (b0 // BATCH) % 5 == 0: print(f"[{name}] gen {min(b0 + BATCH, len(tx))}/{len(tx)}", flush=True)
    return out, nnew, trunc

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["gen", "judge"], required=True); ap.add_argument("--model"); ap.add_argument("--judge"); a = ap.parse_args()
    info = {}
    if a.stage == "gen":
        T = a.model.split("/")[-1]; d = os.path.join(OUT, T); os.makedirs(d, exist_ok=True); model, tok, Wu, nrm = load(a.model); norm = [None]
        ids = {k: int(v) for k, v in json.load(open(storage.path("response_tokens")))["tokens"][T].items()}; pair = [ids["Yes"], ids["No"]]; assert first_token(tok, "Yes") == (ids["Yes"], []), "token ids"
        for task in TASKS:
            R = rows(T, task); P = [chat(tok, CRIT.format(q=r["question"], o=opts(r), a=r["answer"])) for r in R]
            crit, nnew, trunc = gen(model, tok, P, f"{T} {task}")
            Ga, _, _, sa = read(model, tok, Wu, nrm, P, pair, norm, f"{task} a")
            q = np.array([r["qid"] for r in R]); np.savez(os.path.join(d, f"oe_{task}.npz"), qid=q, G_a=Ga)
            pd.DataFrame(dict(qid=q, critique=crit, n_new=nnew, truncated=trunc)).to_parquet(os.path.join(d, f"oe_{task}_gen.parquet"))
            info[task] = dict(n=len(R), n_truncated=int(np.sum(trunc)), mean_new=float(np.mean(nnew)), sec_a=sa, prompt_example=P[0], critique_example=crit[0])
            print(f"[{T} {task}] n={len(R)} truncated={int(np.sum(trunc))}", flush=True)
        info["norm"] = str(norm[0]); json.dump(info, open(os.path.join(d, "oe_gen.json"), "w"), indent=1)
    else:
        J = a.judge.split("/")[-1]; model, tok, Wu, nrm = load(a.judge); norm = [None]
        three = {w: first_token(tok, w)[0] for w in ("Yes", "No", "Unclear")}; assert len(set(three.values())) == 3, three; cols = list(three.values())
        for M in [a.judge]:
            T = M.split("/")[-1]; d = os.path.join(OUT, T)
            for task in TASKS:
                R = rows(T, task); gdf = pd.read_parquet(os.path.join(d, f"oe_{task}_gen.parquet")); assert list(gdf.qid) == [r["qid"] for r in R], "row order"
                P = [chat(tok, JUDGE.format(q=r["question"], o=opts(r), a=r["answer"], c=c)) for r, c in zip(R, gdf.critique)]
                _, Z, top, _ = read(model, tok, Wu, nrm, P, cols, norm, f"judge {J} on {T} {task}")
                lab = np.array(["Yes", "No", "Unclear"])[Z.argmax(1)]
                pd.DataFrame(dict(qid=gdf.qid, judge=lab, z_yes=Z[:, 0], z_no=Z[:, 1], z_unclear=Z[:, 2], top_in_three=np.isin(top, cols))).to_parquet(os.path.join(d, f"oe_judge_{J}_on_{T}_{task}.parquet"))
                info[f"{T}_{task}"] = dict(unclear=float((lab == "Unclear").mean()), yes=float((lab == "Yes").mean()), top_in_three=float(np.isin(top, cols).mean()), prompt_example=P[0])
                print(f"[judge {J} on {T} {task}] yes={info[f'{T}_{task}']['yes']:.3f} unclear={info[f'{T}_{task}']['unclear']:.3f}", flush=True)
        json.dump(info, open(os.path.join(OUT, f"oe_judge_{J}.json"), "w"), indent=1)

main()
