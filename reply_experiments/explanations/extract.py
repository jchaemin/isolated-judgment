"""GPU: an explanation, then the state and the Yes/No margin at the verdict. Usage: python reply_experiments/explanations/extract.py --model M --task T"""
import argparse, json, os, re, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from readout import config, data, models, prompts, storage
from preprocessing.extract_variants import load, cell, text, BATCH_SIZE, ROWS
from transformers import StoppingCriteria, StoppingCriteriaList

SUFFIX = ' First explain your reasoning in a few sentences. Then end with a final line of the form "Verdict: Yes" or "Verdict: No".'
MAXNEW = 320; TAIL = 6
HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = {"cot_negated": "negated"}

class VerdictStop(StoppingCriteria):
    def __init__(self, tok, L, n): self.tok, self.L, self.seen = tok, L, [None] * n
    def __call__(self, input_ids, scores, **kw):
        step = input_ids.shape[1] - self.L; done = []
        for i in range(input_ids.shape[0]):
            if self.seen[i] is None and step >= 3 and "Verdict:" in self.tok.decode(input_ids[i, max(self.L, input_ids.shape[1] - 24):], skip_special_tokens=True):
                self.seen[i] = step
            done.append(self.seen[i] is not None and step - self.seen[i] >= TAIL)
        return torch.tensor(done, device=input_ids.device)

def gen_word(full):
    k = full.find("Verdict:")
    if k < 0: return "none"
    m = re.match(r"\s*[\*\"'\[\(]*\s*([A-Za-z]+)", full[k + len("Verdict:"):])
    return m.group(1) if m and m.group(1) in ("Yes", "No") else ("other" if m or full[k + 8:].strip() else "none")

def generate(model, tok, tx):
    ids_, am_ = models.encode_full(tok, tx, None, model.device); L = ids_.shape[1]
    with torch.no_grad():
        out = model.generate(input_ids=ids_, attention_mask=am_, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=MAXNEW,
                             stopping_criteria=StoppingCriteriaList([VerdictStop(tok, L, len(tx))]), pad_token_id=tok.pad_token_id)
    conts = [tok.decode(out[i, L:], skip_special_tokens=True) for i in range(len(tx))]
    P = [ids_[i][am_[i].bool()].tolist() for i in range(len(tx))]
    return P, conts

def cut(c):
    k = c.find("Verdict:")
    return (c[:k + len("Verdict:")], True) if k >= 0 else (c.rstrip() + "\nVerdict:", False)

def read_last(model, tok, Wu, nrm, seqs, norm, yes, no):
    Lm = max(map(len, seqs)); pad = tok.pad_token_id
    ids = torch.tensor([[pad] * (Lm - len(s)) + s for s in seqs], device=model.device); am = torch.tensor([[0] * (Lm - len(s)) + [1] * len(s) for s in seqs], device=model.device)
    pos = (am.cumsum(1) - 1).clamp(min=0)
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=am, position_ids=pos, output_hidden_states=True, logits_to_keep=1)
        lg = o.logits[:, -1, :].float(); h = o.hidden_states[-1][:, -1, :]
        if norm[0] is None: norm[0] = models.decide_norm(h, lg, Wu, nrm)
        G = models.apply_state(h, norm[0], Wu, nrm).cpu().numpy().astype(np.float16)
        M = (torch.logsumexp(lg[:, yes], -1) - torch.logsumexp(lg[:, no], -1)).cpu().numpy(); top = lg.argmax(-1).cpu().numpy()
    del o; return G, M, top

def read_plain(model, tok, Wu, nrm, tx, norm):
    n = len(tx); G = np.zeros((n, model.config.hidden_size), np.float16)
    for b0 in range(0, n, BATCH_SIZE):
        idx = list(range(b0, min(b0 + BATCH_SIZE, n))); ids_, am_ = models.encode_full(tok, [tx[i] for i in idx], None, model.device)
        with torch.no_grad():
            o = model(input_ids=ids_[:, :-1], attention_mask=am_[:, :-1], use_cache=True, logits_to_keep=1); cache = o.past_key_values; del o
            o = model(input_ids=ids_[:, -1:], attention_mask=am_, past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=1)
            lg = o.logits[:, -1, :].float(); h = o.hidden_states[-1][:, -1, :]
            if norm[0] is None: norm[0] = models.decide_norm(h, lg, Wu, nrm)
            G[idx] = models.apply_state(h, norm[0], Wu, nrm).cpu().numpy().astype(np.float16)
        del o, cache
    return G

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--wordings", default=",".join(config.FIVE)); a = ap.parse_args()
    T = a.model.split("/")[-1]; d = os.path.join(HERE, "out", T, a.task); os.makedirs(d, exist_ok=True); path = os.path.join(d, "cot.npz")
    if os.path.exists(path): print("exists", path); return
    D1 = data.states(T, a.task, "yes_no"); n = ROWS; q = D1["qid"].astype(int)[:n]
    rows, _ = prompts.rows_of(T, a.task, n); assert (np.array([r["qid"] for r in rows]) == q).all()
    I = prompts.instructions(pd.read_csv(storage.path("wordings")).set_index("wording")); P01 = I[("yes_no", "P01")]
    model, tok, Wu, nrm = load(a.model); norm = [None]; _, ids = config.response_tokens(cell(T, a.task))
    pre = tok("Verdict:", add_special_tokens=False).input_ids; sp = lambda w: tok("Verdict: " + w, add_special_tokens=False).input_ids[len(pre):]
    assert tok("Verdict: Yes", add_special_tokens=False).input_ids[:len(pre)] == pre and len(sp("Yes")) == 1 and len(sp("No")) == 1, "space-prefixed Yes/No not a single token"
    YES = list(dict.fromkeys([ids["Yes"], sp("Yes")[0]])); NO = list(dict.fromkeys([ids["No"], sp("No")[0]])); assert not set(YES) & set(NO), (YES, NO)
    read_plain(model, tok, Wu, nrm, [text(tok, r, P01) for r in rows[:16]], norm)
    out = dict(qid=q); ex = {}; stats = {}; t0 = time.time()
    for form, fm in FORMS.items():
        ex[form] = []
        for p in a.wordings.split(","):
            instr = I[(fm, p)] + SUFFIX; tx = [text(tok, r, instr) for r in rows]; k = f"{form}_{p}"
            G = np.zeros((n, model.config.hidden_size), np.float16); M = np.zeros(n, np.float32); top = np.zeros(n, np.int64); found = np.zeros(n, bool); gw = np.empty(n, "U8"); ln = np.zeros(n, np.int64)
            for b0 in range(0, n, BATCH_SIZE):
                idx = list(range(b0, min(b0 + BATCH_SIZE, n))); P, conts = generate(model, tok, [tx[i] for i in idx]); seqs = []
                for j, i in enumerate(idx):
                    kept, f = cut(conts[j]); found[i] = f; gw[i] = gen_word(conts[j]); c = tok(kept, add_special_tokens=False).input_ids; ln[i] = len(c); seqs.append(P[j] + c)
                    if p == "P01" and i < 3: ex[form].append(dict(row=i, qid=int(q[i]), found=f, gen_word=gw[i], kept=kept, full=conts[j]))
                g, m, t = read_last(model, tok, Wu, nrm, seqs, norm, YES, NO); G[idx], M[idx], top[idx] = g, m, t
                if (b0 // BATCH_SIZE) % 5 == 0: print(f"[{k}] rows {idx[-1] + 1}/{n} {time.time()-t0:.0f}s", flush=True)
            C = np.isin(top, YES + NO); out.update({f"G_{k}": G, f"M_{k}": M, f"C_{k}": C, f"top_{k}": top, f"found_{k}": found, f"gen_word_{k}": gw, f"len_{k}": ln})
            stats[k] = dict(compliance=float(C.mean()), found=float(found.mean()), gen_Yes=float((gw == "Yes").mean()), gen_No=float((gw == "No").mean()), gen_other=float((gw == "other").mean()), gen_none=float((gw == "none").mean()), len_median=float(np.median(ln)), sec=time.time() - t0)
            print(k, stats[k], flush=True)
    tmp = path[:-4] + ".tmp.npz"; np.savez(tmp, **out); os.replace(tmp, path)
    json.dump(dict(model=a.model, task=a.task, n=n, suffix=SUFFIX, max_new_tokens=MAXNEW, stop_tail_tokens=TAIL, yes_ids=YES, no_ids=NO, norm=str(norm[0]), stats=stats, examples=ex), open(os.path.join(d, "cot.json"), "w"), indent=1)
    print("wrote", d, "elapsed", time.time() - t0, flush=True)

main()
