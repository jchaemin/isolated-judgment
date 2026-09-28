"""GPU: hidden states for the correctness conditions, the entailment and safety tasks, and the additional models."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import config, data, models, prompts, storage
from readout.storage import key

BATCH_SIZE = 16
ALL_WORDINGS = list(config.WORDINGS)
DEPTHS = [0.25, 0.5, 0.75, 0.9]
FORMATS = ["yes_no", "true_false", "a_b", "b_a"]
TASK_FORMATS = ["yes_no", "a_b", "b_a"]
FORMAT_FIRST = ["yes_no", "a_b", "b_a"]
CLAUSES = {"one_zero": "Respond 1 if correct, 0 if incorrect.", "correct_incorrect": "Respond with the word correct or incorrect.",
           "zero_means_correct": "Respond 0 if correct, 1 if incorrect.", "false_means_correct": "Respond False if correct, True if incorrect.",
           "no_means_correct": "Answer No if the candidate is correct and Yes if it is incorrect."}
STATEMENT_CORRECT, STATEMENT_INCORRECT, STATEMENT_STEM = "The proposed answer is correct.", "The proposed answer is incorrect.", "The proposed answer is"
ONE_ZERO_CORRECT_INCORRECT = {"one_zero": ("1", "0"), "correct_incorrect": ("Correct", "Incorrect")}
ZERO_FALSE_MEANS_CORRECT = {"zero_means_correct": ("0", "1"), "false_means_correct": ("False", "True")}
MORE_CONDITIONS = ["one_zero", "correct_incorrect", "zero_means_correct", "false_means_correct", "no_means_correct"]
MORE_WORDINGS, CHECK = ["P03", "P07"], "P01"
DEPTH_CONDITIONS = ["yes_no", "true_false", "a_b", "b_a", "sentence", "one_zero", "correct_incorrect", "negated", "no_means_correct"]
LAYER_CHECK = ["yes_no", "a_b", "b_a", "negated", "no_means_correct"]
ADDITIONAL_BATCH = 8
FIVE_WORDING_MODELS = ("gemma-2-9b-it", "Mistral-Nemo-Instruct-2407", "Ministral-8B-Instruct-2410", "OLMo-2-1124-7B-Instruct")
ALL_CONDITIONS = ["sentence", "one_zero", "correct_incorrect", "zero_means_correct", "false_means_correct", "no_means_correct"]
LEAN_CONDITIONS = ["sentence", "no_means_correct"]
WORDS = {"one_zero": ("1", "0"), "correct_incorrect": ("correct", "incorrect"), "zero_means_correct": ("0", "1"), "false_means_correct": ("False", "True"),
         "no_means_correct": ("No", "Yes")}


def out_path(kind, out, **fields):
    return storage.path(kind, **fields) if out is None else os.path.join(out, storage.LAYOUT[kind].format(**fields))


def cell(model, task):
    return dict(model=model, task=task, family="correctness" if task in config.TASKS else task)


def load(model_name):
    model, tok = models.chat_model(model_name)
    assert models.uses_chat_template(tok)
    return model, tok, model.get_output_embeddings().weight, models.final_norm(model)


def batch_for(task):
    return 8 if task in ("gsm8k", "mmlu") else 16


def wordings():
    return pd.read_csv(storage.path("wordings")).set_index("wording")


def task_wordings(task):
    PT = pd.read_csv(storage.path("task_wordings"))
    return PT[PT.task == task].set_index("wording")


def text(tok, r, instr, chat=True):
    return models.wrap(tok, prompts.TEMPLATE.format(q=r["question"], o=("\n\n" + r["options"]) if r["options"] else "", a=r["answer"], instr=instr), chat=chat)


def task_text(tok, stem, task, r, instr):
    return models.wrap(tok, stem.format(p=r.get("p"), h=r.get("h"), instr=instr) if task == "entailment" else stem.format(q=r.get("q"), r=r.get("r"), instr=instr))


def stated(PT, p, clause):
    return f"{PT.loc[p, 'stem']} {clause}"


def word_tokens(tok, w, chat=True):
    e = tok(w if chat else " " + w, add_special_tokens=False).input_ids; pre = None
    while len(e) > 1 and tok.decode([e[0]]).strip() == "":
        pre = int(e[0]); e = e[1:]
    return e, pre


def first_token(tok, w, chat=True):
    return int(word_tokens(tok, w, chat)[0][0])


def one_zero_correct_incorrect_tokens(tok):
    out = {}
    for c, (wpos, wneg) in ONE_ZERO_CORRECT_INCORRECT.items():
        (ip, pp), (ineg, pn) = word_tokens(tok, wpos), word_tokens(tok, wneg)
        out[c] = dict(pair=(int(ip[0]), int(ineg[0])), prefix=pp if (pp is not None and pp == pn) else None)
        assert out[c]["pair"][0] != out[c]["pair"][1], ("the two response words share their first token", c, out[c])
        if c == "correct_incorrect":
            out[c]["lower_pair"] = (first_token(tok, "correct"), first_token(tok, "incorrect"))
    return out


def zero_false_means_correct_tokens(tok):
    out = {}
    for c, (wp, wn) in ZERO_FALSE_MEANS_CORRECT.items():
        cand = [(wp, wn)] + ([(wp.capitalize(), wn.capitalize())] if wp[0].islower() else [(wp.lower(), wn.lower())])
        ids_ = [(first_token(tok, u), first_token(tok, v)) for u, v in cand]
        assert ids_[0][0] != ids_[0][1], ("the two response words share their first token", c, ids_)
        out[c] = dict(pair=ids_[0], alt_pair=ids_[1])
    return out


def no_means_correct_tokens(tok, c):
    des, ids = config.response_tokens(c); names = ["Yes", "No", des["true_false"][0], des["true_false"][1], "A", "B"]
    right, wrong = first_token(tok, "Correct"), first_token(tok, "Incorrect")
    if right == wrong:
        right, wrong = first_token(tok, "correct"), first_token(tok, "incorrect")
    return names + ["1", "0", "Correct", "Incorrect"], [int(ids[nm]) for nm in names] + [first_token(tok, "1"), first_token(tok, "0"), right, wrong]


def blocks(model):
    for m_ in (model, getattr(model, "model", None)):
        if m_ is not None and hasattr(m_, "layers"):
            return m_.layers
        if m_ is not None and hasattr(m_, "model") and hasattr(m_.model, "layers"):
            return m_.model.layers


def identity(A, B):
    cs = models.cosine(A.astype(np.float64), B.astype(np.float64))
    return [float(np.median(cs)), float(cs.min()), bool(models.identity_bad(cs))]


def shared_states(model, tok, Wu, nrm, texts, extra, keys, norm):
    pl = models.plan(tok, texts, extra=extra); sh_ = models.Shared(model, tok, pl) if pl is not None else None
    for k in keys:
        if sh_ is not None:
            last_ids, mask, npre = sh_.extend(k); cache = sh_.cache; sh_.cache.crop(npre)
        else:
            ids_, am_ = models.encode_full(tok, texts[k], extra.get(k), model.device)
            with torch.no_grad():
                o = model(input_ids=ids_[:, :-1], attention_mask=am_[:, :-1], use_cache=True, logits_to_keep=1); cache = o.past_key_values; del o
            last_ids, mask = ids_[:, -1:], am_
        with torch.no_grad():
            o = model(input_ids=last_ids, attention_mask=mask, past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=1)
            lg = o.logits[:, -1, :].float(); h = o.hidden_states[-1][:, -1, :]
            if norm[0] is None:
                norm[0] = models.decide_norm(h, lg, Wu, nrm)
            g = models.apply_state(h, norm[0], Wu, nrm).cpu().numpy()
        del o
        yield k, g, lg
    del sh_


def run_formats(model, task, max_rows=0, out=None):
    T = model.split("/")[-1]; paths = {f: out_path("states", out, model=T, task=task, condition=f) for f in FORMATS}
    todo = [f for f in FORMATS if max_rows or not os.path.exists(paths[f])]
    if not todo:
        print(f"[formats] {T} {task} SKIP", flush=True); return
    _, ids = config.response_tokens(cell(T, task)); tid = [ids[nm] for nm in config.SIX]; PT = wordings()
    rows, _ = prompts.answer_rows(T, task, n=max_rows)
    net, tok, W, nrm = load(model)
    for f in todo:
        store = {"y": np.array([r["y"] for r in rows], int), "qid": np.array([r["qid"] for r in rows], int)} if f == "yes_no" else {}
        if f == "true_false":
            store["token_names"] = np.array(config.SIX)
        for p in ALL_WORDINGS:
            G, Z, t20i = models.forward(net, W, nrm, tok, [text(tok, r, str(PT.loc[p, f])) for r in rows], BATCH_SIZE, tid)
            store.update({f"G_{p}": G, f"Z_{p}": Z, f"t20i_{p}": t20i})
        models.save(paths[f], compressed=True, **store)
        print(f"[formats] {T} {task} {f}", flush=True)


def run_task_formats(model, task, max_rows=0, out=None):
    T = model.split("/")[-1]; paths = {f: out_path("states", out, model=T, task=task, condition=f) for f in TASK_FORMATS}
    todo = [f for f in TASK_FORMATS if max_rows or not os.path.exists(paths[f])]
    if not todo:
        print(f"[cross-task] {T} {task} SKIP", flush=True); return
    _, ids = config.response_tokens(cell(T, task)); names = ["Yes", "No", "A", "B"]; tid = [ids[nm] for nm in names]
    PT = task_wordings(task); stem = json.load(open(storage.path("task_tokens")))["templates"][task]; rows = prompts.cross_rows(task, n=max_rows)
    net, tok, W, nrm = load(model)
    for f in todo:
        store = {}
        if f == "yes_no":
            store = {"y": np.array([r["y"] for r in rows], int), "qid": np.array([r["example_id"] for r in rows], int), "token_names": np.array(names), "token_ids": np.array(tid)}
        for p in config.FIVE:
            G, Z, t20i = models.forward(net, W, nrm, tok, [task_text(tok, stem, task, r, str(PT.loc[p, f])) for r in rows], BATCH_SIZE, tid)
            store.update({f"G_{p}": G, f"Z_{p}": Z, **({f"t20i_{p}": t20i} if f == "yes_no" else {})})
        models.save(paths[f], compressed=True, **store)
        print(f"[cross-task] {T} {task} {f}", flush=True)


def run_negated(model, task, max_rows=0, out=None):
    T = model.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="negated")
    if os.path.exists(path) and not max_rows:
        print(f"[negated] {T} {task} SKIP", flush=True); return
    _, ids = config.response_tokens(cell(T, task)); tid = [ids[nm] for nm in config.SIX]; PT = wordings()
    rows, _ = prompts.answer_rows(T, task, n=max_rows)
    net, tok, W, nrm = load(model); store = {}
    for p in ALL_WORDINGS:
        G, Z, t20i = models.forward(net, W, nrm, tok, [text(tok, r, prompts.negate(str(PT.loc[p, "yes_no"]))) for r in rows], BATCH_SIZE, tid)
        store.update({f"G_{p}": G, f"Z_{p}": Z, f"t20i_{p}": t20i})
    models.save(path, compressed=True, **store)
    print(f"[negated] {T} {task}", flush=True)


def run_sentence(model, task, max_rows=0, out=None):
    T = model.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="sentence")
    if os.path.exists(path) and not max_rows:
        print(f"[sentence] {T} {task} SKIP", flush=True); return
    _, ids = config.response_tokens(cell(T, task)); tid = [ids[nm] for nm in config.SIX]; PT = wordings()
    rows, _ = prompts.answer_rows(T, task, n=max_rows)
    net, tok, W, nrm = load(model)
    ids_pos, ids_neg, ids_pre = (tok(x, add_special_tokens=False).input_ids for x in (STATEMENT_CORRECT, STATEMENT_INCORRECT, STATEMENT_STEM)); k = len(ids_pre)
    assert ids_pos[:k] == ids_pre and ids_neg[:k] == ids_pre and ids_pos[k] != ids_neg[k], (ids_pos, ids_neg, ids_pre)
    div = torch.tensor([ids_pos[k], ids_neg[k]]); store = {}
    for p in ALL_WORDINGS:
        instr = f"{PT.loc[p, 'stem']} {PT.loc[p, 'verb']} with exactly one of the following sentences:\n{STATEMENT_CORRECT}\n{STATEMENT_INCORRECT}"
        texts = [text(tok, r, instr) for r in rows]
        store[f"G_{p}"] = models.forward(net, W, nrm, tok, texts, BATCH_SIZE, tid)[0]
        if p not in config.FIVE:
            continue
        Zdiv = np.zeros((len(texts), 2), np.float32)
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tok(texts[i:i + BATCH_SIZE], return_tensors="pt", padding=True).to(net.device); pre = torch.tensor([ids_pre] * enc.input_ids.shape[0], device=net.device)
            with torch.no_grad():
                o = net(input_ids=torch.cat([enc.input_ids, pre], 1), attention_mask=torch.cat([enc.attention_mask, torch.ones_like(pre)], 1), output_hidden_states=True, logits_to_keep=1)
                _, lg = models.state(o, o.hidden_states[-1][:, -1, :], W, nrm)
            Zdiv[i:i + enc.input_ids.shape[0]] = lg[:, div.to(lg.device)].cpu().numpy()
            del enc, o, lg
        store[f"Zdiv_{p}"] = Zdiv
    models.save(path, compressed=True, **store)
    print(f"[sentence] {T} {task}", flush=True)


def run_format_first(model, task, max_rows=0, out=None):
    T = model.split("/")[-1]; paths = {(f, p): out_path("format_first", out, model=T, task=task, condition=f, wording=p) for f in FORMAT_FIRST for p in config.FIVE}
    todo = [k for k in paths if max_rows or not os.path.exists(paths[k])]
    if not todo:
        print(f"[format-first] {T} {task} SKIP", flush=True); return
    PT = wordings(); rows, _ = prompts.answer_rows(T, task, n=max_rows); n = len(rows)
    part1 = [r["question"] + (("\n\n" + r["options"]) if r["options"] else "") for r in rows]; part2 = [r["answer"] for r in rows]
    net, tok, _, nrm = load(model); assert tok.is_fast
    dmod = net.config.hidden_size; last_block = net.model.layers[-1]

    def content(i, clause, stem):
        return clause + "\n\n" + prompts.TEMPLATE.format(q=rows[i]["question"], o=("\n\n" + rows[i]["options"]) if rows[i]["options"] else "", a=rows[i]["answer"], instr=stem)

    def ends(i, clause, stem):
        c = content(i, clause, stem); w = models.wrap(tok, c); k = w.find(c)
        kq = c.find(part1[i], len(clause)); e1 = k + kq + len(part1[i]); ka = c.find(part2[i], kq + len(part1[i])); e2 = k + ka + len(part2[i]); e3 = k + c.rfind(stem) + len(stem)
        return w, [e1, e2, e3]

    def tok_positions(txt, es):
        off = tok(txt, return_offsets_mapping=True)["offset_mapping"]; pos = []
        for e in es:
            c = e - 1; hit = [k for k, (s_, t_) in enumerate(off) if s_ <= c < t_ and t_ > s_]; pos.append(hit[-1])
        return pos + [len(off) - 1]

    cap = {}; hook = last_block.register_forward_hook(lambda _mod, _inp, o_: cap.__setitem__("h", (o_[0] if isinstance(o_, tuple) else o_).detach()))
    for f, p in todo:
        stem = str(PT.loc[p, "stem"]); instr = str(PT.loc[p, f]); clause = instr[len(stem):].strip()
        H = np.zeros((n, 4, dmod), np.float16); texts, ptoks = [], []
        for i in range(n):
            w, es = ends(i, clause, stem); texts.append(w); ptoks.append(tok_positions(w, es))
        for i in range(0, n, BATCH_SIZE):
            bt = texts[i:i + BATCH_SIZE]; enc = tok(bt, return_tensors="pt", padding=True).to(net.device); nb = len(bt); Tn = enc["input_ids"].shape[1]; pad = Tn - enc["attention_mask"].sum(1)
            with torch.no_grad():
                net(**enc, logits_to_keep=1); h = cap["h"]
                idx = torch.tensor([[int(pad[b]) + pp for pp in ptoks[i + b]] for b in range(nb)], device=h.device)
                g = torch.gather(h, 1, idx[:, :, None].expand(-1, -1, h.shape[2])); gp = nrm(g.to(next(nrm.parameters()).device)).float()
                H[i:i + nb] = gp.cpu().numpy().astype(np.float16)
            del enc
        models.save(paths[(f, p)], compressed=True, H=H)
        print(f"[format-first] {T} {task} {f} {p}", flush=True)
    hook.remove()


def run_one_zero_correct_incorrect(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="one_zero_correct_incorrect")
    if os.path.exists(path) and not max_rows:
        print("[one-zero-correct-incorrect] SKIP", path, flush=True); return
    t0 = time.time(); qid = data.states(T, task, "yes_no")["qid"].astype(int); PT = wordings()
    d, _ = prompts.answers(T, task); d = d.set_index("qid")
    assert set(qid.tolist()) <= set(d.index.astype(int).tolist()), "the candidate answers do not cover the questions of the stored states"
    d = d.loc[qid]
    rows = [{"question": r.question, "options": (r.options if "options" in d.columns and isinstance(r.options, str) and r.options else ""), "answer": r.generated_text} for _, r in d.iterrows()]
    rows = rows[:max_rows] if max_rows else rows; n = len(rows)
    model, tok, Wu, nrm = load(model_name); TOK = one_zero_correct_incorrect_tokens(tok); store = {}; bs = batch_for(task)
    for c in ONE_ZERO_CORRECT_INCORRECT:
        pair = torch.tensor(TOK[c]["pair"]); lpair = torch.tensor(TOK[c]["lower_pair"]) if "lower_pair" in TOK[c] else None
        for p in config.THREE:
            texts = [text(tok, r, stated(PT, p, CLAUSES[c])) for r in rows]
            G = np.zeros((n, Wu.shape[1]), np.float16); Z = np.zeros((n, 2), np.float32); Zl = np.zeros((n, 2), np.float32); t20i = np.zeros((n, 20), np.int32)
            for i in range(0, n, bs):
                bt = texts[i:i + bs]; s = slice(i, i + len(bt)); enc = tok(bt, return_tensors="pt", padding=True).to(model.device)
                if TOK[c]["prefix"] is not None:
                    pre = torch.full((enc.input_ids.shape[0], 1), TOK[c]["prefix"], dtype=enc.input_ids.dtype, device=enc.input_ids.device)
                    enc = {"input_ids": torch.cat([enc.input_ids, pre], 1), "attention_mask": torch.cat([enc.attention_mask, torch.ones_like(pre)], 1)}
                with torch.no_grad():
                    o = model(**enc, output_hidden_states=True, logits_to_keep=1); g, lg = models.state(o, o.hidden_states[-1][:, -1, :], Wu, nrm)
                    G[s] = g.cpu().numpy().astype(np.float16); Z[s] = lg[:, pair.to(lg.device)].cpu().numpy()
                    Zl[s] = lg[:, lpair.to(lg.device)].cpu().numpy() if lpair is not None else 0; t20i[s] = torch.topk(lg, 20, dim=-1).indices.cpu().numpy()
                    del o
            store.update({key("G", c, p): G, key("Z", c, p): Z, key("t20i", c, p): t20i, **({key("Zlower", c, p): Zl} if lpair is not None else {})})
            print(f"[one-zero-correct-incorrect] {T} {task} {c} {p}  {time.time()-t0:.0f}s", flush=True)
    models.save(path, compressed=True, **store)
    tokens = {c: {k: v for k, v in TOK[c].items() if k in ("pair", "lower_pair")} for c in TOK}
    json.dump(dict(tokens=tokens), open(out_path("states_meta", out, model=T, task=task, condition="one_zero_correct_incorrect"), "w"), indent=1)
    print(f"[one-zero-correct-incorrect] wrote {path}  {time.time()-t0:.0f}s", flush=True)


def run_zero_false_means_correct(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="zero_false_means_correct")
    if os.path.exists(path) and not max_rows:
        print("[zero-false-means-correct] SKIP", path, flush=True); return
    t0 = time.time(); q = data.states(T, task, "yes_no")["qid"].astype(int); PT = wordings()
    rows, _ = prompts.rows_of(T, task, 0); assert (np.array([r["qid"] for r in rows]) == q).all()
    rows = rows[:max_rows] if max_rows else rows; n = len(rows)
    model, tok, Wu, nrm = load(model_name); TOK = zero_false_means_correct_tokens(tok); store = {}; bs = batch_for(task)
    for c in ZERO_FALSE_MEANS_CORRECT:
        ids4 = torch.tensor([TOK[c]["pair"][0], TOK[c]["pair"][1], TOK[c]["alt_pair"][0], TOK[c]["alt_pair"][1]])
        for p in config.THREE:
            texts = [text(tok, r, stated(PT, p, CLAUSES[c])) for r in rows]
            G = np.zeros((n, Wu.shape[1]), np.float16); Z = np.zeros((n, 4), np.float32); t20i = np.zeros((n, 20), np.int32)
            for i in range(0, n, bs):
                bt = texts[i:i + bs]; s = slice(i, i + len(bt)); enc = tok(bt, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    o = model(**enc, output_hidden_states=True, logits_to_keep=1); g, lg = models.state(o, o.hidden_states[-1][:, -1, :], Wu, nrm)
                    t20i[s] = torch.topk(lg, 20, dim=-1).indices.cpu().numpy(); G[s] = g.cpu().numpy().astype(np.float16); Z[s] = lg[:, ids4.to(lg.device)].cpu().numpy()
                    del o
            store.update({key("G", c, p): G, key("Z", c, p): Z, key("t20i", c, p): t20i})
            print(f"[zero-false-means-correct] {T} {task} {c} {p}  {time.time()-t0:.0f}s", flush=True)
    models.save(path, compressed=True, **store)
    json.dump(dict(tokens=TOK), open(out_path("states_meta", out, model=T, task=task, condition="zero_false_means_correct"), "w"), indent=1)
    print(f"[zero-false-means-correct] wrote {path}  {time.time()-t0:.0f}s", flush=True)


def run_no_means_correct(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; paths = {p: out_path("layers", out, model=T, task=task, condition="no_means_correct", wording=p) for p in config.THREE}
    need = [p for p in config.THREE if max_rows or not os.path.exists(paths[p])]
    if not need:
        print("[no-means-correct] SKIP", T, task, flush=True); return
    q = data.states(T, task, "yes_no")["qid"].astype(int); PT = wordings()
    rows, _ = prompts.rows_of(T, task, 0); assert (np.array([r["qid"] for r in rows]) == q).all()
    rows = rows[:max_rows] if max_rows else rows; n = len(rows)
    model, tok, Wu, nrm = load(model_name); names, tid = no_means_correct_tokens(tok, cell(T, task))
    tid_t = torch.tensor(tid); bs = batch_for(task); t0 = time.time()
    for p in need:
        texts = [text(tok, r, stated(PT, p, CLAUSES["no_means_correct"])) for r in rows]
        H = None; Z = np.zeros((n, len(names)), np.float32); t20i = np.zeros((n, 20), np.int32)
        for i in range(0, n, bs):
            bt = texts[i:i + bs]; s = slice(i, i + len(bt)); enc = tok(bt, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                o = model(**enc, output_hidden_states=True, logits_to_keep=1)
                hs = torch.stack([h[:, -1, :] for h in o.hidden_states], 1).float()
                if H is None:
                    H = np.zeros((n, hs.shape[1], hs.shape[2]), np.float16)
                H[s] = hs.cpu().numpy().astype(np.float16)
                g, lg = models.state(o, o.hidden_states[-1][:, -1, :], Wu, nrm)
                t20i[s] = torch.topk(lg, 20, dim=-1).indices.cpu().numpy(); Z[s] = lg[:, tid_t.to(lg.device)].cpu().numpy()
                del o, hs
        models.save(paths[p], compressed=True, H=H, Z=Z, t20i=t20i, token_names=np.array(names), token_ids=np.array(tid), semantic=np.array(("No", "Yes")))
        print(f"[no-means-correct] {T} {task} {p}: layers {H.shape[1]}  {time.time()-t0:.0f}s", flush=True)


def run_more_wordings(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="more_wordings")
    if os.path.exists(path) and not max_rows:
        print("[more-wordings] SKIP", path, flush=True); return
    t0 = time.time(); q = data.states(T, task, "yes_no")["qid"].astype(int)[:(max_rows or 400)]; n = len(q); PT = wordings()
    rows, _ = prompts.rows_of(T, task, n); assert (np.array([r["qid"] for r in rows]) == q).all()
    model, tok, Wu, nrm = load(model_name); TK2 = one_zero_correct_incorrect_tokens(tok); TK3 = zero_false_means_correct_tokens(tok); names, tid = no_means_correct_tokens(tok, cell(T, task))
    COLS = {"one_zero": list(TK2["one_zero"]["pair"]), "correct_incorrect": list(TK2["correct_incorrect"]["pair"]), "no_means_correct": tid,
            **{c: list(TK3[c]["pair"]) + list(TK3[c]["alt_pair"]) for c in ZERO_FALSE_MEANS_CORRECT}}
    LOWER = list(TK2["correct_incorrect"]["lower_pair"]); ws = [CHECK] + MORE_WORDINGS
    KEYS = [(c, p) for p in ws for c in MORE_CONDITIONS]
    EXTRA = {(c, p): [TK2[c]["prefix"]] for c in ONE_ZERO_CORRECT_INCORRECT if TK2[c]["prefix"] is not None for p in ws}
    R = {k: dict(G=np.zeros((n, model.config.hidden_size), np.float16), Z=np.zeros((n, len(COLS[k[0]])), np.float32), t20i=np.zeros((n, 20), np.int32), Zl=np.zeros((n, 2), np.float32)) for k in KEYS}
    norm = [None]
    for b0 in range(0, n, BATCH_SIZE):
        idx = list(range(b0, min(b0 + BATCH_SIZE, n))); texts = {(c, p): [text(tok, rows[i], stated(PT, p, CLAUSES[c])) for i in idx] for c, p in KEYS}
        for k, g, lg in shared_states(model, tok, Wu, nrm, texts, EXTRA, KEYS, norm):
            r = R[k]; r["G"][idx] = g.astype(np.float16); r["Z"][idx] = lg[:, COLS[k[0]]].cpu().numpy(); r["t20i"][idx] = torch.topk(lg, 20, dim=-1).indices.cpu().numpy()
            if k[0] == "correct_incorrect":
                r["Zl"][idx] = lg[:, LOWER].cpu().numpy()
        if b0 == 0 or (b0 // BATCH_SIZE) % 10 == 0:
            print(f"[more-wordings] {T} {task} rows {idx[-1] + 1}/{n}  {time.time()-t0:.0f}s", flush=True)
    Z2 = data.states(T, task, "one_zero_correct_incorrect"); Z3 = data.states(T, task, "zero_false_means_correct")
    Zr = np.load(storage.path("layers", model=T, task=task, condition="no_means_correct", wording=CHECK), allow_pickle=True)
    chk = {c: identity(R[(c, CHECK)]["G"], ((Z2 if c in ONE_ZERO_CORRECT_INCORRECT else Z3)[key("G", c, CHECK)] if c != "no_means_correct" else Zr["H"][:, -1, :])[:n]) for c in MORE_CONDITIONS}
    bad = {c: v for c, v in chk.items() if v[2]}
    print(f"[more-wordings] {T} {task} identity at {CHECK}: " + (f"MISMATCH {bad}" if bad else "OK") + f" {chk}", flush=True)
    if bad:
        raise SystemExit(3)
    store = {key("token_ids", "no_means_correct"): np.array(tid), key("token_names", "no_means_correct"): np.array(names), key("semantic", "no_means_correct"): np.array(("No", "Yes"))}
    for (c, p), r in R.items():
        if p != CHECK:
            store.update({key("G", c, p): r["G"], key("Z", c, p): r["Z"], key("t20i", c, p): r["t20i"], **({key("Zlower", c, p): r["Zl"]} if c == "correct_incorrect" else {})})
    models.save(path, compressed=True, **store)
    print(f"[more-wordings] wrote {path}  {time.time()-t0:.0f}s", flush=True)


def run_depth(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; path = out_path("depth", out, model=T, task=task)
    if os.path.exists(path) and not max_rows:
        print("[depth] SKIP", path, flush=True); return
    t0 = time.time(); q = data.states(T, task, "yes_no")["qid"].astype(int); PT = wordings()
    INSTR = prompts.instructions(PT, config.FIVE); INSTR.update({(c, p): stated(PT, p, CLAUSES[c]) for c in ONE_ZERO_CORRECT_INCORRECT for p in config.FIVE})
    rows, _ = prompts.rows_of(T, task, 0); assert (np.array([r["qid"] for r in rows]) == q).all()
    model, tok, Wu, nrm = load(model_name); layers = blocks(model)
    L = len(layers); LI = [int(round(dd * L)) for dd in DEPTHS]; CAP = {}

    def mk(k_):
        def hook(_mod, _inp, outp):
            if CAP.get("on"):
                hs = outp[0] if isinstance(outp, tuple) else outp; CAP[k_] = hs[:, -1, :].detach().float().cpu().numpy()
            return outp
        return hook

    hooks = [layers[li - 1].register_forward_hook(mk(j)) for j, li in enumerate(LI)]
    KEYS = [(c, p) for p in config.FIVE for c in DEPTH_CONDITIONS]; prefix = one_zero_correct_incorrect_tokens(tok)["one_zero"]["prefix"]
    EXTRA = {("one_zero", p): [prefix] for p in config.FIVE} if prefix is not None else {}
    n = min(max_rows or 400, len(rows)); R = {k: np.zeros((n, len(LI), model.config.hidden_size), np.float16) for k in KEYS}
    for b0 in range(0, n, BATCH_SIZE):
        idx = list(range(b0, min(b0 + BATCH_SIZE, n))); texts = {k: [text(tok, rows[i], INSTR[k]) for i in idx] for k in KEYS}; pl = models.plan(tok, texts, extra=EXTRA)
        sh_ = models.Shared(model, tok, pl) if pl is not None else None
        for k in KEYS:
            CAP["on"] = True
            if sh_ is not None:
                sh_.full(k)
            else:
                ids_, am_ = models.encode_full(tok, texts[k], EXTRA.get(k), model.device)
                with torch.no_grad():
                    model(input_ids=ids_, attention_mask=am_, logits_to_keep=1)
            CAP["on"] = False
            R[k][idx] = np.stack([CAP[j] for j in range(len(LI))], 1).astype(np.float16)
        del sh_
        if b0 == 0 or (b0 // BATCH_SIZE) % 10 == 0:
            print(f"[depth] {T} {task} rows {idx[-1] + 1}/{n}  {time.time()-t0:.0f}s", flush=True)
    for h_ in hooks:
        h_.remove()
    chk = {}
    for c in LAYER_CHECK:
        lf = storage.path("layers", model=T, task=task, condition=c, wording="P01")
        if os.path.exists(lf):
            Hl = np.load(lf, allow_pickle=True)["H"]
            chk.update({f"{c}|{dd}": identity(R[(c, "P01")][:, j, :], Hl[:n, LI[j], :]) for j, dd in enumerate(DEPTHS) if dd in (0.5, 0.75)})
    bad = {k_: v for k_, v in chk.items() if v[2]}
    print(f"[depth] {T} {task} identity at P01: " + (f"MISMATCH {bad}" if bad else "OK") + f" {chk}", flush=True)
    if bad:
        raise SystemExit(3)
    models.save(path, **{key("H", c, p): v for (c, p), v in R.items()})
    print(f"[depth] wrote {path} (layers {LI} of {L})  {time.time()-t0:.0f}s", flush=True)


def run_task_depth(model_name, max_rows=0, out=None):
    T = model_name.split("/")[-1]; model, tok, _, _ = load(model_name)
    L = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else model.config.text_config.num_hidden_layers
    LI = [int(round(dd * L)) for dd in DEPTHS]; t0 = time.time()

    @torch.no_grad()
    def states(texts):
        Hs = np.zeros((len(texts), len(LI), model.config.hidden_size), np.float16)
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], return_tensors="pt", padding=True).to(model.device)
            o = model(**enc, output_hidden_states=True, logits_to_keep=1)
            Hs[i:i + len(enc.input_ids)] = torch.stack([o.hidden_states[l][:, -1, :].float() for l in LI], 1).cpu().numpy().astype(np.float16); del o
        return Hs

    stems = json.load(open(storage.path("task_tokens")))["templates"]
    for task in config.CROSS_TASKS:
        PT = task_wordings(task); rows = prompts.cross_rows(task, n=max_rows)
        for p in config.FIVE:
            for c in ("yes_no", "negated"):
                path = out_path("task_depth", out, model=T, task=task, condition=c, wording=p)
                if os.path.exists(path) and not max_rows:
                    continue
                ins = str(PT.loc[p, "yes_no"]); ins = ins if c == "yes_no" else prompts.negate_cross(ins, task)
                models.save(path, compressed=True, H=states([task_text(tok, stems[task], task, r, ins) for r in rows]))
                print(f"[cross-task-depth] {T} {task} {c} {p}  {time.time()-t0:.0f}s", flush=True)
    print(f"[cross-task-depth] done {T} layers {LI} of {L}  {time.time()-t0:.0f}s", flush=True)


def run_task_negated(model, task, max_rows=0, out=None):
    T = model.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="negated")
    if os.path.exists(path) and not max_rows:
        print(f"[cross-task-negated] {T} {task} SKIP", flush=True); return
    _, ids = config.response_tokens(cell(T, task)); names = ["Yes", "No", "A", "B"]; tid = [ids[nm] for nm in names]
    PT = task_wordings(task); stem = json.load(open(storage.path("task_tokens")))["templates"][task]; rows = prompts.cross_rows(task, n=max_rows)
    net, tok, W, nrm = load(model)
    store = {"qid": np.array([r["example_id"] for r in rows], int), "prompt_ids": np.array(config.FIVE), "token_names": np.array(names), "token_ids": np.array(tid)}
    for p in config.FIVE:
        G, Z, t20i = models.forward(net, W, nrm, tok, [task_text(tok, stem, task, r, prompts.negate_cross(str(PT.loc[p, "yes_no"]), task)) for r in rows], BATCH_SIZE, tid)
        store.update({f"G_{p}": G, f"Z_{p}": Z, f"t20i_{p}": t20i})
    models.save(path, compressed=True, **store)
    print(f"[cross-task-negated] {T} {task}", flush=True)


def response_token_rule(tok, model, probe_texts, first):
    a = tok(probe_texts[0]).input_ids

    def ctx(w):
        b = tok(probe_texts[0] + w).input_ids
        return int(b[len(a)]) if len(b) > len(a) and b[:len(a)] == a else None
    words = ("Yes", "No"); plain = {w: first(w) for w in words}; alt = {w: ctx(w) for w in words}
    if any(alt[w] is None or alt[w] == plain[w] for w in words):
        return first, "first token"
    enc = tok(probe_texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        pr = torch.softmax(model(**enc, logits_to_keep=1).logits[:, -1, :].float(), -1)
    if float(pr[:, [alt[w] for w in words]].sum(1).mean()) <= float(pr[:, [plain[w] for w in words]].sum(1).mean()):
        return first, "first token"
    return (lambda w: ctx(w) if ctx(w) is not None else first(w)), "in context"


def final_layer(model, tok, Wu, nrm, texts, cols):
    n = len(texts); G = np.zeros((n, Wu.shape[1]), np.float16); Z = np.zeros((n, len(cols)), np.float32); cols_t = torch.tensor(cols) if cols else None
    for i in range(0, n, ADDITIONAL_BATCH):
        bt = texts[i:i + ADDITIONAL_BATCH]; s = slice(i, i + len(bt)); enc = tok(bt, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True, logits_to_keep=1); g, lg = models.state(o, o.hidden_states[-1][:, -1, :], Wu, nrm)
            G[s] = g.cpu().numpy().astype(np.float16)
            if cols_t is not None:
                Z[s] = lg[:, cols_t.to(lg.device)].cpu().numpy()
            del o
    return G, Z


def run_other_models(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; ws = config.FIVE if T in FIVE_WORDING_MODELS else config.THREE
    conditions = ALL_CONDITIONS if T == "gemma-2-9b-it" else LEAN_CONDITIONS; t0 = time.time()
    d, _ = prompts.answers(T, task)
    rows = [{"qid": int(r.qid), "question": r.question, "options": (r.options if "options" in d.columns and isinstance(r.options, str) and r.options else ""), "answer": r.generated_text, "y": int(r.correct)} for _, r in d.iterrows()]
    rows = rows[:max_rows] if max_rows else rows; y = np.array([r["y"] for r in rows], int); q = np.array([r["qid"] for r in rows], int)
    PT = wordings(); INSTR = {(f, p): str(PT.loc[p, f]) for f in FORMATS for p in ws}
    for p in ws:
        stem, verb = str(PT.loc[p, "stem"]), str(PT.loc[p, "verb"])
        INSTR.update({("negated", p): prompts.negate(str(PT.loc[p, "yes_no"])), ("sentence", p): f"{stem} {verb} with exactly one of the following sentences:\n{STATEMENT_CORRECT}\n{STATEMENT_INCORRECT}",
                      **{(c, p): f"{stem} {CLAUSES[c]}" for c in WORDS}})
    model, tok = models.chat_model(model_name); chat = models.uses_chat_template(tok); Wu = model.get_output_embeddings().weight; nrm = models.final_norm(model)
    probe = [text(tok, r, str(PT.loc[ws[0], "yes_no"]), chat) for r in rows[:8]]
    first, rule = response_token_rule(tok, model, probe, lambda w: first_token(tok, w, chat))
    TID = [first(w) for w in config.SIX]; assert len(set(TID)) == 6, ("response tokens collide", TID)
    print(f"[other-models] {T} {task}: response tokens by the {rule} rule", flush=True)
    for f in FORMATS + ["negated"]:
        for p in ws:
            path = out_path("final_layer", out, model=T, task=task, condition=f, wording=p)
            if os.path.exists(path) and not max_rows:
                continue
            G, Z = final_layer(model, tok, Wu, nrm, [text(tok, r, INSTR[(f, p)], chat) for r in rows], TID)
            models.save(path, compressed=True, G=G, Z=Z, y=y, qid=q, token_names=np.array(config.SIX))
            print(f"[other-models] {T} {task} {f} {p}  {time.time()-t0:.0f}s", flush=True)
    for c in conditions:
        cols = [first(w) for w in WORDS[c]] + [first(w.capitalize()) for w in WORDS[c]] if c in WORDS else []
        for p in ws:
            path = out_path("final_layer", out, model=T, task=task, condition=c, wording=p)
            if os.path.exists(path) and not max_rows:
                continue
            G, Z = final_layer(model, tok, Wu, nrm, [text(tok, r, INSTR[(c, p)], chat) for r in rows], cols)
            models.save(path, compressed=True, G=G, **({"Z": Z} if cols else {}))
            print(f"[other-models] {T} {task} {c} {p}  {time.time()-t0:.0f}s", flush=True)


COMMANDS = {"formats": run_formats, "cross-task": run_task_formats, "negated": run_negated, "sentence": run_sentence, "format-first": run_format_first,
            "one-zero-correct-incorrect": run_one_zero_correct_incorrect, "zero-false-means-correct": run_zero_false_means_correct, "no-means-correct": run_no_means_correct,
            "more-wordings": run_more_wordings, "depth": run_depth, "cross-task-negated": run_task_negated, "other-models": run_other_models}


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest="command", required=True)
    for name in list(COMMANDS) + ["cross-task-depth"]:
        s = sub.add_parser(name); s.add_argument("--model", required=True)
        if name != "cross-task-depth":
            s.add_argument("--task", required=True)
        s.add_argument("--max_rows", type=int, default=0, help="only the first rows, for a check"); s.add_argument("--out", default=None, help="root of the written files (default data/)")
    a = ap.parse_args()
    if a.command == "cross-task-depth":
        run_task_depth(a.model, max_rows=a.max_rows, out=a.out)
    else:
        COMMANDS[a.command](a.model, a.task, max_rows=a.max_rows, out=a.out)


if __name__ == "__main__":
    main()
