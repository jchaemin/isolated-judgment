"""GPU: hidden states for the entailment and safety conditions, other models' answers, dataset candidates and a new instruction family."""
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

ROWS = 400
BATCH_SIZE = 16


def out_path(kind, out, **fields):
    return storage.path(kind, **fields) if out is None else os.path.join(out, storage.LAYOUT[kind].format(**fields))


def cell(model, task):
    return dict(model=model, task=task, family="correctness")


def load(model_name):
    model, tok = models.chat_model(model_name)
    assert models.uses_chat_template(tok)
    return model, tok, model.get_output_embeddings().weight, models.final_norm(model)


def text(tok, r, instr):
    return models.wrap(tok, prompts.TEMPLATE.format(q=r["question"], o=("\n\n" + r["options"]) if r["options"] else "", a=r["answer"], instr=instr))


def identity(A, B):
    cs = models.cosine(A.astype(np.float64), B.astype(np.float64))
    return [float(np.median(cs)), float(cs.min()), bool(models.identity_bad(cs))]


def read(model, tok, Wu, nrm, texts, n, cols=None, extra=None, norm=None, name=""):
    keys = list(texts); cols = cols or {}; extra = extra or {}; norm = [None] if norm is None else norm; t0 = time.time()
    G = {k: np.zeros((n, model.config.hidden_size), np.float16) for k in keys}; Z = {k: np.zeros((n, len(cols[k[0]])), np.float32) for k in keys if k[0] in cols}
    for b0 in range(0, n, BATCH_SIZE):
        idx = list(range(b0, min(b0 + BATCH_SIZE, n))); bt = {k: [texts[k][i] for i in idx] for k in keys}; pl = models.plan(tok, bt, extra=extra)
        sh_ = models.Shared(model, tok, pl) if pl is not None else None
        for k in keys:
            if sh_ is not None:
                last_ids, mask, npre = sh_.extend(k); cache = sh_.cache; sh_.cache.crop(npre)
            else:
                ids_, am_ = models.encode_full(tok, bt[k], extra.get(k), model.device)
                with torch.no_grad():
                    o = model(input_ids=ids_[:, :-1], attention_mask=am_[:, :-1], use_cache=True, logits_to_keep=1); cache = o.past_key_values; del o
                last_ids, mask = ids_[:, -1:], am_
            with torch.no_grad():
                o = model(input_ids=last_ids, attention_mask=mask, past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=1)
                lg = o.logits[:, -1, :].float(); h = o.hidden_states[-1][:, -1, :]
                if norm[0] is None:
                    norm[0] = models.decide_norm(h, lg, Wu, nrm)
                G[k][idx] = models.apply_state(h, norm[0], Wu, nrm).cpu().numpy().astype(np.float16)
            if k[0] in cols:
                Z[k][idx] = lg[:, cols[k[0]]].cpu().numpy()
            del o
        del sh_
        if b0 == 0 or (b0 // BATCH_SIZE) % 10 == 0:
            print(f"[{name}] rows {idx[-1] + 1}/{n}  {time.time()-t0:.0f}s", flush=True)
    return G, Z


def checked(G, stored, name):
    chk = identity(G, stored)
    print(f"[{name}] identity with the stored Yes/No states at P01, first rows: " + ("MISMATCH " if chk[2] else "OK ") + str(chk), flush=True)
    if chk[2]:
        raise SystemExit(3)


TASK_CONDITIONS = ["true_false", "sentence", "one_zero", "correct_incorrect", "zero_means_correct", "false_means_correct", "no_means_correct"]
POSITIVE = {"entailment": ("entailed", "not entailed"), "safety": ("safe", "unsafe")}
SUBJECT = {"entailment": ("the hypothesis is entailed", "it is not entailed"), "safety": ("the response is safe", "it is unsafe")}
STATEMENTS = {"entailment": ("The premise entails the hypothesis.", "The premise does not entail the hypothesis."), "safety": ("The response is safe.", "The response is unsafe.")}


def task_instructions(task, stem, verb):
    pw, nw = POSITIVE[task]; ps, ns = SUBJECT[task]; s1, s0 = STATEMENTS[task]
    return {"true_false": f"{stem} {verb} True or False.",
            "sentence": f"{stem} {verb} with exactly one of the following sentences:\n{s1}\n{s0}",
            "one_zero": f"{stem} Respond 1 if {pw}, 0 if {nw}.",
            "correct_incorrect": f"{stem} Respond with the word {pw} or {nw}.",
            "zero_means_correct": f"{stem} Respond 0 if {pw}, 1 if {nw}.",
            "false_means_correct": f"{stem} Respond False if {pw}, True if {nw}.",
            "no_means_correct": f"{stem} Answer No if {ps} and Yes if {ns}."}


def task_answers(task):
    pw, nw = POSITIVE[task]
    return {"true_false": ("True", "False"), "one_zero": ("1", "0"), "correct_incorrect": (pw, nw), "zero_means_correct": ("0", "1"), "false_means_correct": ("False", "True"),
            "no_means_correct": ("No", "Yes"), "yes_no": ("Yes", "No")}


def first_token(tok, w):
    e = tok(w, add_special_tokens=False).input_ids; pre = []
    while len(e) > 1 and tok.decode([e[0]]).strip() == "":
        pre.append(int(e[0])); e = e[1:]
    return int(e[0]), pre


def run_tasks(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="conditions")
    if os.path.exists(path) and not max_rows:
        print("[tasks] SKIP", path, flush=True); return
    stem = json.load(open(storage.path("task_tokens")))["templates"][task]; PT = pd.read_csv(storage.path("task_wordings")); PT = PT[PT.task == task].set_index("wording")
    rows = prompts.cross_rows(task); Z1 = data.states(T, task, "yes_no"); n = min(max_rows or ROWS, len(rows))
    rows = rows[:n]; q = np.array([r["example_id"] for r in rows], int)
    assert (Z1["qid"].astype(int)[:n] == q).all(), "rows differ from the stored states"
    INSTR = {(c, p): v for p in config.FIVE for c, v in task_instructions(task, str(PT.loc[p, "stem"]), str(PT.loc[p, "verb"])).items()}
    INSTR[("yes_no", "P01")] = str(PT.loc["P01", "yes_no"])
    model, tok, Wu, nrm = load(model_name); cols, extra = {}, {}
    for c, (w1, w0) in task_answers(task).items():
        (t1, pre1), (t0, _) = first_token(tok, w1), first_token(tok, w0); (a1, _), (a0, _) = first_token(tok, w1[0].upper() + w1[1:]), first_token(tok, w0[0].upper() + w0[1:])
        assert t1 != t0, (c, "the two response words share their first token", w1, w0)
        cols[c] = [t1, t0, a1, a0]
        if pre1 and c in ("one_zero", "zero_means_correct"):
            extra.update({(c, p): pre1 for p in config.FIVE})
    keys = [("yes_no", "P01")] + [(c, p) for p in config.FIVE for c in TASK_CONDITIONS]
    G, Z = read(model, tok, Wu, nrm, {k: [models.wrap(tok, stem.format(p=r.get("p"), h=r.get("h"), instr=INSTR[k]) if task == "entailment" else stem.format(q=r.get("q"), r=r.get("r"), instr=INSTR[k])) for r in rows] for k in keys},
                n, cols=cols, extra=extra, name=f"tasks {T} {task}")
    checked(G[("yes_no", "P01")], Z1["G_P01"][:n], f"tasks {T} {task}")
    store = {"qid": q, "conventions": np.array(TASK_CONDITIONS)}
    for c, p in keys[1:]:
        store.update({key("G", c, p): G[(c, p)], **({key("Z", c, p): Z[(c, p)]} if c != "sentence" else {})})
    models.save(path, **store)
    print(f"[tasks] wrote {path}", flush=True)


def run_other_model(judge, candidates, task, max_rows=0, out=None):
    J = judge.split("/")[-1]; C = candidates.split("/")[-1]; path = out_path("answers_of", out, model=J, task=task, source=C)
    if os.path.exists(path) and not max_rows:
        print("[other-model] SKIP", path, flush=True); return
    instr = prompts.instructions(pd.read_csv(storage.path("wordings")).set_index("wording")); n = max_rows or ROWS
    yc, qc = data.labels(cell(C, task), {"yes_no": data.states(C, task, "yes_no")}); yc, qc = np.asarray(yc).astype(int)[:n], np.asarray(qc)[:n]
    rows, _ = prompts.rows_of(C, task, n); assert (np.array([r["qid"] for r in rows]) == qc).all(), "candidate rows differ from the candidate model's stored states"
    own, _ = prompts.rows_of(J, task, 16)
    model, tok, Wu, nrm = load(judge); norm = [None]
    G, _ = read(model, tok, Wu, nrm, {("yes_no", "P01"): [text(tok, r, instr[("yes_no", "P01")]) for r in own]}, 16, norm=norm, name=f"other-model {J} own")
    checked(G[("yes_no", "P01")], data.states(J, task, "yes_no")["G_P01"][:16], f"other-model {J}")
    G, _ = read(model, tok, Wu, nrm, {("yes_no", p): [text(tok, r, instr[("yes_no", p)]) for r in rows] for p in config.FIVE}, len(rows), norm=norm, name=f"other-model {J} reads {C} {task}")
    models.save(path, qid=qc, y=yc, **{key("G", c, p): v for (c, p), v in G.items()})
    print(f"[other-model] wrote {path}", flush=True)


def run_instruction_family(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="instruction_family")
    if os.path.exists(path) and not max_rows:
        print("[instruction-family] SKIP", path, flush=True); return
    instr = prompts.new_family(); check = prompts.instructions(pd.read_csv(storage.path("wordings")).set_index("wording"))[("yes_no", "P01")]
    D1 = data.states(T, task, "yes_no"); n = max_rows or ROWS; q = D1["qid"].astype(int)[:n]
    rows, _ = prompts.rows_of(T, task, n); assert (np.array([r["qid"] for r in rows]) == q).all()
    model, tok, Wu, nrm = load(model_name); norm = [None]
    G, _ = read(model, tok, Wu, nrm, {("yes_no", "P01"): [text(tok, r, check) for r in rows[:16]]}, 16, norm=norm, name=f"instruction-family {T} {task} check")
    checked(G[("yes_no", "P01")], D1["G_P01"][:16], f"instruction-family {T} {task}")
    G, _ = read(model, tok, Wu, nrm, {(c, w): [text(tok, r, instr[(c, w)]) for r in rows] for w in prompts.NEWW for c in prompts.SEVEN}, len(rows), norm=norm, name=f"instruction-family {T} {task}")
    models.save(path, qid=q, **{key("G", c, w): v for (c, w), v in G.items()})
    print(f"[instruction-family] wrote {path}", flush=True)


def run_dataset_candidates(model_name, task, max_rows=0, out=None):
    T = model_name.split("/")[-1]; path = out_path("states", out, model=T, task=task, condition="dataset_candidates")
    if os.path.exists(path) and not max_rows:
        print("[dataset-candidates] SKIP", path, flush=True); return
    instr = prompts.instructions(pd.read_csv(storage.path("wordings")).set_index("wording"))
    rows, _ = prompts.option_rows(T, task); rows = rows[:max_rows] if max_rows else rows; n = len(rows)
    own, _ = prompts.rows_of(T, task, 16); _, ids = config.response_tokens(cell(T, task))
    cols = {c: [int(ids[w]) for w in prompts.PAIR[c]] for c in prompts.SEVEN if c in prompts.PAIR}
    model, tok, Wu, nrm = load(model_name); norm = [None]
    G, _ = read(model, tok, Wu, nrm, {("yes_no", "P01"): [text(tok, r, instr[("yes_no", "P01")]) for r in own]}, 16, norm=norm, name=f"dataset-candidates {T} {task} check")
    checked(G[("yes_no", "P01")], data.states(T, task, "yes_no")["G_P01"][:16], f"dataset-candidates {T} {task}")
    G, Z = read(model, tok, Wu, nrm, {(c, p): [text(tok, r, instr[(c, p)]) for r in rows] for p in config.FIVE for c in prompts.SEVEN}, n, cols=cols, norm=norm, name=f"dataset-candidates {T} {task}")
    store = {"qid": np.array([r["qid"] for r in rows], int), "y": np.array([r["y"] for r in rows], int), "conventions": np.array(prompts.SEVEN)}
    for c, p in G:
        store.update({key("G", c, p): G[(c, p)], key("Z", c, p): Z[(c, p)] if c in cols else np.zeros((n, 2), np.float32)})
    models.save(path, **store)
    print(f"[dataset-candidates] wrote {path}", flush=True)


COMMANDS = {"tasks": run_tasks, "instruction-family": run_instruction_family, "dataset-candidates": run_dataset_candidates}


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest="command", required=True)
    for name in list(COMMANDS) + ["other-model"]:
        s = sub.add_parser(name)
        if name == "other-model":
            s.add_argument("--judge", required=True); s.add_argument("--candidates", required=True)
        else:
            s.add_argument("--model", required=True)
        s.add_argument("--task", required=True, **({"choices": config.CROSS_TASKS} if name == "tasks" else {}))
        s.add_argument("--max_rows", type=int, default=0, help="only the first rows, for a check"); s.add_argument("--out", default=None, help="root of the written files (default data/)")
    a = ap.parse_args()
    if a.command == "other-model":
        run_other_model(a.judge, a.candidates, a.task, max_rows=a.max_rows, out=a.out)
    else:
        COMMANDS[a.command](a.model, a.task, max_rows=a.max_rows, out=a.out)


if __name__ == "__main__":
    main()
