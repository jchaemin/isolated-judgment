"""GPU: states at four prompt positions (--check_only: token positions only, CPU). Usage: python preprocessing/write_position_states.py --model Qwen/Qwen2.5-7B-Instruct --task arc --out states_regenerated"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import config, models, prompts, storage

WORDINGS = ["P01", "P04", "P07"]
POSITIONS = ["q_end", "a_end", "stem_end", "ans"]
BATCH = 16


def wordings():
    import pandas as pd
    return pd.read_csv(storage.path("wordings")).set_index("wording")


def build(tok, row, instr, stem):
    q = row["question"] + (("\n\n" + row["options"]) if row["options"] else "")
    c = prompts.TEMPLATE.format(q=row["question"], o=("\n\n" + row["options"]) if row["options"] else "", a=row["answer"], instr=instr)
    w = models.wrap(tok, c)
    k = w.find(c)
    kq = c.find(q)
    e1 = k + kq + len(q)
    ka = c.find(row["answer"], kq + len(q))
    e2 = k + ka + len(row["answer"])
    e3 = k + c.rfind(stem) + len(stem)
    return w, [e1, e2, e3]


def token_positions(tok, text, ends):
    off = tok(text, return_offsets_mapping=True)["offset_mapping"]
    pos = []
    for e in ends:
        hit = [k for k, (s_, t_) in enumerate(off) if s_ <= e - 1 < t_ and t_ > s_]
        pos.append(hit[-1])
    return pos + [len(off) - 1], len(off)


def prompt_set(tok, rows, PT, p):
    stem = str(PT.loc[p, "stem"])
    out = {}
    for name, instr in (("yn", str(PT.loc[p, "yes_no"])), ("noclause", stem)):
        texts, idx, ntok = [], [], []
        for r in rows:
            w, ends = build(tok, r, instr, stem)
            pos, n = token_positions(tok, w, ends)
            texts.append(w); idx.append(pos); ntok.append(n)
        out[name] = (texts, np.array(idx, np.int32), np.array(ntok, np.int32))
    return out, stem, str(PT.loc[p, "yes_no"])


def check_only(a):
    T = a.model.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(a.model)
    assert tok.is_fast
    PT = wordings()
    rows, _ = prompts.answer_rows(T, a.task)
    bad = 0
    for p in WORDINGS:
        ps, _, _ = prompt_set(tok, rows, PT, p)
        ref = np.load(a.reference.format(model=T, task=a.task, wording=p), allow_pickle=True)
        for name in ("yn", "noclause"):
            n = min(len(rows), len(ref[f"idx_{name}"]))
            ok_i = bool((ps[name][1][:n] == ref[f"idx_{name}"][:n]).all())
            ok_n = bool((ps[name][2][:n] == ref[f"ntok_{name}"][:n]).all())
            bad += (not ok_i) + (not ok_n)
            print(f"{T} {a.task} {p} {name}: rows={n} idx_equal={ok_i} ntok_equal={ok_n}")
    sys.exit(1 if bad else 0)


def write(a):
    T = a.model.split("/")[-1]
    model, tok = models.chat_model(a.model)
    assert models.uses_chat_template(tok) and tok.is_fast
    nrm = models.final_norm(model)
    last_block = model.model.layers[-1]
    PT = wordings()
    rows, _ = prompts.answer_rows(T, a.task)
    names = ["Yes", "No", "True", "False", "A", "B"]
    _, ids = config.response_tokens(dict(model=T, task=a.task, family="correctness"))
    tid = torch.tensor([ids[n] for n in names])
    cap = {}
    hook = last_block.register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    for p in WORDINGS:
        ps, stem, instr_yn = prompt_set(tok, rows, PT, p)
        store = dict(positions=np.array(POSITIONS), y=np.array([r["y"] for r in rows]), qid=np.array([r["qid"] for r in rows]),
                     token_names=np.array(names), token_ids=tid.numpy(), stem=np.array(stem), instr_yn=np.array(instr_yn))
        for name, (texts, idx, ntok) in ps.items():
            H = np.zeros((len(rows), 4, model.config.hidden_size), np.float16)
            Z = np.zeros((len(rows), 6), np.float32); lse = np.zeros(len(rows), np.float32); t20 = np.zeros((len(rows), 20), np.int32)
            for i in range(0, len(rows), BATCH):
                enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True).to(model.device)
                nb = enc["input_ids"].shape[0]
                pad = enc["input_ids"].shape[1] - enc["attention_mask"].sum(1)
                with torch.no_grad():
                    o = model(**enc, logits_to_keep=1)
                    h = cap["h"]
                    ix = torch.tensor([[int(pad[b]) + int(q) for q in idx[i + b]] for b in range(nb)], device=h.device)
                    g = torch.gather(h, 1, ix[:, :, None].expand(-1, -1, h.shape[2]))
                    H[i:i + nb] = nrm(g.to(next(nrm.parameters()).device)).float().cpu().numpy().astype(np.float16)
                    lg = o.logits[:, -1, :].float()
                    Z[i:i + nb] = lg[:, tid.to(lg.device)].cpu().numpy(); lse[i:i + nb] = torch.logsumexp(lg, -1).cpu().numpy()
                    t20[i:i + nb] = torch.topk(lg, 20, dim=-1).indices.cpu().numpy()
            store.update({f"H_{name}": H, f"idx_{name}": idx, f"ntok_{name}": ntok, f"Z_{name}": Z})
            if name == "noclause":
                store.update(lse_noclause=lse, t20i_noclause=t20)
        path = os.path.join(a.out, storage.LAYOUT["position"].format(model=T, task=a.task, wording=p))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **store)
        print("wrote", path, flush=True)
    hook.remove()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face id, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default="states_regenerated")
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--reference", default="data/states/{model}/{task}/position/{wording}.npz")
    a = ap.parse_args()
    check_only(a) if a.check_only else write(a)
