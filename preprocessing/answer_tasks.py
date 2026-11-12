"""Question sets, prompts, answer parsers and graders for the candidate answers."""
import re
import string

import numpy as np
from datasets import load_dataset

LETTERS = list(string.ascii_uppercase)
L4 = ["A", "B", "C", "D"]

ARC_PROMPT = ("Answer the following multiple-choice question. Respond with the letter of the "
              "correct option only.\n\nQuestion: {q}\n\nOptions:\n{o}\n\nAnswer:")
TRIVIA_PROMPT = ("Answer the following question with a short factual answer and nothing else.\n\n"
                 "Question: {q}\n\nAnswer:")
GSM_PROMPT = ('Solve the problem. Show your reasoning, then end with "The answer is N".\n\n'
              'Question: {q}\n\nAnswer:')
MMLU_PROMPT = ("Answer the following multiple-choice question.\n"
               "Respond with only A, B, C, or D.\n\n"
               "Question: {q}\n\n{opts}\n\nAnswer:")
MAX_NEW = {"arc": 32, "triviaqa": 32, "gsm8k": 320, "mmlu": 8}
SAMPLED = dict(temperature=0.7, top_p=0.95, seeds=(0, 1, 2))


def arc_items(n=800, seed=0):
    d = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    keep = [r for r in d if r["answerKey"] in LETTERS[:len(r["choices"]["text"])]
            or r["answerKey"] in [str(i + 1) for i in range(len(r["choices"]["text"]))]]
    idx = np.random.default_rng(seed).permutation(len(keep))[:min(n, len(keep))]
    out = []
    for j in sorted(idx):
        r = keep[int(j)]
        txts = r["choices"]["text"]
        opts = "\n".join("%s. %s" % (LETTERS[i], t) for i, t in enumerate(txts))
        out.append({"qid": len(out), "question": r["question"], "options": opts,
                    "gold": LETTERS[r["choices"]["label"].index(r["answerKey"])]})
    return out


def triviaqa_items(n=800, seed=0):
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["validation"]
    idx = np.random.default_rng(seed).permutation(len(d))[:n]
    out = []
    for j in sorted(idx):
        al = d[int(j)]["answer"]
        aliases = set(al.get("normalized_aliases") or [])
        if al.get("normalized_value"):
            aliases.add(al["normalized_value"])
        out.append({"qid": len(out), "question": d[int(j)]["question"], "options": "",
                    "gold": al.get("value", ""), "aliases": sorted(aliases)})
    return out


def gsm8k_items(n=800):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in range(len(ds)):
        r = ds[i]
        try:
            val = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        except (ValueError, IndexError):
            continue
        out.append({"qid": len(out), "question": r["question"], "options": "", "gold": val})
        if len(out) >= n:
            break
    return out


def mmlu_items(n=817, seed=0):
    ds = load_dataset("cais/mmlu", "all")["test"]
    out = []
    for qid, j in enumerate(np.random.default_rng(seed).permutation(len(ds))[:n]):
        ex = ds[int(j)]
        ch, gold = ex["choices"], int(ex["answer"])
        bad = [c for k, c in enumerate(ch) if k != gold and str(c).strip()]
        if not str(ch[gold]).strip() or not bad:
            continue
        out.append({"qid": qid, "question": ex["question"], "gold": L4[gold],
                    "options": "\n".join(f"{L4[k]}. {c}" for k, c in enumerate(ch))})
    return out


ITEMS = {"arc": arc_items, "triviaqa": triviaqa_items, "gsm8k": gsm8k_items, "mmlu": mmlu_items}


def prompt_of(task, it):
    if task == "arc":
        return ARC_PROMPT.format(q=it["question"], o=it["options"])
    if task == "triviaqa":
        return TRIVIA_PROMPT.format(q=it["question"])
    if task == "gsm8k":
        return GSM_PROMPT.format(q=it["question"])
    return MMLU_PROMPT.format(q=it["question"], opts=it["options"])


_ART = re.compile(r"\b(a|an|the)\b", re.I)
_PUNC = re.compile(r"[%s]" % re.escape(string.punctuation))


def normalize_answer(s):
    s = _PUNC.sub(" ", str(s).lower())
    return " ".join(_ART.sub(" ", s).split())


def first_line(t):
    return str(t).strip().split("\n")[0].strip()


def triviaqa_exact(pred, aliases):
    p = normalize_answer(pred)
    return int(bool(p) and any(p == normalize_answer(a) for a in aliases))


def triviaqa_contains(pred, aliases):
    nf = normalize_answer(first_line(pred))
    for al in aliases:
        na = normalize_answer(al)
        if na and na in nf:
            return 1, al
    return 0, ""


STRICT = re.compile(r"^[\s\"'\(\[\*\-\.:]*([ABCD])(?![A-Za-z0-9])")
FALL = re.compile(r"(?<![A-Za-z0-9])([ABCD])(?![A-Za-z0-9])")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_arc(txt):
    for ch in str(txt).strip():
        if ch.upper() in "ABCDEFGH":
            return ch.upper()
    return ""


def parse_mmlu(text):
    s = text.strip()
    m = STRICT.match(s)
    if m:
        return m.group(1).upper(), True
    hits = set(FALL.findall(s.split("\n")[0]))
    if len(hits) == 1:
        return hits.pop(), True
    t = s.split("\n")[0].strip()
    if len(t) <= 3 and t.strip("().  ").upper() in L4 and t.strip("().  "):
        return t.strip("().  ").upper(), True
    return None, False


def parse_gsm(text):
    low = text.lower()
    seg = text[low.rfind("answer is") + len("answer is"):] if "answer is" in low else text
    ms = NUM.findall(seg) or NUM.findall(text)
    if not ms:
        return None, False
    try:
        return float(ms[0 if "answer is" in low else -1].replace(",", "")), True
    except ValueError:
        return None, False


def grade(task, text, it):
    if task == "arc":
        p = parse_arc(text)
        return p, it["gold"], int(p.strip().upper() == it["gold"]), True
    if task == "triviaqa":
        p = first_line(text)
        return p, it["gold"], triviaqa_exact(p, it["aliases"]), True
    if task == "gsm8k":
        ext, ok = parse_gsm(text)
        return ("" if ext is None else repr(ext)), repr(it["gold"]), int(ok and ext is not None and abs(ext - it["gold"]) < 1e-6), bool(ok)
    ext, ok = parse_mmlu(text)
    return (ext or ""), it["gold"], int(ok and ext == it["gold"]), bool(ok)
