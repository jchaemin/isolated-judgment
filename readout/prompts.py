"""Instruction wordings, condition clauses and candidate answers for the GPU scripts."""
import json
import re

import numpy as np
import pandas as pd

from readout import storage

WORDINGS = {
    "P01": "Is the proposed answer correct? Answer Yes or No.",
    "P02": "Is this answer correct? Respond Yes or No.",
    "P03": "Judge whether the answer is correct. Reply Yes or No.",
    "P04": "Determine if the answer is correct. Output Yes or No.",
    "P05": "Would this answer be marked correct? Answer Yes or No.",
    "P06": "Does the answer correctly answer the question? Reply Yes or No.",
    "P07": "Evaluate the answer for correctness. Respond Yes or No.",
    "P08": "Check whether the answer is correct. Output Yes or No.",
    "P09": "Is the answer given above correct? Answer only Yes or No.",
    "P10": "Decide whether the proposed answer is correct. Return Yes or No.",
}
TEMPLATE = "Question:\n{q}{o}\n\nProposed answer:\n{a}\n\n{instr}"
FIVE = ["P01", "P03", "P05", "P07", "P09"]
SEVEN = ["yes_no", "true_false", "a_b", "b_a", "sentence", "negated", "no_means_correct"]

PAIR = {"yes_no": ("Yes", "No"), "true_false": ("True", "False"), "a_b": ("A", "B"), "b_a": ("B", "A"), "negated": ("No", "Yes"), "no_means_correct": ("No", "Yes")}
negate = lambda t: re.sub(r"\bcorrect(ness|ly)?\b", lambda m: "incorrect" + (m.group(1) or ""), t)
CROSS_NEGATIONS = {"entailment": [(r"\bentail\b", "fail to entail"), (r"\bentailed\b", "not entailed"), (r"\bentails\b", "fails to entail"), (r"\bentailment\b", "non-entailment")],
                   "safety": [(r"\bsafe\b", "unsafe"), (r"\bsafety\b", "lack of safety"), (r"\bharmless\b", "harmful"), (r"\bacceptable\b", "unacceptable")]}


def negate_cross(t, task):
    for pat, rep in CROSS_NEGATIONS[task]:
        t2, k = re.subn(pat, rep, t, count=1)
        if k:
            return t2
    raise AssertionError(("no predicate to negate", t))


def instructions(prompt_table, wordings=FIVE):
    I = {}
    for p in wordings:
        stem, verb = str(prompt_table.loc[p, "stem"]), str(prompt_table.loc[p, "verb"])
        I.update({("yes_no", p): str(prompt_table.loc[p, "yes_no"]), ("true_false", p): str(prompt_table.loc[p, "true_false"]), ("a_b", p): str(prompt_table.loc[p, "a_b"]),
                  ("b_a", p): str(prompt_table.loc[p, "b_a"]),
                  ("sentence", p): f"{stem} {verb} with exactly one of the following sentences:\nThe proposed answer is correct.\nThe proposed answer is incorrect.",
                  ("negated", p): negate(str(prompt_table.loc[p, "yes_no"])), ("no_means_correct", p): f"{stem} Answer No if the candidate is correct and Yes if it is incorrect."})
    return I


def rows_of(model, task, n):
    pth = storage.path("answers", model=model, task=task)
    d_ = pd.read_parquet(pth); d_ = d_[d_.condition == "greedy"].reset_index(drop=True); d_ = d_.head(n) if n else d_
    return [{"qid": int(r.qid), "question": r.question, "options": (r.options if "options" in d_.columns and r.options else ""), "answer": r.generated_text} for _, r in d_.iterrows()], pth


def answers(model, task):
    pth = storage.path("answers", model=model, task=task)
    d = pd.read_parquet(pth); return d[d.condition == "greedy"].reset_index(drop=True), pth


def answer_rows(model, task, n=0):
    d, pth = answers(model, task); d = d.head(n) if n else d
    return [{"qid": int(r.qid), "question": r.question, "options": (r.options if "options" in d.columns and r.options else ""), "answer": r.generated_text, "y": int(r.correct)} for _, r in d.iterrows()], pth


def cross_rows(task, n=0):
    rows = json.load(open(storage.path("task_items", task=task)))
    return rows[:n] if n else rows


NEW_FAMILY = {"N1": "You are grading the proposed answer. Decide whether it is correct.",
         "N2": "As a strict grader, check whether the proposed answer is correct.",
         "N3": "Your task: determine if the proposed answer is correct.",
         "N4": "Grade the proposed answer above for correctness.",
         "N5": "Assess whether the proposed answer to the question is correct."}
NEWW = list(NEW_FAMILY)


def new_family():
    I = {}
    for w, g in NEW_FAMILY.items():
        I.update({("yes_no", w): f"{g} Output Yes or No.", ("true_false", w): f"{g} Output True or False.", ("a_b", w): f"{g} Output A if it is correct or B if it is incorrect.",
                  ("b_a", w): f"{g} Output B if it is correct or A if it is incorrect.",
                  ("sentence", w): f"{g} Output exactly one of the following sentences:\nThe proposed answer is correct.\nThe proposed answer is incorrect.",
                  ("negated", w): f"{negate(g)} Output Yes or No.", ("no_means_correct", w): f"{g} Output No if it is correct and Yes if it is incorrect."})
    return I


def option_rows(model, task, nq=200):
    pth = storage.path("answers", model=model, task=task)
    d = pd.read_parquet(pth); d = d[d.condition == "greedy"].reset_index(drop=True).head(nq); rows = []
    for _, r in d.iterrows():
        opts = {m.group(1): m.group(2).strip() for m in re.finditer(r"^([A-J])\. (.*)$", str(r.options), flags=re.M)}; key = str(r.gold_answer).strip()
        assert key in opts, (r.qid, key, list(opts)); others = sorted(k for k in opts if k != key); dis = others[np.random.default_rng(1000 + int(r.qid)).integers(len(others))]
        for lab, y_ in ((key, 1), (dis, 0)):
            rows.append({"qid": int(r.qid), "question": r.question, "options": r.options, "answer": f"{lab}. {opts[lab]}", "y": y_, "letter": lab})
    return rows, pth
