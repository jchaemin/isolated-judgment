"""Models, tasks, cells, conditions, wordings, prompt template and response tokens."""
import json

import pandas as pd

from readout import storage

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-14B-Instruct"]
TASKS = ["gsm8k", "mmlu", "arc", "triviaqa"]
CROSS_TASKS = ["entailment", "safety"]

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
TEN = [f"P{i:02d}" for i in range(1, 11)]
FIVE = ["P01", "P03", "P05", "P07", "P09"]
THREE = ["P01", "P05", "P09"]

FORMATS = ["yes_no", "true_false", "a_b", "b_a", "sentence"]
HELD_OUT = ["negated", "zero_means_correct", "false_means_correct", "no_means_correct"]
CONDITIONS = FORMATS + ["one_zero", "correct_incorrect", "negated", "zero_means_correct", "false_means_correct", "no_means_correct"]

MODEL_NAMES = {
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B", "gemma-2-27b-it": "Gemma-2-27B",
    "Mistral-Nemo-Instruct-2407": "Mistral-Nemo-12B", "Ministral-8B-Instruct-2410": "Ministral-8B",
    "OLMo-2-1124-7B-Instruct": "OLMo-2-7B",
}
TASK_NAMES = {"gsm8k": "GSM8K", "mmlu": "MMLU", "arc": "ARC", "triviaqa": "TriviaQA"}
SIX = ["Yes", "No", "True", "False", "A", "B"]

def model_name(model):
    return MODEL_NAMES[model]


def cell_name(model, task):
    return f"{model_name(model)} {TASK_NAMES.get(task, task.capitalize())}"


def cells():
    out = []
    for m in MODELS:
        for t in TASKS:
            s = pd.read_csv(storage.path("cell_flags", model=m, task=t))
            out.append(dict(model=m, task=t, family="correctness", primary=bool(s.primary.iloc[0]), name=cell_name(m, t)))
    for t in CROSS_TASKS:
        for m in MODELS:
            out.append(dict(model=m, task=t, family=t, primary=True, name=cell_name(m, t)))
    return out


def response_tokens(cell):
    model = cell["model"]
    rt = json.load(open(storage.path("response_tokens"))); ids = rt["tokens"][model]
    if cell["family"] == "correctness":
        des = {f: (v["correct"], v["incorrect"]) for f, v in rt["formats"].items()}
    else:
        tt = json.load(open(storage.path("task_tokens")))
        des = {f: (v["positive"], v["negative"]) for f, v in tt["formats"].items()}
        for k, v in tt["tokens"][model].items():
            assert int(ids[k]) == int(v), ("task token ids differ from the correctness token ids", k)
    return des, {k: int(ids[k]) for k in SIX}
