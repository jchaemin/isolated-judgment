"""Qwen3.5-9B entries for the package and the settings of its stored answers and states."""
import functools, os, shutil
from readout import config, storage

MODEL, REVISION = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
NAME = MODEL.split("/")[-1]
CELL_TASKS = {NAME: ["arc", "gsm8k", "triviaqa", "mmlu"]}
EOS_IDS = [248046, 248044]
MAX_NEW_GSM8K = 768
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))


def cells():
    out = []
    for m in config.MODELS:
        for t in CELL_TASKS.get(m, config.TASKS):
            s = config.pd.read_csv(storage.path("cell_flags", model=m, task=t))
            out.append(dict(model=m, task=t, family="correctness", primary=bool(s.primary.iloc[0]), name=config.cell_name(m, t)))
    for t in config.CROSS_TASKS:
        for m in config.MODELS:
            out.append(dict(model=m, task=t, family=t, primary=True, name=config.cell_name(m, t)))
    return out


if NAME not in config.MODELS:
    config.MODELS.append(NAME); config.MODEL_NAMES[NAME] = NAME; config.cells = cells


def use_9b(data_root=None):
    from readout import models
    import answer_tasks
    answer_tasks.MAX_NEW["gsm8k"] = MAX_NEW_GSM8K
    tok_cls, lm_cls = models.AutoTokenizer, models.AutoModelForCausalLM

    class Tok:
        @staticmethod
        def from_pretrained(name, **kw):
            t = tok_cls.from_pretrained(name, revision=REVISION, **kw)
            t.apply_chat_template = functools.partial(t.apply_chat_template, enable_thinking=False)
            return t

    class LM:
        @staticmethod
        def from_pretrained(name, **kw):
            m = lm_cls.from_pretrained(name, revision=REVISION, **kw)
            fwd, gen = m.forward, m.generate
            m.forward = lambda *a, **k: fwd(*a, **{"logits_to_keep": 1, **k})
            m.generate = lambda *a, **k: gen(*a, **{"eos_token_id": EOS_IDS, **k})
            return m

    models.AutoTokenizer, models.AutoModelForCausalLM = Tok, LM
    models.plan = lambda *a, **k: None
    D = data_root or os.path.join(HERE, "data")
    for sub in ("answers", "cells", "labels", "states", "steering"):
        os.makedirs(os.path.join(D, sub), exist_ok=True)
        if os.path.isdir(os.path.join(ROOT, "data", sub)):
            for m in os.listdir(os.path.join(ROOT, "data", sub)):
                L = os.path.join(D, sub, m)
                if not os.path.lexists(L): os.symlink(os.path.join(ROOT, "data", sub, m), L)
    os.makedirs(os.path.join(D, "prompts"), exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "data", "prompts")):
        L = os.path.join(D, "prompts", f)
        if not os.path.lexists(L):
            if f in ("response_tokens.json", "task_tokens.json"): shutil.copy(os.path.join(HERE, "data", "prompts", f), L)
            else: os.symlink(os.path.join(ROOT, "data", "prompts", f), L)
    storage.DATA = D
