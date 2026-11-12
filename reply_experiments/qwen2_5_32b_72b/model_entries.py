"""Qwen2.5-32B and Qwen2.5-72B entries for the package and the settings of their stored answers and states."""
import os, shutil
from readout import config, storage

REPOS = {"Qwen2.5-32B-Instruct": ("Qwen/Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd", False),
         "Qwen2.5-72B-Instruct": ("Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8", "4f953f7634fef56affc59a40656ebe7461f7e545", True)}
NAMES = {"Qwen2.5-32B-Instruct": "Qwen2.5-32B", "Qwen2.5-72B-Instruct": "Qwen2.5-72B"}
CELL_TASKS = {m: ["gsm8k", "mmlu", "arc", "triviaqa"] for m in REPOS}
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


for _m in REPOS:
    if _m not in config.MODELS:
        config.MODELS.append(_m); config.MODEL_NAMES[_m] = NAMES[_m]
config.cells = cells


def link_view(data_root=None):
    D = data_root or os.path.join(HERE, "data")
    for sub in ("answers", "cells", "labels", "states", "steering"):
        os.makedirs(os.path.join(D, sub), exist_ok=True)
        if os.path.isdir(os.path.join(ROOT, "data", sub)):
            for m in os.listdir(os.path.join(ROOT, "data", sub)):
                L = os.path.join(D, sub, m)
                if m not in REPOS and not os.path.lexists(L): os.symlink(os.path.join(ROOT, "data", sub, m), L)
    os.makedirs(os.path.join(D, "prompts"), exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "data", "prompts")):
        L = os.path.join(D, "prompts", f)
        if not os.path.lexists(L):
            if f in ("response_tokens.json", "task_tokens.json"): shutil.copy(os.path.join(HERE, "data", "prompts", f), L)
            else: os.symlink(os.path.join(ROOT, "data", "prompts", f), L)
    storage.DATA = D


class _Keep1:
    def __init__(self, net): self.net = net
    def __call__(self, **kw): return self.net(**kw, logits_to_keep=1)
    def __getattr__(self, k): return getattr(self.net, k)


def use_32b_72b(model, data_root=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    from readout import models
    import answer_tasks, extract_states, regrade_triviaqa, cell_flags
    assert model in REPOS, model
    load0, fwd0 = models.load_model, models.forward

    def load_model(model_name, device=None):
        T = model_name.split("/")[-1]
        if T not in REPOS: return load0(model_name, device)
        repo, sha, gptq = REPOS[T]
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False, revision=sha)
        kw = dict(quantization_config=GPTQConfig(bits=8, backend="gptq_triton")) if gptq else {}
        m = AutoModelForCausalLM.from_pretrained(repo, revision=sha, device_map={"": 0}, dtype=torch.float16 if gptq else models.DTYPE, trust_remote_code=False, **kw)
        m.eval()
        return m, tok

    models.load_model = load_model
    models.LARGE_4BIT.pop("Qwen2.5-32B-Instruct", None)
    models.forward = lambda net, *a, **k: fwd0(_Keep1(net), *a, **k)
    answer_tasks.SAMPLED["seeds"] = ()
    for name, f in list(extract_states.COMMANDS.items()):
        def run(m_, task, _f=f, **kw):
            extract_states.BATCH_SIZE = extract_states.batch_for(task)
            return _f(m_, task, **kw)
        extract_states.COMMANDS[name] = run
    regrade_triviaqa.MODELS = [model]; cell_flags.MODELS = [model]
    link_view(data_root)
