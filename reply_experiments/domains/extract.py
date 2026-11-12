"""GPU: states for one model on BBQ and ToxiGen. Usage: python reply_experiments/domains/extract.py --model M [--tasks bbq,toxigen] [--offload]"""
import argparse, json, os, sys, time
import numpy as np
RE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
from readout import config, models, prompts, storage
from preprocessing import extract_states as ex, extract_variants as ev
CONDITIONS = ["true_false", "sentence", "zero_means_correct", "false_means_correct", "no_means_correct"]


def inject(task):
    t = json.load(open(os.path.join(RE, task, f"data/prompts/{task}_tables.json")))
    ev.POSITIVE[task] = tuple(t["POSITIVE"]); ev.SUBJECT[task] = tuple(t["SUBJECT"]); ev.STATEMENTS[task] = tuple(t["STATEMENTS"])
    prompts.CROSS_NEGATIONS[task] = [tuple(x) for x in t["CROSS_NEGATIONS"]]


def offload():
    import transformers
    _orig = transformers.AutoModelForCausalLM.from_pretrained
    def fp(name, **kw):
        n = transformers.AutoConfig.from_pretrained(name).num_hidden_layers
        dm = {"model.embed_tokens": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": 0}
        for i in range(n): dm[f"model.layers.{i}"] = "cpu" if 10 <= i < 34 else 0
        kw.pop("max_memory", None); kw["device_map"] = dm; return _orig(name, **kw)
    transformers.AutoModelForCausalLM.from_pretrained = fp


def setup(task):
    names = ["Yes", "No", "A", "B"]; return names, ex.task_wordings(task), json.load(open(storage.path("task_tokens")))["templates"][task]


def task_formats(task, T, net, tok, W, nrm, rows, wordings, root):
    names, PT, stem = setup(task); ids = json.load(open(storage.path("task_tokens")))["tokens"][T]; tid = [int(ids[k]) for k in names]
    for f in ex.TASK_FORMATS:
        path = os.path.join(root, f"states/{T}/{task}/{f}.npz")
        if os.path.exists(path): print("[formats] SKIP", path, flush=True); continue
        store = dict(y=np.array([r["y"] for r in rows], int), qid=np.array([r["example_id"] for r in rows], int), prompt_ids=np.array(wordings), token_names=np.array(names), token_ids=np.array(tid))
        for p in wordings:
            G, Z, _ = models.forward(net, W, nrm, tok, [ex.task_text(tok, stem, task, r, str(PT.loc[p, f])) for r in rows], ex.BATCH_SIZE, tid); store.update({f"G_{p}": G, f"Z_{p}": Z})
            print(f"[formats] {T} {task} {f} {p}", flush=True)
        models.save(path, compressed=True, **store)


def negated(task, T, net, tok, W, nrm, rows, wordings, root):
    path = os.path.join(root, f"states/{T}/{task}/negated.npz")
    if os.path.exists(path): print("[negated] SKIP", flush=True); return
    names, PT, stem = setup(task); ids = json.load(open(storage.path("task_tokens")))["tokens"][T]; tid = [int(ids[k]) for k in names]
    store = dict(qid=np.array([r["example_id"] for r in rows], int), prompt_ids=np.array(wordings), token_names=np.array(names), token_ids=np.array(tid))
    for p in wordings:
        G, Z, _ = models.forward(net, W, nrm, tok, [ex.task_text(tok, stem, task, r, prompts.negate_cross(str(PT.loc[p, "yes_no"]), task)) for r in rows], ex.BATCH_SIZE, tid); store.update({f"G_{p}": G, f"Z_{p}": Z})
        print(f"[negated] {T} {task} {p}", flush=True)
    models.save(path, compressed=True, **store)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tasks", default="bbq,toxigen"); ap.add_argument("--offload", action="store_true"); a = ap.parse_args()
    if a.offload: offload()
    T = a.model.split("/")[-1]; wordings = list(config.FIVE); cached = None
    for task in a.tasks.split(","):
        root = os.path.join(RE, task, "data"); storage.DATA = root; inject(task)
        if os.path.exists(os.path.join(root, f"states/{T}/{task}/conditions.npz")): print("[task] SKIP", T, task, flush=True); continue
        rows = prompts.cross_rows(task, n=0)
        if cached is None:
            cached = ev.load(a.model); ex.load = ev.load = lambda *_: cached
        model, tok, Wu, nrm = cached; t0 = time.time()
        task_formats(task, T, model, tok, Wu, nrm, rows, wordings, root); negated(task, T, model, tok, Wu, nrm, rows, wordings, root)
        ev.TASK_CONDITIONS = CONDITIONS; ev.run_tasks(a.model, task, max_rows=0, out=None)
        print("wrote", T, task, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__": main()
