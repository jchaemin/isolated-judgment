"""GPU: states for one model on ETHICS. Usage: python reply_experiments/ethics/extract.py --model M"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.getcwd())
from readout import config, data, models, prompts, storage
from preprocessing import extract_states as ex, extract_variants as ev
CONDITIONS = ["true_false", "sentence", "zero_means_correct", "false_means_correct", "no_means_correct"]

def inject():
    t = json.load(open(os.path.join(HERE, "data/prompts/ethics_tables.json")))
    ev.POSITIVE["ethics"] = tuple(t["POSITIVE"]); ev.SUBJECT["ethics"] = tuple(t["SUBJECT"]); ev.STATEMENTS["ethics"] = tuple(t["STATEMENTS"])
    prompts.CROSS_NEGATIONS["ethics"] = [tuple(x) for x in t["CROSS_NEGATIONS"]]

def task_formats(T, net, tok, W, nrm, rows, wordings, root):
    names = ["Yes", "No", "A", "B"]; ids = json.load(open(storage.path("task_tokens")))["tokens"][T]; tid = [int(ids[k]) for k in names]
    PT = ex.task_wordings("ethics"); stem = json.load(open(storage.path("task_tokens")))["templates"]["ethics"]
    for f in ex.TASK_FORMATS:
        path = os.path.join(root, f"states/{T}/ethics/{f}.npz")
        if os.path.exists(path): print("[formats] SKIP", path, flush=True); continue
        store = dict(y=np.array([r["y"] for r in rows], int), qid=np.array([r["example_id"] for r in rows], int), prompt_ids=np.array(wordings), token_names=np.array(names), token_ids=np.array(tid))
        for p in wordings:
            G, Z, _ = models.forward(net, W, nrm, tok, [ex.task_text(tok, stem, "ethics", r, str(PT.loc[p, f])) for r in rows], ex.BATCH_SIZE, tid); store.update({f"G_{p}": G, f"Z_{p}": Z})
            print(f"[formats] {T} {f} {p}", flush=True)
        models.save(path, compressed=True, **store)

def negated(T, net, tok, W, nrm, rows, wordings, root):
    path = os.path.join(root, f"states/{T}/ethics/negated.npz")
    if os.path.exists(path): print("[negated] SKIP", flush=True); return
    names = ["Yes", "No", "A", "B"]; ids = json.load(open(storage.path("task_tokens")))["tokens"][T]; tid = [int(ids[k]) for k in names]
    PT = ex.task_wordings("ethics"); stem = json.load(open(storage.path("task_tokens")))["templates"]["ethics"]
    store = dict(qid=np.array([r["example_id"] for r in rows], int), prompt_ids=np.array(wordings), token_names=np.array(names), token_ids=np.array(tid))
    for p in wordings:
        G, Z, _ = models.forward(net, W, nrm, tok, [ex.task_text(tok, stem, "ethics", r, prompts.negate_cross(str(PT.loc[p, "yes_no"]), "ethics")) for r in rows], ex.BATCH_SIZE, tid); store.update({f"G_{p}": G, f"Z_{p}": Z})
        print(f"[negated] {T} {p}", flush=True)
    models.save(path, compressed=True, **store)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    root = os.path.join(HERE, "data"); T = a.model.split("/")[-1]
    storage.DATA = root; inject(); wordings = list(config.FIVE)
    rows = prompts.cross_rows("ethics", n=0); assert len(rows) == 1000
    model, tok, Wu, nrm = ev.load(a.model); cached = (model, tok, Wu, nrm); ex.load = ev.load = lambda *_: cached
    t0 = time.time(); task_formats(T, model, tok, Wu, nrm, rows, wordings, root); negated(T, model, tok, Wu, nrm, rows, wordings, root)
    ev.TASK_CONDITIONS = CONDITIONS; ev.run_tasks(a.model, "ethics", max_rows=0, out=None)
    print("wrote", T, f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__": main()
