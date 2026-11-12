"""Entailment and safety items, 500 per class. Usage: python preprocessing/select_task_items.py --out data/prompts_regenerated"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from datasets import load_dataset

SEED, N_PER_CLASS = 0, 500


def entailment():
    parts = []
    for r in ("test_r1", "test_r2", "test_r3"):
        d = load_dataset("facebook/anli", split=r)
        parts += [dict(uid=x["uid"], round=r, premise=x["premise"], hypothesis=x["hypothesis"], label=int(x["label"])) for x in d]
    A = pd.DataFrame(parts)
    A = A[A.label.isin([0, 2])].drop_duplicates("uid").sort_values("uid").reset_index(drop=True)
    A["y"] = (A.label == 0).astype(int)
    return A, lambda r: dict(source_id=r.uid, source_split=r["round"], y=int(r.y), p=r.premise, h=r.hypothesis)


def safety():
    d = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    B = pd.DataFrame(dict(idx=np.arange(len(d)), prompt=d["prompt"], response=d["response"],
                          is_safe=[bool(x) if not isinstance(x, str) else x == "True" for x in d["is_safe"]],
                          category=[json.dumps(x) if not isinstance(x, str) else x for x in d["category"]]))
    B = B.drop_duplicates(["prompt", "response"]).sort_values("idx").reset_index(drop=True)
    B["y"] = B.is_safe.astype(int)
    return B, lambda r: dict(source_id=int(r.idx), source_split="30k_test", y=int(r.y), q=r.prompt, r=r.response, category=r.category)


def sample(D, rec):
    n_pos, n_neg = int((D.y == 1).sum()), int((D.y == 0).sum())
    k = min(N_PER_CLASS, n_pos, n_neg)
    rng = np.random.default_rng(SEED)
    pos = D[D.y == 1].iloc[np.sort(rng.choice(n_pos, k, replace=False))]
    neg = D[D.y == 0].iloc[np.sort(rng.choice(n_neg, k, replace=False))]
    S = pd.concat([pos, neg]).reset_index(drop=True)
    S = S.iloc[rng.permutation(len(S))].reset_index(drop=True)
    out = []
    for i, (_, r) in enumerate(S.iterrows()):
        out.append(dict(example_id=i, **rec(r)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prompts_regenerated")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in (("entailment", entailment), ("safety", safety)):
        D, rec = fn()
        items = sample(D, rec)
        json.dump(items, open(f"{a.out}/{name}_items.json", "w"), indent=0)
        print(name, len(items), "items,", sum(x["y"] for x in items), "positive")


if __name__ == "__main__":
    main()
