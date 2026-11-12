"""ETHICS commonsense items, 1000 rows. Usage: python reply_experiments/ethics/make_items.py --csv <ethics/commonsense/test.csv>"""
import argparse, json, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "data", "prompts")
ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); CSV = ap.parse_args().csv
d = pd.read_csv(CSV); short = d[d.is_short.astype(str) == "True"]; assert len(short) == 2109
rng = np.random.default_rng(0); pick = []
for lab in (0, 1):
    idx = short.index[short.label == lab].to_numpy(); pick.append(rng.choice(idx, 500, replace=False))
pick = np.concatenate(pick); rng.shuffle(pick)
items = [dict(example_id=i, source_id=int(r), source_split="test", y=int(1 - d.label[r]), q=str(d.input[r])) for i, r in enumerate(pick)]
assert len({it["source_id"] for it in items}) == 1000 and np.mean([it["y"] for it in items]) == 0.5
json.dump(items, open(os.path.join(OUT, "ethics_items.json"), "w"), indent=1, ensure_ascii=False)
