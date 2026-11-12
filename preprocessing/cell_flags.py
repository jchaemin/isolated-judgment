"""Primary and descriptive cell flags. Usage: python preprocessing/cell_flags.py --data data --out data_regenerated/cells"""
import argparse
import os

import pandas as pd

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-14B-Instruct"]
TASKS = ["arc", "gsm8k", "mmlu", "triviaqa"]
MIN_NEG, N_ROWS_IN_FILE = 100, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data_regenerated/cells")
    a = ap.parse_args()
    for m in MODELS:
        os.makedirs(f"{a.out}/{m}", exist_ok=True)
        for t in TASKS:
            g = pd.read_parquet(f"{a.data}/answers/{m}/{t}.parquet")
            g = g[g.condition == "greedy"].sort_values("qid")
            y = g.correct.values.astype(int)
            if t == "triviaqa":
                y = pd.read_csv(f"{a.data}/labels/{m}/triviaqa.csv").regraded.values.astype(int)
            neg = int((y == 0).sum())
            pd.DataFrame({"primary": [neg >= MIN_NEG] * N_ROWS_IN_FILE}).to_csv(f"{a.out}/{m}/{t}.csv", index=False)
            print(f"{m} {t}: rows={len(y)} negatives={neg} primary={neg >= MIN_NEG}")


if __name__ == "__main__":
    main()
