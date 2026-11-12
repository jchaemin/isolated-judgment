"""TriviaQA labels under the original and the regraded grader. Usage: python preprocessing/regrade_triviaqa.py --answers data/answers --out labels_regenerated --compare data/labels"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_tasks import first_line, triviaqa_contains, triviaqa_exact, triviaqa_items

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-14B-Instruct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="data/answers")
    ap.add_argument("--out", default="labels_regenerated")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    al = {it["qid"]: it["aliases"] for it in triviaqa_items()}
    for m in MODELS:
        d = pd.read_parquet(f"{a.answers}/{m}/triviaqa.parquet")
        g = d[d.condition == "greedy"].sort_values("qid").reset_index(drop=True)
        R = [triviaqa_contains(t, al[int(q)]) for t, q in zip(g.generated_text, g.qid)]
        out = pd.DataFrame(dict(qid=g.qid.astype(int), original=[triviaqa_exact(first_line(t), al[int(q)]) for t, q in zip(g.generated_text, g.qid)],
                                regraded=[r[0] for r in R]))
        assert (out.original.values == g.correct.values).all(), "grader a must reproduce the stored labels"
        os.makedirs(f"{a.out}/{m}", exist_ok=True)
        out.to_csv(f"{a.out}/{m}/triviaqa.csv", index=False)
        out.assign(matched_alias=[r[1] for r in R]).to_csv(f"{a.out}/{m}/triviaqa_matched_alias.csv", index=False)
        msg = f"{m}: n={len(out)} exact positives={out.original.sum()} containment positives={out.regraded.sum()} changed={(out.original != out.regraded).sum()}"
        if a.compare:
            ref = pd.read_csv(f"{a.compare}/{m}/triviaqa.csv")
            msg += f" | identical to shipped file: {ref.equals(out)}"
        print(msg)


if __name__ == "__main__":
    main()
