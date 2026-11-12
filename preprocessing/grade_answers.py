"""Re-grading of the stored answers, compared with the stored files. Usage: python preprocessing/grade_answers.py --answers data/answers --out regraded_answers_check.csv"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_tasks import ITEMS, grade

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-14B-Instruct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="data/answers")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--tasks", nargs="*", default=list(ITEMS))
    ap.add_argument("--out", default="grade_check.csv")
    a = ap.parse_args()
    rep = []
    for t in a.tasks:
        items = {it["qid"]: it for it in ITEMS[t]()}
        for m in a.models:
            d = pd.read_parquet(f"{a.answers}/{m}/{t}.parquet")
            R = [grade(t, x, items[int(q)]) for x, q in zip(d.generated_text, d.qid)]
            ext, gold, cor, ps = (pd.Series([r[i] for r in R]) for i in range(4))
            qtext = pd.Series([items[int(q)]["question"] for q in d.qid])
            rep.append(dict(model=m, task=t, rows=len(d),
                            question_text_match=int((qtext.values == d.question.values).sum()),
                            extracted_match=int((ext.values == d.extracted_answer.astype(str).values).sum()),
                            gold_match=int((gold.values == d.gold_answer.astype(str).values).sum()),
                            correct_match=int((cor.values == d.correct.values).sum()),
                            parse_match=(int((ps.values == d.parse_success.values).sum()) if "parse_success" in d else -1)))
            print(rep[-1], flush=True)
    pd.DataFrame(rep).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
