"""GPU: greedy and three sampled candidate answers for one model and task. Usage: python preprocessing/generate_answers.py --model Qwen/Qwen2.5-7B-Instruct --task arc --out answers_regenerated"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_tasks import ITEMS, MAX_NEW, SAMPLED, grade, prompt_of
from readout.models import chat_model, uses_chat_template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=list(ITEMS))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="answers_regenerated")
    a = ap.parse_args()
    T = a.model.split("/")[-1]
    items = ITEMS[a.task]()
    model, tok = chat_model(a.model)
    chat = uses_chat_template(tok)
    prompts = [tok.apply_chat_template([{"role": "user", "content": prompt_of(a.task, it)}], tokenize=False,
                                       add_generation_prompt=True) if chat else prompt_of(a.task, it) for it in items]

    def gen(sample, seed):
        if seed is not None:
            torch.manual_seed(seed)
        outs = []
        for i in range(0, len(prompts), a.batch_size):
            enc = tok(prompts[i:i + a.batch_size], return_tensors="pt", padding=True).to(model.device)
            kw = dict(max_new_tokens=MAX_NEW[a.task], pad_token_id=tok.pad_token_id)
            if sample:
                kw.update(do_sample=True, temperature=SAMPLED["temperature"], top_p=SAMPLED["top_p"])
            else:
                kw.update(do_sample=False)
            with torch.no_grad():
                g = model.generate(**enc, **kw)
            outs += [tok.decode(g[b, enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for b in range(g.shape[0])]
        return outs

    rows = []
    for cond, seed in [("greedy", None)] + [("sampled", s) for s in SAMPLED["seeds"]]:
        for it, txt in zip(items, gen(cond == "sampled", seed)):
            ext, gold, cor, ok = grade(a.task, txt, it)
            r = dict(qid=it["qid"], model=T, condition=cond, gen_seed=-1 if seed is None else seed, question=it["question"],
                     options=it["options"], generated_text=txt, extracted_answer=ext, gold_answer=gold, correct=cor)
            if a.task in ("gsm8k", "mmlu"):
                r["parse_success"] = ok
            rows.append(r)
    d = pd.DataFrame(rows)
    d.insert(2, "suite" if a.task in ("gsm8k", "mmlu") else "task", a.task)
    if a.task in ("gsm8k", "mmlu"):
        d = d.drop(columns=["options"]) if a.task == "gsm8k" else d
    os.makedirs(f"{a.out}/{T}", exist_ok=True)
    d.to_parquet(f"{a.out}/{T}/{a.task}.parquet", index=False)
    print(T, a.task, "greedy accuracy", d[d.condition == "greedy"].correct.mean())


if __name__ == "__main__":
    main()
