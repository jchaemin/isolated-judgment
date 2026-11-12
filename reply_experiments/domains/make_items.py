"""BBQ and ToxiGen items. Usage: python reply_experiments/domains/make_items.py --task {bbq,toxigen} --src DIR"""
import argparse, hashlib, json, os, subprocess
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RE = os.path.dirname(HERE); FIRST = 400
BBQ_CATS = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance", "Race_ethnicity", "Religion", "SES", "Sexual_orientation"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def git_rev(d):
    try: return subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], text=True).strip()
    except Exception: return None


def quota(n, k, rng):
    q = np.full(k, n // k); q[rng.choice(k, n - q.sum(), replace=False)] += 1; return q


def stratified_first(strata, n_first, rng):
    strata = np.asarray(strata); idx = rng.permutation(len(strata)); first = []
    labs, cnt = np.unique(strata, return_counts=True); exact = cnt * n_first / len(strata); take = np.floor(exact).astype(int)
    take[np.argsort(-(exact - take))[: n_first - take.sum()]] += 1
    for lab, t in zip(labs, take): first += [i for i in idx if strata[i] == lab][:t]
    first = list(rng.permutation(first)); rest = [i for i in idx if i not in set(first)]
    return np.array(first + rest)


def bbq(src):
    rng = np.random.default_rng(0); files = {c: os.path.join(src, "data", f"{c}.jsonl") for c in BBQ_CATS}
    meta = pd.read_csv(os.path.join(src, "supplemental", "additional_metadata.csv"), low_memory=False)
    meta = meta[meta.category.isin(BBQ_CATS)].dropna(subset=["target_loc"]); tl = {(r.category, int(r.example_id)): int(r.target_loc) for r in meta.itertuples()}
    ex = []
    for c, f in files.items():
        for line in open(f):
            e = json.loads(line); k = (c, int(e["example_id"]))
            if k not in tl: continue
            unk = [i for i in range(3) if e["answer_info"][f"ans{i}"][1] == "unknown"]
            if len(unk) != 1 or tl[k] == unk[0]: continue
            e["target_loc"] = tl[k]; e["unknown_loc"] = unk[0]; e["template"] = f"{c}|{e['question_index']}"; ex.append(e)
    E = pd.DataFrame(ex).drop_duplicates(subset=["context", "question"]); strata = [(c, p) for c in BBQ_CATS for p in ("neg", "nonneg")]
    picks = {}
    for cond, n in (("ambig", 300), ("disambig", 150)):
        q = quota(n, len(strata), rng); chosen = []
        for (c, p), m in zip(strata, q):
            pool = E[(E.category == c) & (E.question_polarity == p) & (E.context_condition == cond)]
            chosen += list(rng.choice(pool.index.to_numpy(), m, replace=False))
        picks[cond] = list(rng.permutation(chosen))
    A, Dd = picks["ambig"], picks["disambig"]
    order = list(rng.permutation(A[:100] + Dd[:50])) + list(rng.permutation(A[100:] + Dd[50:]))
    tmpl_id = {t: i for i, t in enumerate(sorted(E.template.unique()))}; items = []
    for ei in order:
        e = E.loc[ei]; opts = range(3) if e.context_condition == "ambig" else [i for i in range(3) if i != e.unknown_loc]
        non_target = [i for i in range(3) if i not in (e.target_loc, e.unknown_loc)][0]
        consistent = e.target_loc if e.question_polarity == "neg" else non_target
        for i in opts:
            role = "unknown" if i == e.unknown_loc else ("target" if i == e.target_loc else "non_target")
            items.append(dict(example_id=tmpl_id[e.template], bbq_example=f"{e.category}|{int(e.example_id)}", template=e.template, category=e.category,
                              polarity=e.question_polarity, context_condition=e.context_condition, option=int(i), role=role,
                              stereotype_consistent=bool(i == consistent and e.context_condition == "ambig"), y=int(i == e.label),
                              q=f"Context: {e.context}\nQuestion: {e.question}", r=str(e[f"ans{i}"])))
    n_first = sum(1 for _ in items[:FIRST]); assert n_first == FIRST and len({it["bbq_example"] for it in items[:FIRST]}) == 150
    src_info = dict(dataset="BBQ", url="https://github.com/nyu-mll/BBQ", commit=git_rev(src), files={os.path.relpath(p, src): sha(p) for p in list(files.values()) + [os.path.join(src, "supplemental", "additional_metadata.csv")]})
    return items, src_info


def toxigen(src):
    rng = np.random.default_rng(0); pt, pr = (os.path.join(src, "annotated", f"{s}-00000-of-00001.parquet") for s in ("test", "train"))
    te, tr = pd.read_parquet(pt), pd.read_parquet(pr); norm = lambda s: " ".join(str(s).lower().split())
    lab = lambda h: 1 if h <= 2 else (0 if h >= 4 else -1)
    te["y"] = te.toxicity_human.map(lab); te["split"] = "test"
    need = {1: max(0, 500 - int((te.y == 1).sum())), 0: max(0, 500 - int((te.y == 0).sum()))}
    tr = tr[~tr.text.map(norm).isin(set(te.text.map(norm)))].copy(); tr["y"] = tr.toxicity_human.map(lab); tr["split"] = "train"
    add = [tr.loc[rng.choice(tr.index[tr.y == k].to_numpy(), m, replace=False)] for k, m in need.items() if m]
    D = pd.concat([te] + add, ignore_index=True); assert not D.text.map(norm).duplicated().any()
    perm = stratified_first(D.y.to_numpy(), FIRST, rng); D = D.loc[perm].reset_index(drop=True)
    items = [dict(example_id=i, y=int(r.y), toxicity_human=float(r.toxicity_human), source_split=r.split, target_group=str(r.target_group), q=str(r.text)) for i, r in D.iterrows()]
    src_info = dict(dataset="toxigen/toxigen-data (annotated)", url="https://huggingface.co/datasets/toxigen/toxigen-data", revision="ea082a8973a287097b17973343576e0c8da8eb3b",
                    files={"annotated/test-00000-of-00001.parquet": sha(pt), "annotated/train-00000-of-00001.parquet": sha(pr)}, added_train=need)
    return items, src_info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--src", required=True); a = ap.parse_args()
    items, info = {"bbq": bbq, "toxigen": toxigen}[a.task](a.src)
    keys = [(it["q"], it.get("r")) for it in items]; assert len(set(keys)) == len(keys), "duplicate items"
    y = np.array([it["y"] for it in items]); info.update(n_items=len(items), n_first=FIRST, y_counts={int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
                                                       y_counts_first400={int(k): int(v) for k, v in zip(*np.unique(y[:FIRST], return_counts=True))}, seed=0)
    out = os.path.join(RE, a.task, "data", "prompts"); os.makedirs(out, exist_ok=True)
    json.dump(items, open(os.path.join(out, f"{a.task}_items.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(info, open(os.path.join(out, f"{a.task}_source.json"), "w"), indent=1); print(json.dumps({k: v for k, v in info.items() if k != "files"}))
