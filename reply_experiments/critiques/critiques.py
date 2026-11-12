"""Open-ended critiques: the isolated readout against the model's reading of its own critique. Usage: python reply_experiments/critiques/critiques.py [--indir DIR]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from scope_lib import *
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = config.FIVE; B = 2000
MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]; TASKS = ["gsm8k", "triviaqa"]
NAME = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B"}


def fit_readout(model, task):
    c = [c for c in config.cells() if c["model"] == model and c["task"] == task][0]; S, ctask, cqid, cword = load(c)
    y, q, n, H, cross = (S[k] for k in ("y", "q", "n", "H", "cross")); tr = np.arange(n); W = {}
    for p in FIVE:
        ss = StandardScaler().fit(H[("yes_no", p)]); X = ss.transform(H[("yes_no", p)]); wcm = est.class_mean(X, y)
        Bm, k = build_B(H, tr, p, q, cross, ctask, cqid, cword, config.FORMATS, True, k_fixed=8); Bs = std_basis(Bm, ss); w = wcm - Bs.T @ (Bs @ wcm)
        W[p] = (ss, w)
    return W, c


def proj(W, p, G):
    ss, w = W[p]; return ss.transform(G.astype(np.float64)) @ w


def auc_rows(y, s):
    r = rankdata(s, axis=1); npos = y.sum(1); nneg = y.shape[1] - npos
    return (np.where(y == 1, r, 0).sum(1) - npos * (npos + 1) / 2) / (npos * nneg)


def ci(v): return v[0], np.percentile(v[1:], 2.5), np.percentile(v[1:], 97.5)


def run(T, task, rows, rng, OUT):
    Zs = np.load(os.path.join(OUT, T, f"oe_{task}.npz")); q = Zs["qid"]
    W, cell = fit_readout(T, task)
    S8 = data.load_cell(cell, FIVE, rows=800); qa = np.asarray(S8["q"]).astype(int); assert len(qa) == 800 and (qa[400:800] == q).all()
    c = np.asarray(S8["y"]).astype(int)[400:800]; n = len(c)
    idx = np.vstack([np.arange(n)[None], rng.integers(0, n, (B, n))]); Yb = c[idx]
    base = dict(model=NAME[T], task=task, n=n)
    a_ = np.mean([auc_rows(Yb, proj(W, p, Zs["G_a"])[idx]) for p in FIVE], 0)
    rows.append(dict(**base, quantity="auroc_state_a", arm="rank8", **dict(zip(("est", "lo", "hi"), ci(a_)))))
    j = pd.read_parquet(os.path.join(OUT, T, f"oe_judge_{T}_on_{T}_{task}.parquet")); assert (j.qid.values == q).all(); lab = j.judge.values
    s = np.select([lab == "Yes", lab == "Unclear"], [1.0, 0.5], 0.0)
    rows.append(dict(**base, quantity="critique_auroc", arm="judge_same", **dict(zip(("est", "lo", "hi"), ci(auc_rows(Yb, s[idx]))))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--indir", default=os.path.join(HERE, "out")); A = ap.parse_args()
    rows = []; rng = np.random.default_rng(0)
    for T in MODELS:
        for t in TASKS: run(T, t, rows, rng, A.indir); print(NAME[T], t, flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "oe_rows.csv"), index=False); print(R.round(3).to_string(index=False))
