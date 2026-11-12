"""Isolation gain on the new wordings under negation. Usage: python reply_experiments/new_wordings/gain.py"""
import os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEGATOR = re.compile(r"\b(not|incorrect\w*|inaccurate|invalid|un\w+)\b", re.I)


def main():
    R = pd.read_csv(os.path.join(HERE, "by_cell.csv"))
    R = R[R.form == "negated"].pivot_table(index=["cell", "wording"], columns="method", values="auroc")
    G = R["rank8"] - R["plain"]
    gc = G.groupby("cell").mean().to_numpy(); gw = G.groupby("wording").mean()
    rng = np.random.default_rng(0)
    bs = [gc[rng.integers(0, len(gc), len(gc))].mean() for _ in range(10000)]
    W = pd.read_csv(os.path.join(HERE, "new_wordings.csv")).set_index("wording").negated
    free = [w for w in gw.index if not NEGATOR.search(W[w])]
    rows = [("mean", gc.mean(), f"{len(gc)} cells"), ("lo", np.percentile(bs, 2.5), "2.5 percentile"), ("hi", np.percentile(bs, 97.5), "97.5 percentile"),
            ("lowest_wording", gw.min(), gw.idxmin()), ("highest_wording", gw.max(), gw.idxmax()),
            ("mean_without_negator", gw[free].mean(), " ".join(free))]
    pd.DataFrame(rows, columns=["statistic", "value", "detail"]).to_csv(os.path.join(HERE, "gain.csv"), index=False)


if __name__ == "__main__":
    main()
