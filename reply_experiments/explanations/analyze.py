"""Explanation, then verdict, under negation: readouts and the model's verdict. Usage: python reply_experiments/explanations/analyze.py [--indir DIR]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from scope_lib import *
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); FORM = "cot_negated"; SIGN = -1; ARGS = None


def job(cell):
    S, ctask, cqid, cword = load(cell); y, q = S["y"], S["q"]
    Z = np.load(os.path.join(ARGS.indir, cell["model"], cell["task"], "cot.npz")); assert (Z["qid"].astype(int) == q).all()
    pids = [p for p in config.FIVE if f"G_{FORM}_{p}" in Z]
    NEW = {(FORM, p): Z[f"G_{FORM}_{p}"].astype(np.float64) for p in pids}
    SC = score_cell(S, ctask, cqid, cword, NEW, ["plain", "rank8"])
    rows = readout_rows(cell["name"], y, SC); rows.append(own_row(cell["name"], FORM, y, {p: SIGN * Z[f"M_{FORM}_{p}"].astype(np.float64) for p in pids})); print(cell["name"], flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--indir", default=os.path.join(HERE, "out")); ap.add_argument("--procs", type=int, default=2); ARGS = ap.parse_args()
    cs = [c for c in config.cells() if c["family"] == "correctness" and c["name"] in explanation_cells()]; assert len(cs) == 6
    with Pool(min(ARGS.procs, len(cs))) as P: out = P.map(job, cs)
    R = pd.DataFrame([r for o in out for r in o]); R.to_csv(os.path.join(HERE, "by_cell.csv"), index=False)
    m = means(R); m.to_csv(os.path.join(HERE, "means.csv"), index=False); print(m.round(3).to_string(index=False))
