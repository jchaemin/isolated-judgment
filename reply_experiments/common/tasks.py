"""Cross-task states, the isolated direction and threshold counts."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
import analysis.generalization as gen
ROWS = 400


def task_states(model, task):
    return gen._task_states(model, task)


def task_prov(model):
    ct, cq, cw = [], [], []; X = []
    for ci, c in enumerate([c for c in gen._all_cells() if c["family"] == "correctness" and c["primary"] and c["model"] == model]):
        S = data.load_cell(c, wordings=config.FIVE, rows=ROWS)
        for p in config.FIVE:
            if ("negated", p) in S["H"] and ("yes_no", p) in S["H"]:
                n = len(S["q"]); ct.append(np.full(n, ci)); cq.append(S["q"].astype(np.int64)); cw.append(np.full(n, int(p[1:]))); X.append(S["H"][("negated", p)] - S["H"][("yes_no", p)])
    return (np.concatenate(ct), np.concatenate(cq), np.concatenate(cw)), np.concatenate(X)


def counts(pred, y):
    return int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())


def isolated_direction(H, tr, p, q, cross, prov, ss, wcm, basis=False):
    ctask, cqid, cword = prov
    B, _ = build_B(H, tr, p, q, cross, ctask, cqid, cword, config.FORMATS, True, k_fixed=8); Bs = std_basis(B, ss)
    w = wcm - Bs.T @ (Bs @ wcm); w = w / np.linalg.norm(w)
    return (w, Bs) if basis else w
