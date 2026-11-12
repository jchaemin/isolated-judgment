"""Expression subspace, standardized bases, tuned logistic regression and pooled scoring."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import *
import estimators as Z
from estimators import stack, svd_top, rank_from, masks_coord, cross_provenance, ARMS

RULE, CENTER, SPLIT, SUB = ARMS["all3"]


def build_B(H, tr, p, q, cross, ctask, cqid, cword, fmts, use_cross, k_fixed=None):
    pairs = [H[(g, p)][tr] - H[(f, p)][tr] for i, f in enumerate(fmts) for g in fmts[i + 1:] if (f, p) in H and (g, p) in H]
    nf = len(pairs); qtr = q[tr]
    blocks = list(pairs); ids = [None] * nf; gid = None; ct = None
    if use_cross:
        idx = np.random.default_rng(SUB).choice(len(cross), min(len(cross), 4 * len(tr)), replace=False)
        cr = cross[idx]; gid = ctask[idx] * 10**8 + cqid[idx]; ct = ctask[idx]; sid = ctask[idx] * 100 + cword[idx]
        blocks.append(cr); ids.append(sid)
    if k_fixed is not None:
        k = k_fixed
    else:
        ms = masks_coord(qtr, nf, gid, ct, 0)
        V1 = svd_top(stack([b[m] for b, m in zip(blocks, ms)], [None if s is None else s[m] for s, m in zip(ids, ms)], CENTER), Z.KMAX)
        V2 = svd_top(stack([b[~m] for b, m in zip(blocks, ms)], [None if s is None else s[~m] for s, m in zip(ids, ms)], CENTER), Z.KMAX)
        k = rank_from(V1, V2, RULE)
    return svd_top(stack(blocks, ids, CENTER), k), int(k)


def std_basis(B, ss):
    Bs, _ = np.linalg.qr((B / ss.scale_).T); return Bs.T


def lr_own(X, y, q):
    vo = {C: [] for C in est.GRID}
    for itr, iva in est._inner_splits(q):
        if len(np.unique(y[itr])) < 2 or len(np.unique(y[iva])) < 2: continue
        for C in est.GRID:
            w = est._lr(X[itr], y[itr], C); vo[C].append(est.rank_auroc(y[iva], X[iva] @ w))
    C_own = max(est.GRID, key=lambda C: (np.mean(vo[C]) if vo[C] else -1, -C))
    return est._lr(X, y, C_own), C_own


def pool_put(SC, n, key, te, v):
    SC.setdefault(key, np.full(n, np.nan)); SC[key][te] = v; est.pool(SC[key], te)


def load(cell):
    S = data.load_cell(cell, config.FIVE, rows=400)
    ctask, cqid, cword = cross_provenance(cell["model"]); assert len(ctask) == len(S["cross"])
    return S, ctask, cqid, cword
