"""AUROC, per-fold pooling, class mean, expression subspace, tuned probe, LEACE, INLP."""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd

from readout import config

KMAX = 12
GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
K_INNER = 3


def auroc(y, s):
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(np.asarray(y)[ok])) < 2:
        return np.nan
    return rank_auroc(np.asarray(y)[ok], np.asarray(s)[ok])


def rank_auroc(y, s):
    y = np.asarray(y); r = rankdata(s); p = y == 1; n1 = p.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pool(scores, test_rows):
    v = scores[test_rows]; m = v.mean(); s = v.std()
    scores[test_rows] = (v - m) / (s if s > 0 else 1.0)


def unit(w):
    return w / np.linalg.norm(w)


def aurocs(y, s, folds, mask=None):
    y = np.asarray(y); s = np.asarray(s, dtype=float); z = s.copy()
    for _, te in folds:
        pool(z, te)
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    return auroc(y[m], z[m])


def mean_in_order(values):
    t = values[0]
    for v in values[1:]:
        t = t + v
    return t / len(values)


def aurocs_mean(y, scores, folds, mask=None):
    if not len(scores):
        return np.nan
    return mean_in_order([aurocs(y, v, folds, mask) for v in scores])


def class_mean(X, y):
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / np.linalg.norm(w)


def top_directions(D, k):
    Dc = D - D.mean(0); tot = float((Dc ** 2).sum())
    _, s, Vt = randomized_svd(Dc, n_components=min(k, min(Dc.shape) - 1), n_iter=4, random_state=0)
    return Vt, s, tot


def _half_cos(blocks, masks, kmax):
    V1, _, _ = top_directions(np.concatenate([b[m] for b, m in zip(blocks, masks)]), kmax)
    V2, _, _ = top_directions(np.concatenate([b[~m] for b, m in zip(blocks, masks)]), kmax)
    return np.abs((V1 * V2).sum(1))


def split_half_rank(blocks, groups, seed=0, kmax=None):
    kmax = kmax or KMAX; rng = np.random.default_rng(seed); masks = []
    for g in groups:
        ug = np.unique(g); half = set(rng.permutation(ug)[: len(ug) // 2].tolist())
        masks.append(np.array([x in half for x in g], dtype=bool))
    cs = _half_cos(blocks, masks, kmax); k = 0
    for c in cs:
        if c >= 0.95:
            k += 1
        else:
            break
    return max(k, 1)


def choose_k(pairs, q, kmax=None, seed=0):
    kmax = kmax or KMAX
    uq = np.unique(q); rng = np.random.default_rng(seed); half = set(rng.permutation(uq)[: len(uq) // 2].tolist())
    m = np.array([g in half for g in q])
    cs = _half_cos(pairs, [m] * len(pairs), kmax); k = 0
    for c in cs:
        if c >= 0.95:
            k += 1
        else:
            break
    return max(k, 1), cs


class Cross(np.ndarray):
    def __new__(cls, X, task, qid, wording):
        o = np.asarray(X).view(cls); o.task, o.qid, o.wording = np.asarray(task), np.asarray(qid), np.asarray(wording)
        return o

    def __array_finalize__(self, obj):
        self.task = getattr(obj, "task", None); self.qid = getattr(obj, "qid", None); self.wording = getattr(obj, "wording", None)

    def rows(self, idx):
        return Cross(np.asarray(self)[idx], self.task[idx], self.qid[idx], self.wording[idx])


def cross_block(parts):
    return Cross(np.concatenate([p[0] for p in parts]), np.concatenate([np.full(len(p[0]), p[1]) for p in parts]),
                 np.concatenate([np.asarray(p[2], dtype=np.int64) for p in parts]), np.concatenate([np.full(len(p[0]), p[3]) for p in parts]))


def wording_number(p):
    return int(str(p)[1:])


def sub_cross(cross, n_train):
    return cross.rows(np.random.default_rng(1).choice(len(cross), min(len(cross), 4 * n_train), replace=False))


def _svd_top(M, k):
    _, _, Vt = randomized_svd(M, n_components=min(k, min(M.shape) - 1), n_iter=4, random_state=0)
    return Vt


def _center(blocks, cr):
    out = [b - b.mean(0) for b in blocks]
    if cr is not None:
        X = np.asarray(cr).copy(); sid = cr.task * 100 + cr.wording
        for u in np.unique(sid):
            m = sid == u; X[m] -= X[m].mean(0)
        out.append(X)
    return np.concatenate(out)


def largest_stable_rank(V1, V2):
    K = min(len(V1), len(V2)); ok = [1]
    for j in range(1, K + 1):
        if np.linalg.svd(V1[:j] @ V2[:j].T, compute_uv=False).min() >= 0.95:
            ok.append(j)
    return max(ok)


def fit_blocks(fmt, groups, cr, k=None, seed=0):
    if k is None:
        rng = np.random.default_rng(seed); mf = []
        if fmt:
            uq = np.unique(np.concatenate(groups)); half = set(rng.permutation(uq)[: len(uq) // 2].tolist())
            mf = [np.array([x in half for x in g], dtype=bool) for g in groups]
        mc = None
        if cr is not None:
            gid = cr.task * 10 ** 8 + cr.qid; hc = set()
            for t in np.unique(cr.task):
                ug = np.unique(gid[cr.task == t]); hc |= set(rng.permutation(ug)[: len(ug) // 2].tolist())
            mc = np.array([x in hc for x in gid], dtype=bool)
        V1 = _svd_top(_center([b[m] for b, m in zip(fmt, mf)], None if cr is None else cr.rows(mc)), KMAX)
        V2 = _svd_top(_center([b[~m] for b, m in zip(fmt, mf)], None if cr is None else cr.rows(~mc)), KMAX)
        k = largest_stable_rank(V1, V2)
    return _svd_top(_center(fmt, cr), k), int(k)


def expression_subspace(H, train_rows, wording, q, cross, rule="split_half"):
    tr, p = train_rows, wording
    fmt = [H[(g, p)][tr] - H[(f, p)][tr] for i, f in enumerate(config.FORMATS) for g in config.FORMATS[i + 1:] if (f, p) in H and (g, p) in H]
    cr = sub_cross(cross, len(tr)) if cross is not None else None
    return fit_blocks(fmt, [q[tr]] * len(fmt), cr, k=8 if rule == "fixed8" else None)


def _lr(X, y, C):
    w = LogisticRegression(C=C, max_iter=3000).fit(X, y).coef_[0]
    return w / np.linalg.norm(w)


def _inner_splits(q, seed=0):
    uq = np.unique(q); rng = np.random.default_rng(seed); perm = rng.permutation(uq); out = []
    for k in range(K_INNER):
        ho = set(perm[k::K_INNER].tolist()); va = np.array([g in ho for g in q]); out.append((~va, va))
    return out


def tuned_probe(X, y, q, others=()):
    K = _inner_splits(q); vo = {C: [] for C in GRID}; vx = {C: [] for C in GRID}
    for itr, iva in K:
        if len(np.unique(y[itr])) < 2 or len(np.unique(y[iva])) < 2:
            continue
        for C in GRID:
            w = _lr(X[itr], y[itr], C); vo[C].append(rank_auroc(y[iva], X[iva] @ w))
            vx[C].append(np.mean([rank_auroc(y[iva], Xo[iva] @ w) for Xo in others]) if others else np.nan)
    C_own = max(GRID, key=lambda C: (np.mean(vo[C]) if vo[C] else -1, -C))
    C_across = max(GRID, key=lambda C: (np.nanmean(vx[C]) if vx[C] else -1, -C))
    w_own = _lr(X, y, C_own); w_across = w_own if C_across == C_own else _lr(X, y, C_across)
    return w_own, w_across, C_own, C_across


def leace(X, Z, tol=1e-6):
    mu = X.mean(0); Xc = X - mu; Zc = Z - Z.mean(0); n = len(X); Sxx = Xc.T @ Xc / n; Sxz = Xc.T @ Zc / n
    ev, V = np.linalg.eigh(Sxx); keep = ev > tol * ev.max(); ev, V = ev[keep], V[:, keep]
    isq = V @ np.diag(1 / np.sqrt(ev)) @ V.T; sq = V @ np.diag(np.sqrt(ev)) @ V.T
    W = isq @ Sxz; Q, s, _ = np.linalg.svd(W, full_matrices=False); Q = Q[:, s > 1e-8 * s.max()]; Pm = Q @ Q.T
    A = (sq @ Pm @ isq).T
    return mu, A


def inlp(X, z, iters=8, seed=0):
    d = X.shape[1]; R = np.eye(d); Xc = X.copy(); dirs = []
    for _ in range(iters):
        w = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(Xc, z).coef_
        for wi in w:
            wi = wi / (np.linalg.norm(wi) + 1e-12); dirs.append(wi)
        Q = np.linalg.qr(np.array(dirs).T)[0]; R = np.eye(d) - Q @ Q.T; Xc = X @ R
    return R
