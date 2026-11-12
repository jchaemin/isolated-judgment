"""Points and 95% intervals of Figure 2(a). Usage: python analysis/figure_2a.py --root results [--out results/tables]"""
import argparse, os
import numpy as np, pandas as pd

FORMATS = ['yes_no', 'true_false', 'a_b', 'b_a', 'sentence', 'one_zero', 'correct_incorrect']
HO4 = ['negated', 'zero_means_correct', 'false_means_correct', 'no_means_correct']
SIGNALS = [('verdict', 'readout', 'verdict_auroc'), ('readout', 'readout', 'auroc'), ('isolated', 'isolated', 'auroc'), ('fitted', 'fitted', 'auroc')]

def stat(M):
    return np.nanmean(np.nanmean(M, axis=0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, 'tables'); os.makedirs(out, exist_ok=True)
    H = pd.read_csv(os.path.join(a.root, 'subspace', 'heldout.csv'))
    cells = list(H.cell.unique())
    rows = []
    for name, method, col in SIGNALS:
        P = H[H.method == method].pivot(index='cell', columns='condition', values=col).reindex(cells)
        for cond, conds in [('response_formats', FORMATS)] + [(c, [c]) for c in HO4]:
            M = P[conds].dropna(axis=1, how='all').to_numpy(dtype=float)
            r = dict(signal=name, condition=cond, cells=len(cells), mean=stat(M), lo=np.nan, hi=np.nan)
            if name != 'fitted':
                rng = np.random.default_rng(0)
                bs = [stat(M[rng.integers(0, len(M), len(M))]) for _ in range(10000)]
                r['lo'], r['hi'] = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'figure_2a.csv'), index=False)
    print(f'{len(rows)} rows written to {out}')

if __name__ == '__main__':
    main()
