#!/bin/bash
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for c in position format-first steering regularization; do python analysis/readout.py $c; done
python analysis/subspace.py all
python analysis/generalization.py all
python analysis/verdicts.py all
python analysis/tables.py --root results --data data --out results/tables
python analysis/figure_2a.py --root results --out results/tables
