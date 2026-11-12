"""Paths and package access for the scripts in reply_experiments/."""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from readout import config, data, estimators as est, storage
storage.DATA = os.path.join(ROOT, "data")
