#!/usr/bin/env python
"""Per-node division probability from CatBoost on the 53 division features.
usage: division_cat.py --model models/division_cat.cbm --cand DIR --div-dir DIR --out DIR   -> <out>/<movie>.divscore.npz"""
import argparse
import glob
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from division_features import build

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--cand', required=True); ap.add_argument('--div-dir', required=True)
ap.add_argument('--out', required=True); a = ap.parse_args()
m = CatBoostClassifier(); m.load_model(a.model); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = 0
for cp in sorted(glob.glob(f'{a.cand}/*.candidates.npz')):
    name = Path(cp).name.replace('.candidates.npz', ''); X, coords = build(cp, a.div_dir); X = np.nan_to_num(X.astype(np.float32))
    np.savez_compressed(out / f'{name}.divscore.npz', keys=np.rint(coords).astype(np.int32), score=m.predict_proba(X)[:, 1].astype(np.float32)); n += 1
print(f'division CatBoost: {n} movies -> {out}', flush=True)
