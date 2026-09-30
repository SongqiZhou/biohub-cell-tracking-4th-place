#!/usr/bin/env python
"""Average node division scores (<movie>.divscore.npz: keys (t, z, y, x), score) of several division CNN ensembles.

    python training/mean_division_scores.py --inputs work/div/cnn_plain,work/div/cnn_paste --out work/div/div_cnn
"""
import argparse
import glob
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--inputs', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
dirs = a.inputs.split(','); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = 0
for f in sorted(glob.glob(f'{dirs[0]}/*.divscore.npz')):
    first = np.load(f); keys = first['keys']; total = first['score'].astype(np.float32).copy()
    for d in dirs[1:]:
        z = np.load(Path(d) / Path(f).name); s = {tuple(k): v for k, v in zip(z['keys'], z['score'])}
        total += np.array([s[tuple(k)] for k in keys], np.float32)
    np.savez_compressed(out / Path(f).name, keys=keys, score=(total / len(dirs)).astype(np.float32)); n += 1
print(f'mean of {len(dirs)} division scores: {n} movies -> {out}', flush=True)
