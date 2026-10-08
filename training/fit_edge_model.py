#!/usr/bin/env python
"""Fit the edge model: LightGBM on the 26 edge features plus the extra evidence columns, on labelled candidate links.

    python training/fit_edge_model.py --feat work/link/feat --labels work/link/labels --root work/link \
        --extra tf_fwd,tf_rev,tf_harm,embed --out models/edge_lgbm

<root>/edges_<extra>/<movie>.npz hold the out-of-fold evidence (I, J, p) in candidate order: tf_* from the five fold edge
transformers, embed from the two fold cell-embedding encoders. Only links with a label (y >= 0) are used.

With --oof DIR nothing is saved as a model: two models are fitted on the two halves of the movies (every second one by sorted
name), and each scores the other half -> DIR/<movie>.npz (I, J, p). These out-of-fold link probabilities are the training input of the division
models (run_division.sh).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from edge_model import extra_cols  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--feat', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--root', required=True)
ap.add_argument('--extra', default='tf_fwd,tf_rev,tf_harm,embed'); ap.add_argument('--out', default=''); ap.add_argument('--oof', default='')
ap.add_argument('--n-jobs', type=int, default=32)
a = ap.parse_args()
extras = [e for e in a.extra.split(',') if e]
files = sorted(glob.glob(f'{a.feat}/*.npz'))


def load(f):
    z = np.load(f); y = np.load(Path(a.labels) / Path(f).name)['y']
    P = [np.load(Path(a.root) / f'edges_{e}' / Path(f).name)['p'] for e in extras]
    return z, np.concatenate([z['X'], extra_cols(z['I'], z['J'], P)], 1), y


def fit(fs):
    Xs, ys = [], []
    for f in fs:
        _, X, y = load(f); m = y >= 0; Xs.append(X[m]); ys.append(y[m])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               verbose=-1, n_jobs=a.n_jobs).fit(X, y)
    return model, X, y


if a.oof:
    out = Path(a.oof); out.mkdir(parents=True, exist_ok=True); halves = [files[0::2], files[1::2]]
    for k in range(2):
        model, _, y = fit(halves[1 - k])
        for f in halves[k]:
            z, X, _ = load(f)
            np.savez_compressed(out / Path(f).name, I=z['I'], J=z['J'], p=model.predict_proba(X)[:, 1].astype(np.float32))
        print(f'edge model, half {k} held out: fitted on {len(y)} labelled links, scored {len(halves[k])} movies -> {out}', flush=True)
else:
    model, X, y = fit(files)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / 'model.txt'))
    json.dump(dict(extras=extras, n_features=int(X.shape[1])), open(out / 'spec.json', 'w'), indent=1)
    print(f'edge model: {len(y)} labelled links ({int(y.sum())} positive), {X.shape[1]} features -> {out}', flush=True)
