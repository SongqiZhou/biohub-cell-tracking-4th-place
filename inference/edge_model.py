#!/usr/bin/env python
"""Learned link probability: LightGBM on the 26 edge features plus, for every extra evidence column (transformer forward /
reverse / harmonic probability, embedding similarity), the value, its rank and gap to the best among the source's and
among the target's candidates, and |fwd - rev|. Writes <out>/<movie>.npz (I, J, p).
usage: edge_model.py --model models/edge_lgbm --feat DIR --root DIR --out DIR   (extras are read from <root>/edges_<name>)"""
import argparse
import glob
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


def extra_cols(I, J, P):
    cols = []
    for p in P:
        r_i = np.zeros(len(I)); gap_i = np.zeros(len(I)); r_j = np.zeros(len(I)); gap_j = np.zeros(len(I))
        for key, rr, gg in ((I, r_i, gap_i), (J, r_j, gap_j)):
            by = {}
            for q in range(len(I)):
                by.setdefault(int(key[q]), []).append(q)
            for lst in by.values():
                lst = np.asarray(lst); o = np.argsort(-p[lst]); mx = p[lst].max()
                for rk, q in enumerate(lst[o]):
                    rr[q] = rk; gg[q] = mx - p[q]
        cols += [p, r_i, gap_i, r_j, gap_j]
    if len(P) >= 2:
        cols.append(np.abs(P[0] - P[1]))
    return np.stack(cols, 1).astype(np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--feat', required=True); ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True); a = ap.parse_args()
    spec = json.load(open(f'{a.model}/spec.json')); bst = lgb.Booster(model_file=f'{a.model}/model.txt'); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = 0
    for f in sorted(glob.glob(f'{a.feat}/*.npz')):
        z = np.load(f); P = [np.load(f'{a.root}/edges_{e}/{Path(f).name}')['p'] for e in spec['extras']]
        X = np.concatenate([z['X'], extra_cols(z['I'], z['J'], P)], 1)
        assert X.shape[1] == spec['n_features'], (X.shape, spec['n_features'])
        np.savez_compressed(out / Path(f).name, I=z['I'], J=z['J'], p=bst.predict(X).astype(np.float32)); n += 1
    print(f'edge model [{",".join(spec["extras"])}]: {n} movies -> {out}', flush=True)
