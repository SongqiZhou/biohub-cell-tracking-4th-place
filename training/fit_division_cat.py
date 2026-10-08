#!/usr/bin/env python
"""Fit the per-node division CatBoost (inference/division_cat.py) on the out-of-fold candidate graphs and CNN scores.

Labels: a node matched to an annotated node with two annotated children is a division (1), one with a single child is
not (0); every other node (unmatched, or matched to a node whose annotation ends) is left out. 600 trees of depth 6,
learning rate 0.05, class weights SqrtBalanced, seed 0.

    python training/fit_division_cat.py --cand work/div/cand_npz --div-dir work/div/div_cnn --labels work/div/labels \
        --gt data/train --out models/division_cat.cbm
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'inference'))
from division_features import NAMES, build  # noqa: E402
from link_labels import annotated_graph  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--cand', required=True); ap.add_argument('--div-dir', required=True)
ap.add_argument('--labels', required=True, help='link_labels.py output (gt_match per node)'); ap.add_argument('--gt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--half', type=int, default=-1, choices=[-1, 0, 1], help='leave out this half of the movies (training/fold_movies.py --half)'); a = ap.parse_args()
held = set()
if a.half >= 0:
    held = set(subprocess.run([sys.executable, str(HERE / 'fold_movies.py'), '--half', str(a.half)], capture_output=True, text=True, check=True).stdout.strip().split(','))
Xs, ys = [], []
for cp in sorted(glob.glob(f'{a.cand}/*.candidates.npz')):
    name = Path(cp).name.replace('.candidates.npz', '')
    if name in held:
        continue
    X, coords = build(cp, a.div_dir)
    gm = np.load(Path(a.labels) / f'{name}.npz')['gt_match']
    gt_t, _, gt_e = annotated_graph(Path(a.gt) / f'{name}.geff')
    n_children = np.bincount(gt_e[:, 0], minlength=len(gt_t))
    nc = np.where(gm >= 0, n_children[np.maximum(gm, 0)], 0)
    y = np.full(len(coords), -1, np.int8); y[(gm >= 0) & (nc >= 2)] = 1; y[(gm >= 0) & (nc == 1)] = 0
    m = y >= 0; Xs.append(np.nan_to_num(X.astype(np.float32))[m]); ys.append(y[m])
X = np.concatenate(Xs); y = np.concatenate(ys)
model = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, auto_class_weights='SqrtBalanced', random_seed=0, verbose=0).fit(X, y)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); model.save_model(a.out)
print(f'division CatBoost: {len(y)} nodes, {int(y.sum())} divisions, {X.shape[1]} features ({len(NAMES)} named) -> {a.out}', flush=True)
