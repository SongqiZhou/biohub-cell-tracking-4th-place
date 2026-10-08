#!/usr/bin/env python
"""Detections -> per-movie node table <out>/<movie>.npz with t, um (z, y, x in um), vox (full-res voxels), seed (max cell
probability of the cluster) and size (cluster size). Nodes are sorted by (t, z, y, x); a node without a score gets
seed 1 and the median size.
usage: nodes_table.py --nodes DET_DIR --out DIR   (DET_DIR holds <movie>.nodes.npy and <movie>.scores.npy from detect.py)"""
import argparse
import glob
from pathlib import Path

import numpy as np

from common import SCALE_ZYX

ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
S = np.asarray(SCALE_ZYX, np.float64)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); total = 0
for f in sorted(glob.glob(f'{a.nodes}/*.nodes.npy')):
    name = Path(f).name.replace('.nodes.npy', ''); P = np.load(f).astype(np.float64)
    P = P[np.lexsort((P[:, 3], P[:, 2], P[:, 1], P[:, 0]))]
    seed = np.ones(len(P), np.float32); size = np.zeros(len(P), np.float32); hit = 0
    sp = Path(f'{a.nodes}/{name}.scores.npy')
    if sp.exists():
        Q = np.load(f).astype(np.float64); sc = np.load(sp).astype(np.float32)
        key = {tuple(np.round(q, 3)): k for k, q in enumerate(Q)}
        for i, p in enumerate(P):
            k = key.get(tuple(np.round(p, 3)))
            if k is not None:
                seed[i], size[i] = sc[k, 0], sc[k, 1]; hit += 1
        if hit < len(P):
            size[size == 0] = float(np.median(sc[:, 1]))
    np.savez_compressed(out / f'{name}.npz', t=P[:, 0].astype(np.int64), um=(P[:, 1:] * S[None]).astype(np.float32),
                        vox=P[:, 1:].astype(np.float32), seed=seed, size=size)
    total += len(P)
print(f'node tables: {total} nodes -> {out}', flush=True)
