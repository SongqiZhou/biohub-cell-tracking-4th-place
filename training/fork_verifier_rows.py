#!/usr/bin/env python
"""Training rows of the fork verifier (annotations + hand labels), built from the raw movies.

training/fork_verifier/ holds the training events:
  events.npz       forks of a tracking run on the out-of-fold node set, labelled by the competition's division score
                   (fork_*), and candidate mother / daughter-pair events labelled by the annotations (event_*):
                   mother frame t, the two daughter positions in frame t + 1 (um), label
  hand_labels.csv  hand-screened division candidates: candidate divisions proposed by our models on training movies,
                   each checked by eye (verdict 1 = division, 0 = not a division);
                   mother voxel at frame `frame`, daughter voxels at frame + 1
  motion.npz       per-movie frame-to-frame translation (um) that compensates common motion in the traces
Every event is traced in the raw images exactly as at inference (inference/fork_verify.py: 117 trace features + 40
separation-trend features). Output <out>/<movie>.npz: X, y, kind (0 fork, 1 event), source (0 annotations, 1 hand label),
the input of train_fork_verifier.py.

    python training/fork_verifier_rows.py --data data/train --out work/fork_released/rows            # annotations + hand labels
    python training/fork_verifier_rows.py --data data/train --out work/fork_released/rows_nohand --no-hand-labels
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'inference'))
from common import SCALE_ZYX  # noqa: E402
from fork_verify import Frames, trace, trend_features  # noqa: E402

A = None


def hand_label_events(path):
    """movie -> list of (mother frame, daughter centres (2, 3) um, label), in file order; undecided rows skipped,
    exact duplicates kept once"""
    out, seen = {}, {}
    for row, r in enumerate(csv.DictReader(open(path)), start=2):
        v = r['verdict'].strip()
        if v == '':
            continue
        assert v in ('0', '1'), (row, v)
        xyz = np.array([[float(r[f'{p}_{a}']) for a in 'zyx'] for p in ('mother', 'd1', 'd2')])
        key = (r['volume'], int(r['frame']), tuple(xyz[0]), *sorted([tuple(xyz[1]), tuple(xyz[2])]))
        if key in seen:
            assert seen[key] == v, f'conflicting duplicate at row {row}'
            continue
        seen[key] = v
        assert abs(np.linalg.norm((xyz[1] - xyz[2]) * SCALE_ZYX) - float(r['sister_um'])) <= 0.3, f'voxel units at row {row}'
        out.setdefault(r['volume'], []).append((int(r['frame']), xyz[1:] * SCALE_ZYX.astype(np.float64), int(v)))
    return out


def movie_rows(name):
    ev = np.load(Path(A.dir) / 'events.npz'); shift = np.load(Path(A.dir) / 'motion.npz')[name].cumsum(0)
    f = ev['fork_movie'] == name; e = ev['event_movie'] == name
    ft, fc, fy = ev['fork_t'][f], ev['fork_centers'][f], ev['fork_y'][f]
    et, ec, ey = ev['event_t'][e], ev['event_centers'][e], ev['event_y'][e]
    src = np.zeros(len(ey), np.int8)
    hand = [] if A.no_hand_labels else HAND.get(name, [])
    if hand:
        et = np.r_[et, [h[0] for h in hand]]; ec = np.concatenate([ec, np.array([h[1] for h in hand], np.float32)])
        ey = np.r_[ey, [h[2] for h in hand]].astype(np.int8); src = np.r_[src, np.ones(len(hand), np.int8)]
    frames = Frames(Path(A.data) / f'{name}.zarr'); X = []
    for t, c in ((ft, fc), (et, ec)):
        if len(t):
            x = trace(frames, shift, t.astype(int), c)[0]; X.append(np.column_stack([x, trend_features(x)]))
    X = np.concatenate(X).astype(np.float32) if X else np.zeros((0, 157), np.float32)
    np.savez_compressed(Path(A.out) / f'{name}.npz', X=X, y=np.r_[fy, ey].astype(np.int8),
                        kind=np.r_[np.zeros(len(fy), np.int8), np.ones(len(ey), np.int8)], source=np.r_[np.zeros(len(fy), np.int8), src])
    return len(fy), len(ey), int(src.sum())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=str(HERE / 'fork_verifier')); ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--no-hand-labels', action='store_true'); ap.add_argument('--jobs', type=int, default=16)
    A = ap.parse_args(); Path(A.out).mkdir(parents=True, exist_ok=True)
    HAND = hand_label_events(Path(A.dir) / 'hand_labels.csv')
    names = sorted(np.load(Path(A.dir) / 'motion.npz').files)
    with ProcessPoolExecutor(A.jobs) as ex:
        res = list(ex.map(movie_rows, names))
    print(f'verifier rows: {len(names)} movies, {sum(r[0] for r in res)} forks, {sum(r[1] for r in res)} events '
          f'({sum(r[2] for r in res)} hand-labelled) -> {A.out}', flush=True)
