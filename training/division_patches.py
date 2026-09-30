#!/usr/bin/env python
"""Training patches of the division CNN, cut at the annotated nodes.

Every annotated node with at least one annotated child gives one sample: label 1 if it has two children (a division),
0 if it has one (it certainly did not divide). Nodes without children are skipped: the annotation may simply end there.
A sample is the 5-frame (t-2..t+2) raw window of 9 x 33 x 33 voxels around the node, zero outside the volume, scaled
with the movie's 0.1% / 99.9% intensity quantiles (the same crop the division CNN sees at inference).

The movies are processed in `--shards` interleaved groups (sorted names, movie i -> shard i % shards) and written as
<out>/shard<k>.npz (X float16 (N, 5, 9, 33, 33), y int8, meta 'movie|t|z|y|x'); train_division_cnn.py concatenates the
shards in order.

    python training/division_patches.py --data data/train --out work/div/patches --shards 12
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from division_cnn import HX, HY, HZ, TW, read_frame  # noqa: E402


def annotated_nodes(geff: Path):
    """-> list of (t, z, y, x, n_children) in the order of the node table"""
    g = zarr.open_group(str(geff), mode='r')
    p = g['nodes/props']
    t, z, y, x = (np.asarray(p[f'{k}/values']) for k in 'tzyx')
    ids = np.asarray(g['nodes/ids']).astype(np.int64)
    src = np.asarray(g['edges/ids']).astype(np.int64).reshape(-1, 2)[:, 0]
    n_children = dict(zip(*np.unique(src, return_counts=True)))
    return [(int(t[i]), z[i], y[i], x[i], int(n_children.get(int(ids[i]), 0))) for i in range(len(ids))]


def crop(frame, z, y, x):
    Z, Y, X = frame.shape
    out = np.zeros((2 * HZ + 1, 2 * HY + 1, 2 * HX + 1), np.float32)
    z0, z1 = max(0, z - HZ), min(Z, z + HZ + 1)
    y0, y1 = max(0, y - HY), min(Y, y + HY + 1)
    x0, x1 = max(0, x - HX), min(X, x + HX + 1)
    out[z0 - (z - HZ):z1 - (z - HZ), y0 - (y - HY):y1 - (y - HY), x0 - (x - HX):x1 - (x - HX)] = frame[z0:z1, y0:y1, x0:x1]
    return out


def movie_patches(data: Path, name: str):
    zp = data / f'{name}.zarr'
    meta = json.loads((zp / '0' / 'zarr.json').read_text())
    shape, dtype = tuple(meta['shape']), np.dtype(meta['data_type'])
    q = json.loads((zp / 'zarr.json').read_text()).get('attributes', {}).get('image_statistics', {}).get('quantiles', {})
    qlo, qhi = float(q.get('0.001', 0)), float(q.get('0.999', 1))
    by_t = {}
    for t, z, y, x, nc in annotated_nodes(data / f'{name}.geff'):
        if nc:
            by_t.setdefault(t, []).append((z, y, x, 1 if nc >= 2 else 0))
    X, Y, M, cache = [], [], [], {}
    for t in sorted(by_t):
        frames = [read_frame(zp, min(max(t + d, 0), shape[0] - 1), shape, dtype, cache) for d in range(-TW, TW + 1)]
        if any(f is None for f in frames):
            continue
        for z, y, x, lab in by_t[t]:
            z, y, x = int(round(z)), int(round(y)), int(round(x))
            cube = np.stack([crop(f, z, y, x) for f in frames])
            X.append(((cube - qlo) / max(qhi - qlo, 1e-6)).astype(np.float16)); Y.append(lab); M.append(f'{name}|{t}|{z}|{y}|{x}')
    return X, Y, M


def shard(args):
    k, names, data, out = args
    X, Y, M = [], [], []
    for name in names:
        x, y, m = movie_patches(data, name); X += x; Y += y; M += m
    X = np.stack(X) if X else np.zeros((0, 2 * TW + 1, 2 * HZ + 1, 2 * HY + 1, 2 * HX + 1), np.float16)
    np.savez_compressed(out / f'shard{k}.npz', X=X, y=np.array(Y, np.int8), meta=np.array(M))
    return k, len(Y), int(np.sum(Y))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--shards', type=int, default=12); ap.add_argument('--jobs', type=int, default=12); a = ap.parse_args()
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name[:-5] for p in data.iterdir() if p.name.endswith('.zarr'))
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, n, n_pos in ex.map(shard, [(k, names[k::a.shards], data, out) for k in range(a.shards)]):
            print(f'shard {k}: {n} patches, {n_pos} divisions', flush=True)
