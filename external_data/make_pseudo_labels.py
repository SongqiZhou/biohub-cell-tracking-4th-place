#!/usr/bin/env python
"""Detections -> pseudo-labels for 3D Net training.

Input: <nodes_dir>/<movie>.nodes.npy from inference/detect.py ([t, z, y, x] full-resolution voxels).
The nodes go through the same light post-processing as the tracker output: greedy frame-to-frame linking, removal of
tracks shorter than 3 nodes, and line-fit smoothing along the tracks. Output: <out_dir>/<movie>.npz with `nodes`
(N, 4) [t, z, y, x] voxels and `scores` (N,) = 1.

    python external_data/make_pseudo_labels.py <nodes_dir> <out_dir>
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from common import SCALE_ZYX, filter_short_tracks, linefit_smooth, link_greedy  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
S = np.asarray(SCALE_ZYX, np.float64)
for f in sorted(glob.glob(str(src / '*.nodes.npy'))):
    name = Path(f).name.replace('.nodes.npy', '')
    c = np.load(f).astype(np.float64)
    T = int(c[:, 0].max()) + 1
    nodes, edges = link_greedy([(c[c[:, 0] == t][:, 1:] * S[None]).astype(np.float32) for t in range(T)])
    nodes, edges, _ = filter_short_tracks(nodes, edges, 3)
    nodes = linefit_smooth(nodes, edges, 3, 0.8)
    out = np.concatenate([nodes[:, :1], nodes[:, 1:] / S[None]], 1)
    np.savez_compressed(dst / f'{name}.npz', nodes=out.astype(np.float32), scores=np.ones(len(out), np.float32))
    print(f'{name}: {len(c)} detections -> {len(out)} pseudo-label nodes', flush=True)
