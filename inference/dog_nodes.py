"""Sensitive DoG detections used as a support source by the track gate (track_gate.py).

Waits for dog_route.json, then runs the DoG tracker with a lower relative threshold on every movie whose spacing is
>= --min-sp and saves its nodes as <out-dir>/<movie>.npy (t, z, y, x in full-resolution voxels).
usage: dog_nodes.py --test DIR --route dog_route.json --out-dir DIR [--min-sp 19.7] [--rel 0.01] [--procs 2]"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--test', required=True)
ap.add_argument('--route', required=True)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--min-sp', type=float, default=19.7)
ap.add_argument('--rel', type=float, default=0.01)
ap.add_argument('--procs', type=int, default=2)
ap.add_argument('--wait-s', type=float, default=7200.0)
a = ap.parse_args()


def load_route():
    t0 = time.time()
    while not os.path.exists(a.route):
        if time.time() - t0 > a.wait_s:
            raise TimeoutError(f'no {a.route} after {a.wait_s:.0f}s')
        time.sleep(5)
    return {m: float(v['dog_sp']) for m, v in json.load(open(a.route)).items()}


def dog_nodes(m):
    import dog_tracker as dt
    g = dt.track_movie(os.path.join(a.test, m + '.zarr'), dt.Config(**{**dt.ROUTER_CONFIG, 'rel_threshold': a.rel}))
    np.save(os.path.join(a.out_dir, m + '.npy'), np.stack([g.node_t, g.node_z, g.node_y, g.node_x], 1).astype(np.float32))
    return m, int(g.n_nodes)


if __name__ == '__main__':
    t0 = time.time()
    sp = load_route()
    os.makedirs(a.out_dir, exist_ok=True)
    todo = sorted(m for m, s in sp.items() if s >= a.min_sp)
    res = []
    if todo:
        with Pool(max(1, min(a.procs, len(todo)))) as p:
            res = p.map(dog_nodes, todo)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write(json.dumps({m: n for m, n in res}))
    print(f'dog_nodes: {len(res)} of {len(sp)} movies with spacing >= {a.min_sp} um, rel_threshold {a.rel}, {time.time() - t0:.0f}s', flush=True)
