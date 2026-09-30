#!/usr/bin/env python
"""Training rows of the fork verifier (inference/fork_verify.py), from a tracking run on out-of-fold inputs (run_fork.sh).

Every row is a (mother time, two daughter positions) event, described by the verifier's image traces (117 features) and
their separation trends (40 features). Two kinds of rows per movie:
  * forks (kind 0): every node with two children in the solved graph, daughters ordered by link probability as at
    inference. Label from the official division score of the solved graph: 1 for a fork paired with an annotated
    division, 0 for a rejected or unpaired fork, unknown (left out) otherwise.
  * events (kind 1): for every node whose out-of-fold division prior is >= --min-prior (0.02), the three pairs of its three most probable
    candidate children, plus every such pair that is an annotated division. Label: 1 if the mother and both daughters
    match an annotated division, 0 if the mother matches an annotated node with one child, or with two children but this
    is not their pair; unknown otherwise. Events at the mothers of the graph's forks are left out.

    python training/fork_rows.py --graph work/fork/graph --nodes work/link/nodes --labels work/link/labels \
        --edges work/link/edges_oof --prior work/fork/div_prior --gt data/train --data data/train --out work/fork/rows
"""
import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'inference'))
from common import SCALE_ZYX, topk_edges  # noqa: E402
from fork_verify import Frames, forks_of, trace, trend_features  # noqa: E402
from link_labels import annotated_graph  # noqa: E402

A = None


def official_fork_labels(name, nodes, edges):
    """1 / 0 / -1 per two-child node of the graph, from the competition's division score"""
    import polars as pl
    import tracksdata as td
    from tracking_cellmot.division_metrics import score_divisions
    g = td.graph.InMemoryGraph()
    for key in ('z', 'y', 'x'):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    vox = np.rint(nodes[:, 1:] / SCALE_ZYX[None])
    ids = g.bulk_add_nodes([{'t': int(t), 'z': float(z), 'y': float(y), 'x': float(x)} for t, (z, y, x) in zip(nodes[:, 0], vox)])
    if len(edges):
        g.bulk_add_edges([{'source_id': ids[int(s)], 'target_id': ids[int(d)]} for s, d in edges])
    gt = td.graph.IndexedRXGraph.from_geff(str(Path(A.gt) / f'{name}.geff'))[0]
    scores = score_divisions(g, gt, scale=tuple(float(v) for v in SCALE_ZYX))
    row = {int(v): k for k, v in enumerate(ids)}
    tp = {row[int(v)] for v in scores.tp_forks}; fp = {row[int(v)] for v in scores.fp_forks}
    return tp, fp


def event_labels(gm, children, T):
    y = np.full(len(T), -1, np.int8)
    for k, (s, a, b) in enumerate(gm[T]):
        kids = children.get(int(s), [])
        if len(kids) == 1:
            y[k] = 0
        elif len(kids) == 2:
            y[k] = int(set(kids) == {int(a), int(b)})
    return y


def movie_rows(name):
    g = np.load(Path(A.graph) / f'{name}.npz'); nodes, E, p, velocity, index = g['nodes'], g['E'].astype(np.int64), g['p'], g['velocity'], g['node_index']
    z = np.load(Path(A.nodes) / f'{name}.npz'); t = z['t'].astype(int); um = z['um'].astype(np.float32); vox = np.rint(z['vox']).astype(int); N = len(t)
    gm = np.load(Path(A.labels) / f'{name}.npz')['gt_match']
    _, _, gt_e = annotated_graph(Path(A.gt) / f'{name}.geff')
    children = {}
    for s, d in gt_e:
        children.setdefault(int(s), []).append(int(d))

    F = forks_of(E, p)
    if len(F):
        tp, fp = official_fork_labels(name, nodes, E)
        y_fork = np.array([1 if int(s) in tp else 0 if int(s) in fp else -1 for s in F[:, 0]], np.int8)
    else:
        y_fork = np.zeros(0, np.int8)

    e = np.load(Path(A.edges) / f'{name}.npz'); _, oc, _ = topk_edges(e['I'], e['p'], e['J'], N, k=3)
    d = np.load(Path(A.prior) / f'{name}.divscore.npz'); prior = dict(zip(map(tuple, d['keys'].tolist()), d['score']))
    s_prior = np.array([prior.get((int(t[k]), *vox[k].tolist()), 0.0) for k in range(N)])
    T = []
    for ra, rb in ((0, 1), (0, 2), (1, 2)):
        s = np.flatnonzero(oc[:, rb] >= 0); T.append(np.column_stack([s, oc[s, ra], oc[s, rb]]))
    T = np.concatenate(T).reshape(-1, 3); y_ev = event_labels(gm, children, T)
    keep = ((s_prior[T[:, 0]] >= A.min_prior) | (y_ev == 1)) & (y_ev >= 0) & ~np.isin(T[:, 0], index[F[:, 0]] if len(F) else [])
    T, y_ev = T[keep], y_ev[keep]

    frames = Frames(Path(A.data) / f'{name}.zarr'); shift = np.cumsum(velocity, axis=0).astype(np.float32)
    sel = y_fork >= 0; X = []
    if sel.any():
        x, _, _ = trace(frames, shift, nodes[F[sel, 0], 0].astype(int), nodes[F[sel][:, 1:], 1:].astype(np.float32)); X.append(x)
    if len(T):
        x, _, _ = trace(frames, shift, t[T[:, 0]], um[T[:, 1:]]); X.append(x)
    x = np.concatenate(X) if X else np.zeros((0, 117), np.float32)
    X = np.column_stack([x, trend_features(x)]) if len(x) else np.zeros((0, 157), np.float32)
    y = np.concatenate([y_fork[sel], y_ev]); kind = np.r_[np.zeros(int(sel.sum()), np.int8), np.ones(len(T), np.int8)]
    np.savez_compressed(Path(A.out) / f'{name}.npz', X=X.astype(np.float32), y=y, kind=kind)
    return name, int(sel.sum()), int((y_fork == 1).sum()), len(T), int((y_ev == 1).sum())


def main():
    global A
    ap = argparse.ArgumentParser()
    for k in ('graph', 'nodes', 'labels', 'edges', 'prior', 'gt', 'data', 'out'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--min-prior', type=float, default=0.02); ap.add_argument('--jobs', type=int, default=16)
    A = ap.parse_args(); Path(A.out).mkdir(parents=True, exist_ok=True)
    names = sorted(Path(f).stem for f in glob.glob(f'{A.graph}/*.npz')); tot = np.zeros(4, int)
    with ProcessPoolExecutor(A.jobs) as ex:
        for name, nf, nfp, ne, nep in ex.map(movie_rows, names):
            tot += [nf, nfp, ne, nep]
    print(f'fork rows: {tot[0]} labelled forks ({tot[1]} divisions), {tot[2]} labelled events ({tot[3]} divisions) -> {A.out}', flush=True)


if __name__ == '__main__':
    main()
