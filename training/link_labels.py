#!/usr/bin/env python
"""Ground-truth labels for candidate links (training data of the edge model and the cell embedding).

Detected nodes are matched to annotated nodes frame by frame (one-to-one, within 7 um). A candidate link i -> j gets
  y = 1   if i is matched to an annotated node whose annotated child is matched to j,
  y = 0   if i is matched to an annotated node that has children, but j is not one of them,
  y = -1  otherwise (unknown: most cells are not annotated).
Writes <out>/<movie>.npz with y (per candidate link, in the order of <cand>/<movie>.npz) and gt_match (per node).

    python training/link_labels.py --nodes work/link/nodes --cand work/link/cand_pool --gt data/train --out work/link/labels
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment

SCALE = np.array([1.625, 0.40625, 0.40625], np.float64)


def annotated_graph(geff: Path):
    """-> (t (N,), centres in um (N, 3), edges as row indices (E, 2))"""
    g = zarr.open_group(str(geff), mode='r')
    p = g['nodes/props']
    t = np.asarray(p['t/values']).astype(np.int64)
    zyx = np.stack([np.asarray(p['z/values']), np.asarray(p['y/values']), np.asarray(p['x/values'])], 1).astype(np.float64)
    ids = np.asarray(g['nodes/ids']).astype(np.int64)
    row = {int(v): i for i, v in enumerate(ids)}
    e = np.asarray(g['edges/ids']).astype(np.int64).reshape(-1, 2)
    edges = np.array([[row[int(a)], row[int(b)]] for a, b in e if int(a) in row and int(b) in row], np.int64).reshape(-1, 2)
    return t, zyx * SCALE[None], edges


def match_frame(p_um, g_um, r=7.0):
    out = np.full(len(p_um), -1, np.int64)
    if not len(p_um) or not len(g_um):
        return out
    D = np.linalg.norm(p_um[:, None, :] - g_um[None, :, :], axis=-1)
    C = np.where(D <= r, D, 1e6)
    ri, ci = linear_sum_assignment(C)
    ok = C[ri, ci] <= r
    out[ri[ok]] = ci[ok]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--cand', required=True)
    ap.add_argument('--gt', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n_pos = n_neg = 0
    for f in sorted(glob.glob(f'{a.nodes}/*.npz')):
        name = Path(f).stem; z = np.load(f); c = np.load(f'{a.cand}/{name}.npz'); I, J = c['I'], c['J']
        t = z['t']; um = z['um'].astype(np.float64)
        gt_t, gt_um, gt_e = annotated_graph(Path(a.gt) / f'{name}.geff')
        gm = np.full(len(t), -1, np.int64)
        for tt in np.unique(t):
            ii = np.where(t == tt)[0]; gg = np.where(gt_t == tt)[0]
            if len(gg):
                m = match_frame(um[ii], gt_um[gg]); ok = m >= 0; gm[ii[ok]] = gg[m[ok]]
        child = {}
        for p, q in gt_e:
            child.setdefault(int(p), set()).add(int(q))
        y = np.full(len(I), -1, np.int8)
        for k in range(len(I)):
            g = int(gm[I[k]])
            if g >= 0 and g in child:
                y[k] = 1 if int(gm[J[k]]) in child[g] else 0
        np.savez_compressed(out / f'{name}.npz', y=y, gt_match=gm)
        n_pos += int((y == 1).sum()); n_neg += int((y == 0).sum())
    print(f'link labels: {n_pos} positive, {n_neg} negative -> {out}', flush=True)
