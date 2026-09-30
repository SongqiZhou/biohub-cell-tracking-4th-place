#!/usr/bin/env python
"""Geometric / competition / density features for every candidate link (26 columns, FEATS). A first greedy linking pass
gives each node a velocity and a local tissue flow (median displacement of confident neighbouring links), so the features
include motion residuals as well as distances, ranks and local density. Writes <out>/<movie>.npz (I, J, X).
usage: edge_features.py --nodes DIR --cand DIR --out DIR"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import link_greedy

FEATS = ['d', 'dz', 'dy', 'dx', 'd_nnd', 'nnd_i', 'nnd_j', 'seed_i', 'seed_j', 'size_i', 'size_j', 'rank_ij', 'rank_ji', 'ncand_i', 'ncand_j',
         'd2_i', 'd2_j', 'vel_res', 'vel_known', 'tis_res', 'tis_n', 'fut_j', 'past_i', 'mutual', 'z_i', 't_frac']


def compute(z, I, J):
    t = z['t']; um = z['um'].astype(np.float64); seed = z['seed']; size = z['size']; N = len(t); T = int(t.max()) + 1
    d = np.linalg.norm(um[J] - um[I], axis=1); dv = um[J] - um[I]
    nnd = np.full(N, 15.0)                                           # nearest-neighbour distance in the same frame
    for tt in range(T):
        ii = np.where(t == tt)[0]
        if len(ii) > 1:
            nnd[ii] = np.minimum(cKDTree(um[ii]).query(um[ii], k=2)[0][:, 1], 15.0)
    _, E = link_greedy([um[t == tt] for tt in range(T)], gate_um=7.0, div_gate_um=5.0)
    pred = {}; succ = {}
    for s_, d_ in E:
        pred.setdefault(int(d_), []).append(int(s_)); succ.setdefault(int(s_), []).append(int(d_))
    vel = np.zeros((N, 3)); vk = np.zeros(N)
    for n in range(N):
        p = pred.get(n, [])
        if len(p) == 1:
            vel[n] = um[n] - um[p[0]]; vk[n] = 1
    vel_res = np.linalg.norm(um[J] - (um[I] + vel[I]), axis=1)
    # tissue flow: first-pass links shorter than 6 um are anchors; a source's flow is the median anchor displacement within 3 NND
    tis_res = d.copy(); tis_n = np.zeros(len(I))
    anc_s = np.array([s_ for s_, d_ in E if np.linalg.norm(um[d_] - um[s_]) < 6.0], int)
    anc_v = np.array([um[d_] - um[s_] for s_, d_ in E if np.linalg.norm(um[d_] - um[s_]) < 6.0]).reshape(-1, 3)
    if len(anc_s):
        for tt in range(T - 1):
            sel = np.where(t[anc_s] == tt)[0]; qi = np.where(t[I] == tt)[0]
            if not len(sel) or not len(qi):
                continue
            tree = cKDTree(um[anc_s[sel]]); flow = {}
            for s_ in np.unique(I[qi]):
                lst = [k for k in tree.query_ball_point(um[s_], 3 * max(nnd[s_], 4.0)) if anc_s[sel][k] != s_]
                if len(lst) >= 2:
                    flow[int(s_)] = (np.median(anc_v[sel][lst], 0), len(lst))
            for q in qi:
                f = flow.get(int(I[q]))
                if f:
                    tis_res[q] = np.linalg.norm(dv[q] - f[0]); tis_n[q] = f[1]
    # ranks and competition inside the candidate pool
    rank_ij = np.zeros(len(I)); rank_ji = np.zeros(len(I)); ncand_i = np.zeros(len(I)); ncand_j = np.zeros(len(I))
    d2_i = np.full(len(I), 15.0); d2_j = np.full(len(I), 15.0)
    byi = {}; byj = {}
    for q in range(len(I)):
        byi.setdefault(int(I[q]), []).append(q); byj.setdefault(int(J[q]), []).append(q)
    for lst in byi.values():
        o = np.argsort(d[lst]); ds = d[lst][o]
        for r_, q in enumerate(np.asarray(lst)[o]):
            rank_ij[q] = r_; ncand_i[q] = len(lst); d2_i[q] = ds[1] if len(ds) > 1 else 15.0
    for lst in byj.values():
        o = np.argsort(d[lst]); ds = d[lst][o]
        for r_, q in enumerate(np.asarray(lst)[o]):
            rank_ji[q] = r_; ncand_j[q] = len(lst); d2_j[q] = ds[1] if len(ds) > 1 else 15.0
    mutual = ((rank_ij == 0) & (rank_ji == 0)).astype(np.float32)
    fut_j = np.array([1.0 if succ.get(int(j)) else 0.0 for j in J]); past_i = np.array([1.0 if pred.get(int(i)) else 0.0 for i in I])
    X = np.stack([d, dv[:, 0], dv[:, 1], dv[:, 2], d / np.maximum(nnd[I], 1), nnd[I], nnd[J], seed[I], seed[J], size[I], size[J], rank_ij, rank_ji,
                  ncand_i, ncand_j, np.minimum(d2_i, 15), np.minimum(d2_j, 15), np.minimum(vel_res, 15), vk[I], np.minimum(tis_res, 15), tis_n,
                  fut_j, past_i, mutual, um[I][:, 0], t[I] / T], 1).astype(np.float32)
    return X


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--cand', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = 0
    for f in sorted(glob.glob(f'{a.nodes}/*.npz')):
        z = np.load(f); c = np.load(f'{a.cand}/{Path(f).name}'); I, J = c['I'], c['J']
        np.savez_compressed(out / Path(f).name, I=I, J=J, X=compute(z, I, J)); n += 1
    print(f'edge features: {n} movies -> {out}', flush=True)
