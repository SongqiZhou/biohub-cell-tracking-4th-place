#!/usr/bin/env python
"""Candidate links t -> t+1: all pairs within --r um, plus the --k nearest successors of every node and the --k-in
nearest predecessors of every node. Writes <out>/<movie>.npz (I, J).
usage: candidates.py --nodes DIR --out DIR [--r 10 --k 5 --k-in 3]"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def candidate_pairs(t, um, r=10.0, k=5, k_in=3):
    I = []; J = []
    T = int(t.max()) + 1; idx = {tt: np.where(t == tt)[0] for tt in range(T)}
    for tt in range(T - 1):
        a = idx.get(tt, np.zeros(0, int)); b = idx.get(tt + 1, np.zeros(0, int))
        if not len(a) or not len(b):
            continue
        A = um[a].astype(np.float64); B = um[b].astype(np.float64); tb = cKDTree(B); pairs = set()
        for ai, lst in enumerate(tb.query_ball_point(A, r)):
            for bj in lst:
                pairs.add((ai, bj))
        _, jj = tb.query(A, k=min(k, len(B))); jj = np.atleast_2d(jj).reshape(len(A), -1)
        for ai in range(len(A)):
            for bj in jj[ai]:
                pairs.add((ai, int(bj)))
        if k_in > 0:
            _, ii = cKDTree(A).query(B, k=min(k_in, len(A))); ii = np.atleast_2d(ii).reshape(len(B), -1)
            for bj in range(len(B)):
                for ai in ii[bj]:
                    pairs.add((int(ai), bj))
        for ai, bj in pairs:
            I.append(a[ai]); J.append(b[bj])
    return np.asarray(I, np.int64), np.asarray(J, np.int64)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--r', type=float, default=10.0); ap.add_argument('--k', type=int, default=5); ap.add_argument('--k-in', type=int, default=3); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n_cand = n_node = 0
    for f in sorted(glob.glob(f'{a.nodes}/*.npz')):
        z = np.load(f); I, J = candidate_pairs(z['t'], z['um'], a.r, a.k, a.k_in)
        np.savez_compressed(out / Path(f).name, I=I, J=J); n_cand += len(I); n_node += len(z['t'])
    print(f'candidates: {n_cand} links for {n_node} nodes ({n_cand / max(n_node, 1):.2f} per node) -> {out}', flush=True)
