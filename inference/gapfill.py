#!/usr/bin/env python
"""Bridge short track gaps: a track ending at t (no successor) and a track starting at t+k+1 (no predecessor) within
--step*(k+1) um are joined through k interpolated nodes (k = 1..--max-gap), one-to-one, nearest pairs first, shorter gaps
first. A gap is left open if an existing node already sits within --excl um of an interpolated point.
usage: gapfill.py --csv in.csv --out out.csv [--max-gap 2 --step 4 --excl 2.5]"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import SCALE_ZYX

SC = SCALE_ZYX.astype(np.float64)
COLS = ['dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
ap = argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--max-gap', type=int, default=2); ap.add_argument('--step', type=float, default=4.0); ap.add_argument('--excl', type=float, default=2.5)
ap.add_argument('--jobs', type=int, default=3)
a = ap.parse_args()


def one(args):
    m, g = args; n = g[g.row_type == 'node']; e = g[g.row_type == 'edge']
    ids = n.node_id.to_numpy(); t = n.t.to_numpy().astype(int); um = n[['z', 'y', 'x']].to_numpy().astype(float) * SC
    T = int(t.max())
    has_out = set(e.source_id.astype(int)); has_in = set(e.target_id.astype(int))
    ends = [k for k, i in enumerate(ids) if int(i) not in has_out and t[k] <= T - 2]; starts = [k for k, i in enumerate(ids) if int(i) not in has_in and t[k] >= 2]
    occ = {tt: cKDTree(um[t == tt]) for tt in np.unique(t)}
    used_e, used_s = set(), set(); new_nodes, new_edges = [], []; nid = int(ids.max()) + 1; bridged = {k: 0 for k in range(1, a.max_gap + 1)}
    for k in range(1, a.max_gap + 1):
        by_t = {}
        for s in starts:
            if s not in used_s:
                by_t.setdefault(t[s], []).append(s)
        pairs = []
        for en in ends:
            if en in used_e:
                continue
            c = by_t.get(t[en] + k + 1)
            if not c:
                continue
            d = np.linalg.norm(um[c] - um[en], axis=1)
            for j in np.where(d <= a.step * (k + 1))[0]:
                pairs.append((d[j], en, c[j]))
        pairs.sort()
        for d, en, s in pairs:
            if en in used_e or s in used_s:
                continue
            pts = []
            for i in range(1, k + 1):
                f = i / (k + 1); p_um = um[en] * (1 - f) + um[s] * f; tt = t[en] + i
                if tt in occ and occ[tt].query(p_um, k=1)[0] <= a.excl:
                    break
                pts.append((tt, p_um / SC))
            if len(pts) < k:
                continue
            chain = [int(ids[en])]
            for tt, v in pts:
                new_nodes.append((m, 'node', nid, int(tt), int(round(v[0])), int(round(v[1])), int(round(v[2])), -1, -1)); chain.append(nid); nid += 1
            chain.append(int(ids[s]))
            for u, w in zip(chain[:-1], chain[1:]):
                new_edges.append((m, 'edge', -1, -1, -1, -1, -1, u, w))
            used_e.add(en); used_s.add(s); bridged[k] += 1
    return m, pd.concat([g[COLS], pd.DataFrame(new_nodes + new_edges, columns=COLS)], ignore_index=True), bridged


if __name__ == '__main__':
    df = pd.read_csv(a.csv)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, list(df.groupby('dataset'))))
    pd.concat([r[1] for r in res], ignore_index=True)[COLS].to_csv(a.out, index=False)
    print('gap filling: ' + ', '.join(f'{k}-frame gaps bridged {sum(r[2][k] for r in res)}' for k in range(1, a.max_gap + 1)), flush=True)
