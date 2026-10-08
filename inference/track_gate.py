#!/usr/bin/env python
"""Track-support gate for sparse movies. In every movie whose DoG spacing falls in a band [lo, hi), the tracks are cut into
division-free segments, and a segment is removed (nodes and their links) when fewer than theta of its nodes are supported
by an independent detection in the same frame:
  v    3D Net-64 node within r um
  or   sensitive-DoG node within r um  OR  3D Net-64 node within r2 um
  and  sensitive-DoG node within r um  AND 3D Net-64 node within r2 um
Movies outside all bands are untouched.
usage: track_gate.py --sub submission.csv --route dog_route.json --net64-dir DIR --dog-dir DIR --spec "19.7:25:or:0.4:4:5;25:99:and:0.2:5:5"
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import SCALE_ZYX

SC = SCALE_ZYX.astype(np.float64)
ap = argparse.ArgumentParser(); ap.add_argument('--sub', required=True); ap.add_argument('--route', required=True)
ap.add_argument('--net64-dir', required=True); ap.add_argument('--dog-dir', default=''); ap.add_argument('--spec', required=True); a = ap.parse_args()
BANDS = []
for tok in a.spec.split(';'):
    lo, hi, mode, th, r, r2 = tok.split(':'); assert mode in ('v', 'or', 'and'), mode
    BANDS.append((float(lo), float(hi), mode, float(th), float(r), float(r2)))


def segments(ids, src, dst):
    """segment id per node: maximal chains through single-successor / single-predecessor links"""
    outd = pd.Series(src).value_counts(); ind = pd.Series(dst).value_counts(); nxt = {}
    for s, t in zip(src, dst):
        if outd.get(s, 0) == 1 and ind.get(t, 0) == 1:
            nxt[s] = t
    prv = set(nxt.values()); seg = {}; k = 0
    for n in ids:
        if n in seg or n in prv:
            continue
        x = n
        while True:
            seg[x] = k
            if x not in nxt:
                break
            x = nxt[x]
        k += 1
    for n in ids:
        if n not in seg:
            seg[n] = k; k += 1
    return seg


def support(R, t, um, r):
    sup = np.zeros(len(t), bool); Rt = np.rint(R[:, 0]).astype(int)
    for f in np.unique(t):
        i = np.where(t == f)[0]; Rf = R[Rt == f][:, 1:] * SC
        if len(Rf):
            sup[i] = cKDTree(Rf).query(um[i], k=1)[0] <= r
    return sup


sp = {m: float(v['dog_sp']) for m, v in json.load(open(a.route)).items()}
d = pd.read_csv(a.sub); parts = []; tot_n = tot_d = 0
for m, g in d.groupby('dataset', sort=False):
    band = next((b for b in BANDS if b[0] <= sp.get(m, 0.0) < b[1]), None)
    if band is None:
        parts.append(g); continue
    lo, hi, mode, th, r, r2 = band
    nd = g[g.row_type == 'node']; ed = g[g.row_type == 'edge']; ids = nd.node_id.to_numpy(); t = nd.t.to_numpy(int); um = nd[['z', 'y', 'x']].to_numpy(float) * SC
    V = np.load(os.path.join(a.net64_dir, m + '.nodes.npy')).astype(np.float64)
    if mode == 'v':
        sup = support(V, t, um, r)
    else:
        D = np.load(os.path.join(a.dog_dir, m + '.npy')).astype(np.float64); s1 = support(D, t, um, r); s2 = support(V, t, um, r2)
        sup = (s1 | s2) if mode == 'or' else (s1 & s2)
    seg = segments(ids, ed.source_id.to_numpy(), ed.target_id.to_numpy()); s = np.array([seg[i] for i in ids])
    st = pd.DataFrame({'s': s, 'sup': sup}).groupby('s').sup.mean(); bad = st.index[st < th].to_numpy()
    badn = set(ids[np.isin(s, bad)]); tot_n += len(ids); tot_d += len(badn)
    print(f'  track gate {m}: DoG spacing {sp[m]:.2f} um, band [{lo:g},{hi:g}) {mode} theta {th:g} | support {sup.mean():.3f} | '
          f'removed {len(badn)} of {len(ids)} nodes', flush=True)
    parts.append(pd.concat([nd[~nd.node_id.isin(badn)], ed[~ed.source_id.isin(badn) & ~ed.target_id.isin(badn)]]))
o = pd.concat(parts, ignore_index=True)
if 'id' in o.columns:
    o['id'] = np.arange(len(o))
o.to_csv(a.sub, index=False)
print(f'track gate: removed {tot_d} of {tot_n} gated nodes', flush=True)
