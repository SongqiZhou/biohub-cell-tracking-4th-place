"""Graph + CNN features for the per-node division classifiers (53 columns, NAMES).

A dividing cell owns two strong, uncontested links to two daughters on opposite sides of it, the local cell count goes up,
and both daughters keep going. The features describe a node's best outgoing links, where its candidate daughters are and
whether other nodes compete for them, motion, local density, the two-frames-later geometry and the division CNN score.
"""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import topk_edges

S = np.array([1.625, 0.40625, 0.40625])

NAMES = [
    'p1', 'p2', 'p3', 'p2_over_p1', 'p1_minus_p2', 'p3_over_p2', 'n_out', 'q1', 'n_in', 'p2_rank',
    'd1', 'd2', 'dsib', 'cos_sib', 'dsib_over_d', 'sym', 'd1_rel', 'd2_rel',
    'c1_q1', 'c2_q1', 'c1_is_mine', 'c2_is_mine', 'c1_nin', 'c2_nin', 'p2_margin_c2', 'p1_margin_c1',
    'c1_out', 'c2_out', 'c1_nout', 'c2_nout', 'grand_div',
    'mot_in', 'cos_turn1', 'cos_turn2', 'mot_in_rel',
    'dens15', 'dens8', 'dnext15', 'dnext8', 'ddens15', 'ddens8',
    'cnn', 'cnn_rank', 'trel', 'p2_x_cnn',
    'dsib2', 'dsib_growth', 'cos_sib2', 'gc_min_p',
    'dsib13', 'cos13', 'dsib23', 'cos23',
]


def _rank_per_frame(vals, groups):
    out = np.zeros(len(vals))
    for ids in groups:
        out[ids] = np.argsort(np.argsort(vals[ids])) / max(len(ids) - 1, 1)
    return out


def build(cand_path, div_dir):
    """cand_path: <movie>.candidates.npz (coords (N, 4) t,z,y,x voxels; edges (M, 4) i, j, p, dist) -> (X (N, 53), coords)."""
    d = np.load(cand_path)
    return build_arrays(d['coords'], d['edges'], Path(cand_path).name.replace('.candidates.npz', ''), div_dir)


def build_arrays(coords, edges, name, div_dir):
    coords = coords.astype(float)
    N = len(coords)
    if len(edges):
        src = edges[:, 0].astype(np.int64); tgt = edges[:, 1].astype(np.int64); pr = edges[:, 2]
    else:
        src = tgt = np.zeros(0, np.int64); pr = np.zeros(0)
    OP, OC, ndeg = topk_edges(src, pr, tgt, N)           # outgoing: candidate children
    IP, IPar, ideg = topk_edges(tgt, pr, src, N)         # incoming: candidate parents

    cnn = np.zeros(N)
    p = Path(div_dir) / f'{name}.divscore.npz'
    if p.exists():
        dd = np.load(p)
        sm = {k: float(v) for k, v in zip(map(tuple, dd['keys'].astype(np.int64)), dd['score'])}
        cnn = np.array([sm.get(tuple(r), 0.0) for r in np.rint(coords).astype(np.int64)])

    ti = np.rint(coords[:, 0]).astype(np.int64)
    frames = {int(t): np.flatnonzero(ti == t) for t in np.unique(ti)}
    xyz = coords[:, 1:] * S
    trees = {t: cKDTree(xyz[v]) for t, v in frames.items()}
    Tmax = max(frames) or 1
    groups = list(frames.values())
    cnnr = _rank_per_frame(cnn, groups)
    p2r = _rank_per_frame(OP[:, 1], groups)

    dens15 = np.zeros(N); dens8 = np.zeros(N); dn15 = np.zeros(N); dn8 = np.zeros(N); medmot = np.ones(N)
    for t, ids in frames.items():
        tr = trees[t]; q = xyz[ids]
        dens15[ids] = tr.query_ball_point(q, 15.0, return_length=True) - 1
        dens8[ids] = tr.query_ball_point(q, 8.0, return_length=True) - 1
        if t + 1 in trees:
            t2 = trees[t + 1]
            dn15[ids] = t2.query_ball_point(q, 15.0, return_length=True)
            dn8[ids] = t2.query_ball_point(q, 8.0, return_length=True)
        else:
            dn15[ids] = dens15[ids] + 1; dn8[ids] = dens8[ids] + 1

    has_par = IPar[:, 0] >= 0
    vin = np.zeros((N, 3)); vin[has_par] = xyz[has_par] - xyz[IPar[has_par, 0]]
    mot = np.linalg.norm(vin, axis=1)
    for t, ids in frames.items():
        m = mot[ids][has_par[ids]]
        medmot[ids] = float(np.median(m)) if len(m) else 1.0
    medmot = np.where(medmot > 1e-6, medmot, 1.0)

    c1, c2 = OC[:, 0], OC[:, 1]
    h1, h2 = c1 >= 0, c2 >= 0
    v1 = np.zeros((N, 3)); v1[h1] = xyz[c1[h1]] - xyz[h1.nonzero()[0]]
    v2 = np.zeros((N, 3)); v2[h2] = xyz[c2[h2]] - xyz[h2.nonzero()[0]]
    d1 = np.linalg.norm(v1, axis=1); d2 = np.linalg.norm(v2, axis=1)
    dsib = np.zeros(N); both = h1 & h2
    dsib[both] = np.linalg.norm(xyz[c1[both]] - xyz[c2[both]], axis=1)

    def _cos(a, b, na, nb):
        den = na * nb; out = np.zeros(len(a)); m = den > 1e-6
        out[m] = np.einsum('ij,ij->i', a[m], b[m]) / den[m]
        return out
    cos_sib = _cos(v1, v2, d1, d2)
    ct1 = _cos(vin, v1, mot, d1); ct2 = _cos(vin, v2, mot, d2)

    def side(c, mine_p):
        h = c >= 0; idx = np.where(h, c, 0)
        q = np.where(h, IP[idx, 0], 0.0)
        mine = (h & (IPar[idx, 0] == np.arange(N))).astype(float)
        alt = np.where(IPar[idx, 0] == np.arange(N), IP[idx, 1], IP[idx, 0])      # best incoming link that is not mine
        return (q, mine, np.where(h, ideg[idx], 0), mine_p - np.where(h, alt, 0.0),
                np.where(h, OP[idx, 0], 0.0), np.where(h, ndeg[idx], 0), (h & (OP[idx, 1] > 0.44)).astype(float))
    c1q, c1m, c1ni, m1, c1o, c1no, g1 = side(c1, OP[:, 0])
    c2q, c2m, c2ni, m2, c2o, c2no, g2 = side(c2, OP[:, 1])

    gc1 = np.where(h1, OC[np.where(h1, c1, 0), 0], -1)       # the two branches one frame further on
    gc2 = np.where(h2, OC[np.where(h2, c2, 0), 0], -1)
    hg = (gc1 >= 0) & (gc2 >= 0)
    dsib2 = np.zeros(N)
    dsib2[hg] = np.linalg.norm(xyz[gc1[hg]] - xyz[gc2[hg]], axis=1)
    w1 = np.zeros((N, 3)); w1[hg] = xyz[gc1[hg]] - xyz[np.flatnonzero(hg)]
    w2 = np.zeros((N, 3)); w2[hg] = xyz[gc2[hg]] - xyz[np.flatnonzero(hg)]
    cos_sib2 = _cos(w1, w2, np.linalg.norm(w1, axis=1), np.linalg.norm(w2, axis=1))
    gc_min_p = np.minimum(np.where(h1, OP[np.where(h1, c1, 0), 0], 0.0), np.where(h2, OP[np.where(h2, c2, 0), 0], 0.0))

    c3 = OC[:, 2]; h3 = c3 >= 0                             # the true pair may involve the third candidate
    v3 = np.zeros((N, 3)); v3[h3] = xyz[c3[h3]] - xyz[np.flatnonzero(h3)]
    d3 = np.linalg.norm(v3, axis=1)
    dsib13 = np.zeros(N); dsib23 = np.zeros(N)
    m13 = h1 & h3; m23 = h2 & h3
    dsib13[m13] = np.linalg.norm(xyz[c1[m13]] - xyz[c3[m13]], axis=1)
    dsib23[m23] = np.linalg.norm(xyz[c2[m23]] - xyz[c3[m23]], axis=1)
    cos13 = _cos(v1, v3, d1, d3); cos23 = _cos(v2, v3, d2, d3)

    def sdiv(a, b):
        return a / np.maximum(b, 1e-6)
    X = np.stack([
        OP[:, 0], OP[:, 1], OP[:, 2], sdiv(OP[:, 1], OP[:, 0]), OP[:, 0] - OP[:, 1],
        sdiv(OP[:, 2], OP[:, 1]), ndeg, IP[:, 0], ideg, p2r,
        d1, d2, dsib, cos_sib, sdiv(dsib, d1 + d2), sdiv(np.abs(d1 - d2), d1 + d2),
        sdiv(d1, medmot), sdiv(d2, medmot),
        c1q, c2q, c1m, c2m, c1ni, c2ni, m2, m1,
        c1o, c2o, c1no, c2no, g1 + g2,
        mot, ct1, ct2, sdiv(mot, medmot),
        dens15, dens8, dn15, dn8, dn15 - dens15, dn8 - dens8,
        cnn, cnnr, ti / max(Tmax, 1), OP[:, 1] * cnn,
        dsib2, sdiv(dsib2, dsib), cos_sib2, gc_min_p,
        dsib13, cos13, dsib23, cos23,
    ], 1).astype(np.float32)
    assert X.shape[1] == len(NAMES), (X.shape, len(NAMES))
    return X, coords
