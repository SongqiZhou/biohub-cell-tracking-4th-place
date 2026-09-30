"""Shared constants and small graph utilities used by the downstream steps."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], np.float32)   # full-resolution um / voxel
CSV_HEADER = 'dataset,row_type,node_id,t,z,y,x,source_id,target_id'


def submission_rows(name, nodes, edges) -> list[str]:
    """nodes (N, 4) = [t, z, y, x] in um, edges (E, 2) node indices -> competition rows (integer full-res voxels)."""
    out = []
    vox = nodes[:, 1:] / SCALE_ZYX[None]
    for i, (t, (z, y, x)) in enumerate(zip(nodes[:, 0], vox)):
        out.append(f"{name},node,{i},{int(t)},{int(round(z))},{int(round(y))},{int(round(x))},-1,-1")
    for s, d in edges:
        out.append(f"{name},edge,-1,-1,-1,-1,-1,{int(s)},{int(d)}")
    return out


def link_greedy(frames: list[np.ndarray], gate_um: float = 7.0, div_gate_um: float = 7.0, sym_tol_um: float = 1.5):
    """Frame-to-frame Hungarian matching inside a distance gate, then a second pass in which a parent with one child may
    adopt an unmatched node as a second child (both children at a similar distance). frames[t] = (N_t, 3) um.
    Returns (nodes (N, 4) [t, z, y, x], edges (E, 2))."""
    node_id, nodes, edges = 0, [], []
    ids: list[np.ndarray] = []
    for t, c in enumerate(frames):
        n = len(c)
        ids.append(np.arange(node_id, node_id + n))
        node_id += n
        for p in c:
            nodes.append((t, p[0], p[1], p[2]))
    for t in range(len(frames) - 1):
        A, B = frames[t], frames[t + 1]
        if not len(A) or not len(B):
            continue
        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(np.where(D <= gate_um, D, 1e6))
        n_child = np.zeros(len(A), np.int64)
        taken = np.zeros(len(B), bool)
        child_d = np.full(len(A), np.nan)
        for i, j in zip(ri, ci):
            if D[i, j] <= gate_um:
                edges.append((ids[t][i], ids[t + 1][j]))
                n_child[i] += 1; taken[j] = True; child_d[i] = D[i, j]
        for j in np.flatnonzero(~taken):
            cand = np.flatnonzero((D[:, j] <= div_gate_um) & (n_child == 1))
            if cand.size:
                i = cand[np.argmin(D[cand, j])]
                if sym_tol_um is not None and abs(D[i, j] - child_d[i]) > sym_tol_um:
                    continue
                edges.append((ids[t][i], ids[t + 1][j]))
                n_child[i] += 1
    return np.asarray(nodes, np.float32).reshape(-1, 4), np.asarray(edges, np.int64).reshape(-1, 2)


def topk_edges(key, prob, other, n, k=3):
    """For every node, its k most probable incident edges grouped by `key`.
    Returns (probabilities (n, k), partners (n, k) padded with -1, degree (n,))."""
    P = np.zeros((n, k)); Q = np.full((n, k), -1, np.int64); deg = np.zeros(n, np.int32)
    if len(key) == 0:
        return P, Q, deg
    order = np.lexsort((-prob, key))
    ks, ps, os_ = key[order], prob[order], other[order]
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    counts = np.diff(np.r_[start, len(ks)])
    deg[ks[start]] = counts
    for r in range(k):
        m = counts > r
        idx = start[m] + r
        P[ks[start[m]], r] = ps[idx]; Q[ks[start[m]], r] = os_[idx]
    return P, Q, deg


def filter_short_tracks(nodes, edges, min_len: int = 3, keep_division: bool = True):
    """Drop connected components with fewer than min_len nodes (components containing a division are kept)."""
    n = len(nodes)
    if n == 0 or min_len <= 1:
        return nodes, edges, np.ones(n, bool)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    out = np.zeros(n, np.int64)
    for s_, d_ in edges:
        ra, rb = find(int(s_)), find(int(d_))
        if ra != rb:
            parent[ra] = rb
        out[int(s_)] += 1
    roots = np.array([find(i) for i in range(n)])
    keep = np.zeros(n, bool)
    for r in np.unique(roots):
        m = np.flatnonzero(roots == r)
        if len(m) >= min_len or (keep_division and (out[m] >= 2).any()):
            keep[m] = True
    remap = -np.ones(n, np.int64); remap[keep] = np.arange(int(keep.sum()))
    e = [(remap[int(s_)], remap[int(d_)]) for s_, d_ in edges if keep[int(s_)] and keep[int(d_)]]
    return nodes[keep], np.asarray(e, np.int64).reshape(-1, 2), keep


def linefit_smooth(nodes, edges, window: int = 2, weight: float = 0.8):
    """Smooth coordinates along tracks: fit a line to the +-window frames of each node's unbranched chain and move the
    node towards the fitted value at its own frame by `weight`. Edges are unchanged."""
    n = len(nodes)
    if n == 0 or window < 1 or weight <= 0:
        return nodes
    t = nodes[:, 0].astype(np.int64); pos = nodes[:, 1:].astype(np.float64)
    pred: dict[int, list] = {}; succ: dict[int, list] = {}
    for s_, d_ in edges:
        s_, d_ = int(s_), int(d_)
        if t[d_] == t[s_] + 1:
            succ.setdefault(s_, []).append(d_); pred.setdefault(d_, []).append(s_)
    new = pos.copy()
    for i in range(n):
        nb = [(0, i)]; cur = i
        for st in range(1, window + 1):
            p = pred.get(cur, [])
            if len(p) != 1:
                break
            cur = p[0]; nb.append((-st, cur))
        cur = i
        for st in range(1, window + 1):
            q = succ.get(cur, [])
            if len(q) != 1:
                break
            cur = q[0]; nb.append((st, cur))
        if len(nb) < 3:
            continue
        dts = np.array([a for a, _ in nb], np.float64); co = np.stack([pos[j] for _, j in nb])
        A = np.stack([dts, np.ones_like(dts)], 1)
        coef, *_ = np.linalg.lstsq(A, co, rcond=None)
        new[i] = (1 - weight) * pos[i] + weight * coef[1]
    out = nodes.copy(); out[:, 1:] = new.astype(nodes.dtype)
    return out


def stabilized_linefit(nodes, edges, conf, window: int = 3, weight: float = 0.8, p_min: float = 0.9, min_edges: int = 8):
    """Motion-compensated line-fit smoothing. The per-frame tissue translation is the median displacement of confident
    (conf > p_min) non-division edges (at least min_edges per frame); it is removed before linefit_smooth and added back.
    Returns (smoothed nodes, per-frame translation (T, 3))."""
    n = len(nodes)
    if n == 0 or len(edges) == 0:
        return linefit_smooth(nodes, edges, window, weight), np.zeros((int(nodes[:, 0].max()) + 1 if n else 1, 3))
    t = nodes[:, 0].astype(np.int64); pos = nodes[:, 1:].astype(np.float64)
    e = np.asarray(edges, np.int64).reshape(-1, 2); conf = np.asarray(conf, np.float64).reshape(-1)
    oc = np.bincount(e[:, 0], minlength=n)
    good = (conf > p_min) & (oc[e[:, 0]] == 1) & (t[e[:, 1]] == t[e[:, 0]] + 1)
    delta = pos[e[:, 1]] - pos[e[:, 0]]
    velocity = np.zeros((int(t.max()) + 1, 3), np.float64)
    for tt in np.unique(t):
        ix = good & (t[e[:, 1]] == tt)
        if ix.sum() >= min_edges:
            velocity[tt] = np.median(delta[ix], axis=0)
    shift = velocity.cumsum(0)
    centered = nodes.astype(np.float64).copy(); centered[:, 1:] -= shift[t]
    smooth = linefit_smooth(centered, e, window, weight)
    smooth[:, 1:] += shift[t]
    return smooth.astype(nodes.dtype), velocity
