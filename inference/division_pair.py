#!/usr/bin/env python
"""Mother-daughter pair model for the division prior.

For every node at t < T-1, candidate daughter pairs are taken from the nodes at t+1 (within 13.5 um of the mother, sisters
4.5-17 um apart, their midpoint within 7 um of the mother, on opposite sides; best 8 pairs by geometry). Each (mother, pair)
triplet is described by
  * 174 biology features from the raw images: brightness, peak, texture, compactness, size, anisotropy and background of
    the mother over its last 7 frames and of both daughters over their first 3 frames (followed by nearest nodes), mother
    brightening / slowing before the split, the split signature at the mother's position, sister separation and its growth,
    daughters moving apart along the division axis, local density;
  * the 53 graph features of the mother (division_features.py);
  * 12 embedding features (cosine similarity of the mother, its parent and its candidate children);
  * 22 link features of the two mother-daughter links (learned probability, rank, transformer probability, competing
    incoming links, whether the daughters continue, embedding similarities).
LightGBM and CatBoost are averaged; a node's score is its best pair, and the final prior is MAX(pair score, CatBoost node
prior). Writes <out>/<movie>.divscore.npz (keys (t, z, y, x), score).
usage: division_pair.py --nodes DIR --edges DIR --harm DIR --emb DIR --div-dir DIR --cat DIR --test ZARR_DIR --models models/division_pair --out DIR
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import zarr
from scipy.spatial import cKDTree

from common import topk_edges
from division_features import NAMES as GRAPH_NAMES, build_arrays

ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--edges', required=True); ap.add_argument('--harm', required=True)
ap.add_argument('--emb', required=True); ap.add_argument('--div-dir', required=True); ap.add_argument('--cat', required=True); ap.add_argument('--test', required=True)
ap.add_argument('--models', required=True); ap.add_argument('--out', required=True); ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--pmin', type=float, default=0.02)
A = ap.parse_args()
SC = np.array([1.625, .40625, .40625], np.float32)


def _ball(r):
    rz, ry, rx = [int(np.ceil(r / s)) for s in SC]; kz, ky, kx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return kz.ravel(), ky.ravel(), kx.ravel(), ((kz * SC[0]) ** 2 + (ky * SC[1]) ** 2 + (kx * SC[2]) ** 2).ravel()


KZ, KY, KX, D2 = _ball(3.0); IN = D2 <= 1.5 ** 2; SH = (D2 > 1.5 ** 2) & (D2 <= 3.0 ** 2); B3 = D2 <= 3.0 ** 2; OFFS = np.stack([KZ * SC[0], KY * SC[1], KX * SC[2]], 1)
SHELLS = [((D2 > a ** 2) & (D2 <= b ** 2)) for a, b in ((0, 1), (1, 2), (2, 3))]
LOCAL = ['b', 'peak', 'cv', 'compact', 'size', 'aniso31', 'aniso21', 'bg']
GATE = dict(dmax=13.5, sis_lo=4.5, sis_hi=17.0, mid=7.0, cos=0.6); TOPK = 8
N_TRIPLET = 174


def local_batch(vol, p50, den, um):
    """8 descriptors of the image around each position: mean and peak brightness, texture, compactness (inner ball vs
    shell), radial half-width, two anisotropy ratios of the intensity-weighted covariance, shell brightness."""
    Z, Y, X = vol.shape; vox = np.rint(um / SC).astype(int)
    zi = np.clip(vox[:, 0:1] + KZ[None], 0, Z - 1); yi = np.clip(vox[:, 1:2] + KY[None], 0, Y - 1); xi = np.clip(vox[:, 2:3] + KX[None], 0, X - 1)
    v = vol[zi, yi, xi].astype(np.float64); vi = v[:, IN]; vs = v[:, SH]; mi = vi.mean(1); ms = vs.mean(1); c0 = np.maximum(mi - p50, 1.0)
    w = np.maximum(v[:, B3] - p50, 0); w = w / np.maximum(w.sum(1, keepdims=True), 1e-6); O = OFFS[B3]; mu = w @ O; d = O[None, :, :] - mu[:, None, :]
    ev = np.sort(np.linalg.eigvalsh(np.einsum('nk,nki,nkj->nij', w, d, d)), axis=1)[:, ::-1]
    prof = np.stack([np.maximum(v[:, mk] - p50, 0).mean(1) for mk in SHELLS], 1); half = prof[:, :1] / 2; below = prof < half
    size = np.where(below.any(1), below.argmax(1) + 0.5, 3.0)
    return np.stack([(mi - p50) / den, (vi.max(1) - p50) / den, vi.std(1) / c0, (mi - p50) / np.maximum(ms - p50, 1.0), size,
                     ev[:, 0] / np.maximum(ev[:, 2], 1e-6), ev[:, 0] / np.maximum(ev[:, 1], 1e-6), (ms - p50) / den], 1).astype(np.float32)


def tables(zarr_path, t, um):
    """descriptors of every node in its own frame (tab) and at its position in the next frame (tabn, NaN in the last frame)"""
    arr = zarr.open(zarr_path, mode='r')['0']; N = len(t); T = int(arr.shape[0]); tab = np.zeros((N, 8), np.float32); tabn = np.full((N, 8), np.nan, np.float32)
    for tt in range(T):
        vol = np.asarray(arr[tt]).astype(np.float32); p50, p995 = np.percentile(vol[::2, ::2, ::2], [50, 99.5]); den = max(1.0, p995 - p50)
        ks = np.where(t == tt)[0]
        if len(ks):
            tab[ks] = local_batch(vol, p50, den, um[ks])
        kp = np.where(t == tt - 1)[0]
        if len(kp):
            tabn[kp] = local_batch(vol, p50, den, um[kp])
    return tab, tabn


def triplet_features(H, hq, hist_pos, D, dpos, pm, l0, l1, dens10, dens20):
    f = {}
    for j, nm in enumerate(LOCAL):                                   # mother history (t, t-1, ..., t-6)
        for k in range(7):
            f[f'm_{nm}_t-{k}'] = H[:, k, j]
        xs = np.arange(7); Y = H[:, ::-1, j]; xm = xs.mean()
        f[f'm_{nm}_slope6'] = ((xs - xm)[None] * (Y - Y.mean(1, keepdims=True))).sum(1) / ((xs - xm) ** 2).sum()
        f[f'm_{nm}_max6'] = H[:, :, j].max(1); f[f'm_{nm}_t0_minus_mean'] = H[:, 0, j] - H[:, 1:, j].mean(1); f[f'm_{nm}_t1_minus_t3'] = H[:, 1, j] - H[:, 3, j]
    sp = np.linalg.norm(np.diff(hist_pos, axis=1), axis=2)
    f['m_speed_last2'] = sp[:, :2].mean(1); f['m_speed_prev4'] = sp[:, 2:].mean(1); f['m_slowdown'] = sp[:, 2:].mean(1) - sp[:, :2].mean(1); f['m_track_quality'] = (hq > 3).mean(1)
    for j, nm in enumerate(LOCAL):                                   # daughters (t+1, t+2, t+3)
        for b in range(2):
            for k in range(3):
                f[f'd{b}_{nm}_t+{k+1}'] = D[:, b, k, j]
        f[f'd_{nm}_sum_t1'] = D[:, 0, 0, j] + D[:, 1, 0, j]
        f[f'd_{nm}_asym_t1'] = np.abs(D[:, 0, 0, j] - D[:, 1, 0, j]) / np.maximum(np.abs(D[:, 0, 0, j]) + np.abs(D[:, 1, 0, j]), 1e-6)
    f['mass_ratio_t1'] = (D[:, 0, 0, 0] + D[:, 1, 0, 0]) / np.maximum(H[:, 0, 0], 1e-3); f['mass_ratio_prev'] = (D[:, 0, 0, 0] + D[:, 1, 0, 0]) / np.maximum(H[:, 1, 0], 1e-3)
    sep = np.linalg.norm(dpos[:, 0] - dpos[:, 1], axis=2)
    f['sep_t1'], f['sep_t2'], f['sep_t3'] = sep[:, 0], sep[:, 1], sep[:, 2]; f['sep_growth'] = sep[:, 2] - sep[:, 0]; f['sep_growth_1'] = sep[:, 1] - sep[:, 0]
    a1 = dpos[:, 0, 0] - pm; a2 = dpos[:, 1, 0] - pm; n1 = np.linalg.norm(a1, axis=1); n2 = np.linalg.norm(a2, axis=1)
    f['d1_dist'] = n1; f['d2_dist'] = n2; f['cos_daughters'] = (a1 * a2).sum(1) / np.maximum(n1 * n2, 1e-6); f['midpoint_off'] = np.linalg.norm(0.5 * (dpos[:, 0, 0] + dpos[:, 1, 0]) - pm, axis=1)
    axis = dpos[:, 1, 0] - dpos[:, 0, 0]; axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-6)
    v0 = dpos[:, 0, 2] - dpos[:, 0, 0]; v1 = dpos[:, 1, 2] - dpos[:, 1, 0]; f['d0_along_axis'] = -(v0 * axis).sum(1); f['d1_along_axis'] = (v1 * axis).sum(1)
    f['split_drop_b'] = l0[:, 0] - l1[:, 0]; f['split_drop_peak'] = l0[:, 1] - l1[:, 1]; f['split_ratio_b'] = l1[:, 0] / np.maximum(l0[:, 0], 1e-3)
    f['dens10'] = dens10; f['dens20'] = dens20
    assert len(f) == N_TRIPLET
    return np.stack(list(f.values()), 1).astype(np.float32)


def candidate_triplets(t, um, tab, tabn):
    """-> (mothers, rows (n, 3) = [mother row, daughter a, daughter b], features (n, 174))"""
    T = int(t.max()) + 1; frames = {tt: np.where(t == tt)[0] for tt in range(T)}; trees = {tt: (cKDTree(um[ks]) if len(ks) else None) for tt, ks in frames.items()}

    def nearest(tt, P, r):
        out = np.full(len(P), -1, int)
        if tt < 0 or tt >= T or trees[tt] is None or len(P) == 0:
            return out
        d, j = trees[tt].query(P); ok = d <= r; out[ok] = frames[tt][j[ok]]; return out
    mothers = np.where(t < T - 1)[0]
    chain_pos = np.zeros((len(mothers), 7, 3)); chain_idx = np.full((len(mothers), 7), -1, int); chain_pos[:, 0] = um[mothers]; chain_idx[:, 0] = mothers
    for k in range(1, 7):                                            # mother history by nearest node within 4 um
        prev = chain_pos[:, k - 1]; idx = np.full(len(mothers), -1, int)
        for tt in np.unique(t[mothers]):
            sel = np.where(t[mothers] == tt)[0]; idx[sel] = nearest(tt - k, prev[sel], 4.0)
        chain_idx[:, k] = idx; chain_pos[:, k] = np.where(idx[:, None] >= 0, um[np.maximum(idx, 0)], prev)
    rows = []
    for tt in np.unique(t[mothers]):
        sel = np.where(t[mothers] == tt)[0]; nx = frames.get(tt + 1)
        if nx is None or len(nx) < 2 or trees[tt + 1] is None:
            continue
        for r_, cand in zip(sel, trees[tt + 1].query_ball_point(um[mothers[sel]], GATE['dmax'])):
            if len(cand) < 2:
                continue
            cand = nx[np.array(cand, int)]; P = um[cand]; mu = um[mothers[r_]]; Av = P - mu; d = np.linalg.norm(Av, axis=1); scored = []
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    sis = np.linalg.norm(P[i] - P[j])
                    if not (GATE['sis_lo'] <= sis <= GATE['sis_hi']):
                        continue
                    mid = np.linalg.norm(0.5 * (P[i] + P[j]) - mu)
                    if mid > GATE['mid']:
                        continue
                    if float(Av[i] @ Av[j]) / max(d[i] * d[j], 1e-6) > GATE['cos']:
                        continue
                    scored.append((mid + 0.3 * abs(sis - 8.0), cand[i], cand[j]))
            scored.sort()
            for s_, v, w in scored[:TOPK]:
                rows.append((r_, v, w))
    if not rows:
        return mothers, np.zeros((0, 3), int), np.zeros((0, N_TRIPLET), np.float32)
    R = np.array(rows, int); mr, V, W = R[:, 0], R[:, 1], R[:, 2]; n = len(R)
    dpos = np.zeros((n, 2, 3, 3)); didx = np.full((n, 2, 3), -1, int); dpos[:, 0, 0] = um[V]; dpos[:, 1, 0] = um[W]; didx[:, 0, 0] = V; didx[:, 1, 0] = W; tm = t[mothers[mr]]
    for b in range(2):                                               # daughters followed by nearest node within 5 um
        for k in range(1, 3):
            idx = np.full(n, -1, int)
            for tt in np.unique(tm):
                sel = np.where(tm == tt)[0]; idx[sel] = nearest(tt + 1 + k, dpos[sel, b, k - 1], 5.0)
            didx[:, b, k] = idx; dpos[:, b, k] = np.where(idx[:, None] >= 0, um[np.maximum(idx, 0)], dpos[:, b, k - 1])
    H = np.zeros((n, 7, 8)); hq = np.zeros((n, 7))
    for k in range(7):
        ci = chain_idx[mr, k]; H[:, k] = tab[np.maximum(ci, 0)]; hq[:, k] = np.where(ci >= 0, 5.0, 0.0)
        if k > 0:
            H[ci < 0, k] = H[ci < 0, k - 1]
    hq[:, 0] = 1.0; hist_pos = chain_pos[mr]
    D = np.zeros((n, 2, 3, 8))
    for b in range(2):
        for k in range(3):
            di = didx[:, b, k]; D[:, b, k] = tab[np.maximum(di, 0)]
            if k > 0:
                D[di < 0, b, k] = D[di < 0, b, k - 1]
    swap = D[:, 1, 0, 0] > D[:, 0, 0, 0]; D[swap] = D[swap][:, ::-1]; dpos[swap] = dpos[swap][:, ::-1]       # brighter daughter first
    l0 = tab[mothers[mr]]; l1 = tabn[mothers[mr]]; pm = um[mothers[mr]]; dens10 = np.zeros(n); dens20 = np.zeros(n)
    for tt in np.unique(tm):
        sel = np.where(tm == tt)[0]
        dens10[sel] = np.array([len(x) - 1 for x in trees[tt].query_ball_point(pm[sel], 10.0)])
        dens20[sel] = np.array([len(x) - 1 for x in trees[tt].query_ball_point(pm[sel], 20.0)])
    return mothers, R, triplet_features(H, hq, hist_pos, D, dpos, pm, l0, l1, dens10, dens20)


def embedding_features(N, I, J, P, E):
    """12 per-node features from the embedding: cosines to the top-3 candidate children, between them, to the parent."""
    OP, OC, _ = topk_edges(I, P, J, N, k=3); IP, IPar, _ = topk_edges(J, P, I, N, k=1)

    def cos(a, b, ok):
        out = np.zeros(N, np.float32); a_ = np.where(ok, a, 0); b_ = np.where(ok, b, 0); out[ok] = (E[a_[ok]] * E[b_[ok]]).sum(1); return out
    idx = np.arange(N); c1, c2, c3 = OC[:, 0], OC[:, 1], OC[:, 2]; pv = IPar[:, 0]; h1, h2, h3, hp = c1 >= 0, c2 >= 0, c3 >= 0, pv >= 0
    f = [cos(idx, c1, h1), cos(idx, c2, h2), cos(idx, c3, h3), cos(c1, c2, h1 & h2), cos(c1, c3, h1 & h3), cos(c2, c3, h2 & h3)]
    mean_ic = (f[0] + f[1] + f[2]) / np.maximum(h1.astype(int) + h2.astype(int) + h3.astype(int), 1); sym = np.where(h1 & h2, f[3] - np.maximum(f[0], f[1]), 0.0)
    f += [mean_ic, sym, cos(pv, idx, hp), cos(pv, c1, hp & h1), cos(pv, c2, hp & h2)]
    dmean = np.zeros(N, np.float32); ok = h1 & h2; dmean[ok] = np.linalg.norm(0.5 * (E[c1[ok]] + E[c2[ok]]) - E[ok], axis=1); f.append(dmean)
    return np.stack(f, 1).astype(np.float32)


def link_features(N, I, J, P, HK, HP, emb, u, d1, d2):
    """22 features of the two mother-daughter links (u -> d1, u -> d2)."""
    order = np.lexsort((-P, I)); Is, Js, Ps = I[order], J[order], P[order]; keys = Is * N + Js; start = np.r_[True, Is[1:] != Is[:-1]]
    grp = np.cumsum(start) - 1; first = np.flatnonzero(start); rank = np.arange(len(Is)) - first[grp] + 1
    ko = np.argsort(keys); ks = keys[ko]; pk = Ps[ko]; rk = rank[ko]

    def look(a, b, arr, default):
        q = a * N + b; pos = np.minimum(np.searchsorted(ks, q), len(ks) - 1); hit = (ks[pos] == q) & (a >= 0) & (b >= 0); return np.where(hit, arr[pos], default)

    def lookh(a, b):
        q = a * N + b; pos = np.minimum(np.searchsorted(HK, q), len(HK) - 1); hit = (HK[pos] == q) & (a >= 0) & (b >= 0); return np.where(hit, HP[pos], 0.0)
    IP, IPar, ideg = topk_edges(J, P, I, N, k=2); OP, OPar, odeg = topk_edges(I, P, J, N, k=1)
    f1 = (d1 >= 0).astype(float); f2 = (d2 >= 0).astype(float); p1 = look(u, d1, pk, 0.0); p2 = look(u, d2, pk, 0.0); r1 = look(u, d1, rk, 9.0); r2 = look(u, d2, rk, 9.0)

    def in_other(d):
        dd = np.maximum(d, 0); top1 = IP[dd, 0]; par1 = IPar[dd, 0]; top2 = IP[dd, 1]; return np.where(d >= 0, np.where(par1 == u, top2, top1), 0.0)
    io1 = in_other(d1); io2 = in_other(d2); ob1 = np.where(d1 >= 0, OP[np.maximum(d1, 0), 0], 0.0); ob2 = np.where(d2 >= 0, OP[np.maximum(d2, 0), 0], 0.0)
    no1 = np.where(d1 >= 0, odeg[np.maximum(d1, 0)], 0); no2 = np.where(d2 >= 0, odeg[np.maximum(d2, 0)], 0)

    def cos(a, b, ok):
        return np.where(ok, (emb[np.maximum(a, 0)] * emb[np.maximum(b, 0)]).sum(1), 0.0)
    return np.stack([p1, p2, np.minimum(p1, p2), p1 + p2, r1, r2, np.maximum(r1, r2), lookh(u, d1), lookh(u, d2), io1, io2, p1 / np.maximum(io1, 1e-3), p2 / np.maximum(io2, 1e-3),
                     ob1, ob2, no1, no2, cos(u, d1, d1 >= 0), cos(u, d2, d2 >= 0), cos(d1, d2, (d1 >= 0) & (d2 >= 0)), f1, f2], 1).astype(np.float32)


def graph_features(name, t, vox, I, J, P):
    """the 53 division features on the learned links with p >= pmin"""
    coords = np.concatenate([t[:, None], np.rint(vox)], 1).astype(np.int32); keep = P >= A.pmin; i, j, p = I[keep], J[keep], P[keep]
    edges = np.stack([i, j, p, np.linalg.norm(vox[j] - vox[i], axis=1)], 1).astype(np.float64)
    X, rc = build_arrays(coords, edges, name, A.div_dir)
    assert np.array_equal(rc, coords) and X.shape == (len(coords), len(GRAPH_NAMES)), name
    return np.nan_to_num(X.astype(np.float32)), coords


_M = {}


def models():
    if not _M:
        import lightgbm as lgb
        from catboost import CatBoostClassifier
        cat = CatBoostClassifier(); cat.load_model(os.path.join(A.models, 'cat.cbm'))
        _M['m'] = (lgb.Booster(model_file=os.path.join(A.models, 'lgb.txt')), cat)
    return _M['m']


def one(f):
    name = os.path.basename(f)[:-4]; z = np.load(f); t = z['t'].astype(int); um = z['um'].astype(np.float64); vox = z['vox']; N = len(t)
    e = np.load(os.path.join(A.edges, name + '.npz')); I, J, P = e['I'].astype(np.int64), e['J'].astype(np.int64), e['p'].astype(np.float64)
    h = np.load(os.path.join(A.harm, name + '.npz')); HK = h['I'].astype(np.int64) * N + h['J'].astype(np.int64); ho = np.argsort(HK); HK = HK[ho]; HP = h['p'].astype(np.float64)[ho]
    E = np.load(os.path.join(A.emb, name + '.npy')).astype(np.float64); assert len(E) == N, (name, len(E), N)
    XP, coords = graph_features(name, t, vox, I, J, P); XE = embedding_features(N, I, J, P, E.astype(np.float32))
    tab, tabn = tables(os.path.join(A.test, name + '.zarr'), t, um); mothers, R, X = candidate_triplets(t, um, tab.astype(np.float64), tabn.astype(np.float64))
    s = np.zeros(N, np.float32)
    if len(R):
        u = mothers[R[:, 0]]; bst, cat = models()
        F = np.concatenate([X, XP[u], XE[u], link_features(N, I, J, P, HK, HP, E, u, R[:, 1], R[:, 2])], 1); F = np.nan_to_num(F.astype(np.float64), nan=-1.0)
        p = 0.5 * (bst.predict(F, num_threads=2) + cat.predict_proba(F, thread_count=2)[:, 1])
        np.maximum.at(s, u, p.astype(np.float32))
    c = np.load(os.path.join(A.cat, name + '.divscore.npz')); assert np.array_equal(c['keys'], coords), name
    np.savez_compressed(os.path.join(A.out, name + '.divscore.npz'), keys=coords, score=np.maximum(s, c['score'].astype(np.float32)))
    return name, len(R)


if __name__ == '__main__':
    os.makedirs(A.out, exist_ok=True); files = sorted(glob.glob(os.path.join(A.nodes, '*.npz')))
    with ProcessPoolExecutor(A.jobs) as ex:
        for name, npairs in ex.map(one, files):
            print(f'  division pairs {name}: {npairs} candidate triplets', flush=True)
