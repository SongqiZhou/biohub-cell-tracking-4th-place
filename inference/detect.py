#!/usr/bin/env python
"""Nucleus detection with the 3D Net -> <out>/<movie>.nodes.npy ([t, z, y, x] full-resolution voxels) and
<movie>.scores.npy ([max cell probability, cluster size] per node).

Per frame the network predicts a cell probability and a unit vector field; voxels above --thr are advected along
the field and clustered where they converge (Cellpose-style), followed by a distance NMS. With --nms-adapt the
NMS radius is chosen per movie from the median nearest-neighbour distance of the detections.

With --refine-with, a second 3D Net (trained with pseudo-labels) is run as well. Its nodes are linked into tracks
and a track is kept only if at least --support of its nodes have a node of the first model within --refine-r um.

usage: detect.py --data-dir DIR --vec net3d_128.pt --out DIR --thr 0.97 --nms-adapt 4,7,10
                 [--refine-with net3d_128_pl.pt --refine-r 6 --support 0.5] [--movies a,b] [--shard i/n]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ZarrFrames  # noqa: E402
from net3d import load_net3d  # noqa: E402

warnings.filterwarnings("ignore", message="index_reduce")

SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], np.float32)       # full-resolution um / voxel
NORM = dict(lo_pct=50.0, hi_pct=99.5, clip_lo=-0.5, clip_hi=6.0)


def grid_scale(pool: int) -> np.ndarray:
    return (SCALE_ZYX * np.array([1, pool, pool], np.float32)).astype(np.float32)


def normalize(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol, np.float32)
    lo, hi = np.percentile(v, [NORM["lo_pct"], NORM["hi_pct"]])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(v, np.float32)
    return np.clip((v - lo) / (hi - lo), NORM["clip_lo"], NORM["clip_hi"]).astype(np.float32)


def pool_xy(vol: np.ndarray, f: int) -> np.ndarray:
    z, y, x = vol.shape
    return vol.reshape(z, y // f, f, x // f, f).mean(axis=(2, 4))


class Movie:
    """One movie in RAM (uint16); normalised XY-pooled frames are cached as float16."""

    def __init__(self, zarr_path: Path, pool: int):
        self.raw = ZarrFrames(zarr_path).read_all()          # (T, Z, Y, X)
        self.T = int(self.raw.shape[0])
        self.pool = pool
        self._cache: dict[int, np.ndarray] = {}

    def _frame(self, i: int) -> np.ndarray:
        v = self._cache.get(i)
        if v is None:
            v = normalize(pool_xy(self.raw[i].astype(np.float32), self.pool)).astype(np.float16)
            self._cache[i] = v
        return v

    def window(self, t: int, half: int = 1) -> torch.Tensor:
        ts = [min(max(t + d, 0), self.T - 1) for d in range(-half, half + 1)]
        return torch.from_numpy(np.stack([self._frame(i) for i in ts]).astype(np.float32))


def _empty():
    return dict(centre_um=np.zeros((0, 3), np.float32), size=np.zeros(0, np.int64), seed=np.zeros(0, np.float32))


def nms(d: dict, radius_um: float) -> dict:
    """Greedy distance NMS, highest cell probability first."""
    C = d["centre_um"]
    if len(C) <= 1 or radius_um <= 0:
        return d
    order = np.argsort(-d["seed"])
    keep, tree = [], cKDTree(C)
    dead = np.zeros(len(C), bool)
    for i in order:
        if dead[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(C[i], radius_um):
            if j != i:
                dead[j] = True
    keep = np.sort(np.asarray(keep, np.int64))
    return {k: v[keep] for k, v in d.items()}


@torch.no_grad()
def advect_decode(disp, seed, gs: np.ndarray, seed_thr: float, niter: int = 40, step_um: float = 0.5,
                  min_size: int = 20, nms_um: float = 4.0, sink_um: float = 1.0):
    """Move every voxel with p > seed_thr along the unit field for niter steps; clusters of >= min_size voxels that
    end in the same sink_um cell become one detection at their mean final position."""
    Z, Y, X = seed.shape
    dev = seed.device
    g = torch.tensor(gs, device=dev, dtype=torch.float32)
    p = torch.sigmoid(seed).reshape(-1)
    cand = (p > seed_thr).nonzero(as_tuple=True)[0]
    if cand.numel() == 0:
        return _empty()
    zz = cand // (Y * X)
    yy = (cand // X) % Y
    xx = cand % X
    pos = torch.stack([zz, yy, xx], 1).float() * g
    sc = p[cand]
    flow = disp / (disp.norm(dim=0, keepdim=True) + 1e-6)
    fl = flow.unsqueeze(0)
    size = torch.tensor([Z, Y, X], device=dev, dtype=torch.float32)
    for _ in range(niter):
        n = 2.0 * (pos / g) / (size - 1).clamp_min(1) - 1.0
        v = torch.nn.functional.grid_sample(fl, n.flip(-1).view(1, -1, 1, 1, 3), mode="bilinear",
                                            align_corners=True, padding_mode="border")
        pos = pos + step_um * v.view(3, -1).T
        pos = torch.max(torch.zeros_like(pos), torch.min(pos, (size - 1) * g))
    key = torch.round(pos / sink_um).to(torch.int64)
    key = (key[:, 0] * 100003 + key[:, 1]) * 100003 + key[:, 2]
    uk, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    keep = (cnt >= min_size).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return _empty()
    K = uk.numel()
    cen = torch.zeros(K, 3, device=dev, dtype=pos.dtype).index_add_(0, inv, pos) / cnt[:, None].to(pos.dtype)
    smax = torch.full((K,), float("-inf"), device=dev, dtype=sc.dtype).index_reduce_(0, inv, sc, "amax", include_self=True)
    out = dict(centre_um=cen[keep].cpu().numpy(), size=cnt[keep].cpu().numpy().astype(np.int64),
               seed=smax[keep].cpu().numpy().astype(np.float32))
    return nms(out, nms_um)


def dedup(pts: np.ndarray, r: float) -> np.ndarray:
    """Greedy merge of points within r um (member mean); r = 0 only merges exact duplicates."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    if not len(pts):
        return np.zeros((0, 3))
    tree = cKDTree(pts)
    owner = np.full(len(pts), -1)
    kp = []
    for i in range(len(pts)):
        if owner[i] >= 0:
            continue
        owner[i] = i
        kp.append(i)
        for j in tree.query_ball_point(pts[i], r):
            if owner[j] < 0:
                owner[j] = i
    return np.stack([pts[owner == i].mean(0) for i in kp])


@torch.no_grad()
def detect_movie(model, mv: Movie, device, gs, thr: float, nms_um: float, nms_alt: float | None = None, advect_steps: int = 40):
    """Returns (nodes, scores, alt_nodes, alt_scores): per frame (M, 3) um arrays and (M, 2) [seed, size]."""
    min_size = int(round(20 * 4 / mv.pool ** 2))
    U, S, UA, SA = [], [], [], []
    for t in range(mv.T):
        x = mv.window(t, 1).unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            f = model.backbone(x)
            head = model.head(f)
        flow = head["flow"][0].float()
        prob = head["prob"][0, 0].float()
        d = advect_decode(flow, prob, gs, seed_thr=thr, niter=advect_steps, min_size=min_size, nms_um=nms_um)
        if nms_alt is not None:
            da = advect_decode(flow, prob, gs, seed_thr=thr, niter=advect_steps, min_size=min_size, nms_um=nms_alt)
            UA.append(np.asarray(da["centre_um"], np.float64).reshape(-1, 3))
            SA.append(np.stack([np.asarray(da["seed"], np.float32), np.asarray(da["size"], np.float32)], 1))
        S.append(np.stack([np.asarray(d["seed"], np.float32), np.asarray(d["size"], np.float32)], 1))
        U.append(dedup(np.asarray(d["centre_um"], np.float64).reshape(-1, 3), 0.0).astype(np.float32))
    return U, S, UA, SA


def link_greedy(frames: list, gate_um: float = 7.0, div_gate_um: float = 7.0, sym_tol_um: float = 1.5):
    """Frame-to-frame Hungarian linking inside a distance gate; a second pass lets a parent adopt a second child."""
    node_id, nodes, edges, ids = 0, [], [], []
    for t, f in enumerate(frames):
        n = len(f)
        ids.append(np.arange(node_id, node_id + n))
        node_id += n
        for c in f:
            nodes.append((t, c[0], c[1], c[2]))
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
                n_child[i] += 1
                taken[j] = True
                child_d[i] = D[i, j]
        for j in np.flatnonzero(~taken):
            cand = np.flatnonzero((D[:, j] <= div_gate_um) & (n_child == 1))
            if cand.size:
                i = cand[np.argmin(D[cand, j])]
                if sym_tol_um is not None and abs(D[i, j] - child_d[i]) > sym_tol_um:
                    continue
                edges.append((ids[t][i], ids[t + 1][j]))
                n_child[i] += 1
    return np.asarray(nodes, np.float32).reshape(-1, 4), np.asarray(edges, np.int64).reshape(-1, 2)


def match_one_to_one(P: np.ndarray, Q: np.ndarray, r: float):
    """Nearest one-to-one matching of P to Q within r um -> (matched mask over P, index into Q or -1)."""
    m = np.zeros(len(P), bool)
    mi = np.full(len(P), -1)
    if not len(P) or not len(Q):
        return m, mi
    if len(P) * len(Q) <= 4_000_000:
        d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
        cost = np.where(d <= r, d, 1e6)
        ri, ci = linear_sum_assignment(cost)
        ok = cost[ri, ci] < 1e6
        m[ri[ok]] = True
        mi[ri[ok]] = ci[ok]
    else:
        dd, jj = cKDTree(Q).query(P, distance_upper_bound=r)
        m = np.isfinite(dd)
        mi[m] = jj[m]
    return m, mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--vec', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=float, default=0.97)
    ap.add_argument('--nms-adapt', default='4,7,10', help='r_dense,r_sparse,nn_cut (um)')
    ap.add_argument('--refine-with', default='')
    ap.add_argument('--refine-r', type=float, default=6.0)
    ap.add_argument('--support', type=float, default=0.5)
    ap.add_argument('--movies', default='')
    ap.add_argument('--shard', default='')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()

    dd = Path(a.data_dir)
    names = [x for x in a.movies.split(',') if x] if a.movies else sorted(p.name[:-5] for p in dd.glob('*.zarr'))
    if a.shard:
        i, n = (int(v) for v in a.shard.split('/'))
        names = names[i::n]
    dev = torch.device(f'cuda:{a.gpu}')
    torch.cuda.set_device(dev)
    model, cfg, _ = load_net3d(a.vec, dev)
    model2 = None
    if a.refine_with:
        model2, cfg2, _ = load_net3d(a.refine_with, dev)
        assert cfg2['pool_xy'] == cfg['pool_xy'], 'both models must use the same grid'
    pool = int(cfg['pool_xy'])
    gs = grid_scale(pool)
    rd, rs, cut = (float(v) for v in a.nms_adapt.split(','))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'[detect] {a.vec} thr {a.thr} pool {pool} refine {a.refine_with or "-"} | {len(names)} movies', flush=True)
    for k, name in enumerate(names):
        t0 = time.time()
        mv = Movie(dd / f'{name}.zarr', pool)
        main_n, main_s, alt_n, alt_s = detect_movie(model, mv, dev, gs, a.thr, nms_um=rd, nms_alt=rs)
        meds = [float(np.median(cKDTree(f).query(f, k=2)[0][:, 1])) for f in main_n[::5] if len(f) >= 5]
        nn_med = float(np.median(meds)) if meds else float('nan')
        if nn_med >= cut:
            nodes, scores, nms_used = alt_n, alt_s, rs
        else:
            nodes, scores, nms_used = main_n, main_s, rd
        print(f'    [nms] {name}: median NN {nn_med:.2f} um -> NMS {nms_used:g} um', flush=True)
        if model2 is not None:
            sec, _, _, _ = detect_movie(model2, mv, dev, gs, a.thr, nms_um=nms_used)
            prim = [np.asarray(u, np.float64).reshape(-1, 3).copy() for u in nodes]
            frs = [np.asarray(sec[t] if t < len(sec) else np.zeros((0, 3)), np.float64).reshape(-1, 3) for t in range(len(nodes))]
            supported, sec_scores = [], []
            for t in range(len(nodes)):
                m_, mi = match_one_to_one(frs[t], prim[t], a.refine_r)
                supported.append(m_)
                ps = np.asarray(scores[t], np.float32)
                if len(ps):
                    med = np.median(ps, 0)
                    sc_t = np.tile(med, (len(frs[t]), 1))
                    sc_t[m_] = ps[mi[m_]]
                    sec_scores.append(sc_t.astype(np.float32))
                else:
                    sec_scores.append(np.tile(np.array([1.0, 0.0], np.float32), (len(frs[t]), 1)))
            lnodes, ledges = link_greedy([f.astype(np.float32) for f in frs])
            parent = np.arange(len(lnodes))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for s_, d_ in ledges:
                ra, rb = find(int(s_)), find(int(d_))
                if ra != rb:
                    parent[ra] = rb
            comp = np.array([find(i) for i in range(len(lnodes))])
            sup = np.concatenate(supported) if supported else np.zeros(0, bool)
            _, inv = np.unique(comp, return_inverse=True)
            frac = np.bincount(inv, weights=sup.astype(float)) / np.maximum(np.bincount(inv), 1)
            keep = frac[inv] >= a.support
            off = np.cumsum([0] + [len(f) for f in frs])
            nodes = [frs[t][keep[off[t]:off[t + 1]]] for t in range(len(frs))]
            scores = [sec_scores[t][keep[off[t]:off[t + 1]]] for t in range(len(frs))]
            print(f'    [refine] {name}: kept {int(keep.sum())}/{len(lnodes)} nodes of the second model (support >= {a.support:g})', flush=True)
        rows = [np.concatenate([np.full((len(P), 1), t), np.rint(np.asarray(P, np.float64) / SCALE_ZYX[None])], 1)
                for t, P in enumerate(nodes) if len(P)]
        arr = np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, 4), np.float32)
        np.save(out / f'{name}.nodes.npy', arr)
        if sum(len(s) for s in scores) == len(arr):
            np.save(out / f'{name}.scores.npy', np.concatenate(scores).astype(np.float32) if len(scores) else np.zeros((0, 2), np.float32))
        print(f'[{k + 1}/{len(names)}] {name} nodes={len(arr)} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
