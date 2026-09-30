"""Rule-based DoG nucleus tracker used for per-movie density estimation.

A difference-of-Gaussians blob detector runs on every frame, the detections are linked frame to frame, and the
per-movie nucleus spacing is derived from the node count. The pipeline uses it to choose the detection threshold
of each movie (dog_route.py) and as a detector-independent support source for the track gate (dog_nodes.py, track_gate.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

@dataclass
class ImageVolume:
    path: str
    shape: tuple  # (T, Z, Y, X)
    dtype: np.dtype
    chunk: tuple

    @property
    def n_t(self) -> int:
        return int(self.shape[0])

    def frame(self, t: int) -> np.ndarray:
        """Return the (Z, Y, X) volume for timepoint t."""
        return _read_chunk(self.path, t, self.shape, self.dtype)

def open_image(zarr_path: str) -> ImageVolume:
    with open(os.path.join(zarr_path, "0", "zarr.json")) as f:
        meta = json.load(f)
    shape = tuple(int(s) for s in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    chunk = None
    # chunk grid configuration (zarr v3)
    cg = meta.get("chunk_grid", {})
    conf = cg.get("configuration", {})
    if "chunk_shape" in conf:
        chunk = tuple(int(s) for s in conf["chunk_shape"])
    return ImageVolume(path=zarr_path, shape=shape, dtype=dtype, chunk=chunk)

_BLOSC2 = None

def _blosc2():
    global _BLOSC2
    if _BLOSC2 is None:
        import blosc2
        _BLOSC2 = blosc2
    return _BLOSC2

def _read_chunk(zarr_path: str, t: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Read and decode one timepoint chunk -> (Z, Y, X)."""
    frame_shape = shape[1:]
    chunk_path = os.path.join(zarr_path, "0", "c", str(t), "0", "0", "0")
    with open(chunk_path, "rb") as f:
        raw = f.read()
    # Decode: data chunks are blosc2 frames of the raw (Z,Y,X) array.
    try:
        dec = _blosc2().decompress(raw)
        arr = np.frombuffer(dec, dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            return arr.reshape(frame_shape).copy()
    except Exception:
        pass
    # Fallback: let zarr handle the full array (slower).
    import zarr
    z = zarr.open(os.path.join(zarr_path, "0"), mode="r")
    return np.asarray(z[t])

@dataclass
class TrackGraph:
    """A tracking graph: node coords + directed edges."""
    node_t: np.ndarray   # (N,) int
    node_z: np.ndarray   # (N,)
    node_y: np.ndarray
    node_x: np.ndarray
    node_ids: np.ndarray  # (N,) original ids
    edges: np.ndarray     # (E, 2) source_id, target_id (original ids)
    meta: dict

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def coords_by_id(self) -> dict:
        out = {}
        for i, nid in enumerate(self.node_ids):
            out[int(nid)] = (int(self.node_t[i]), float(self.node_z[i]),
                             float(self.node_y[i]), float(self.node_x[i]))
        return out

def detect_blobs(vol: np.ndarray,
                 xy_downsample: int = 4,
                 dog_small_um: float = 2.0,
                 dog_large_um: float = 6.0,
                 min_distance_um: float = 3.0,
                 rel_threshold: float = 0.04,
                 abs_percentile: float = 50.0,
                 max_peaks: int | None = 30000,
                 dog_scales: list | None = None) -> np.ndarray:
    """Difference-of-Gaussians blob detector with local-maxima picking.

    Designed to recover dim nuclei that a global percentile threshold misses.
    If dog_scales (list of [small_um, large_um] pairs) is given, use a scale-space
    max over those DoG responses (multi-scale, catches varying cell sizes).
    Returns (N, 3) centroids in ORIGINAL voxel coords (z, y, x).
    """
    vf = vol.astype(np.float32)
    ds = vf[:, ::xy_downsample, ::xy_downsample]

    eff = np.array([SCALE[0], SCALE[1] * xy_downsample, SCALE[2] * xy_downsample])

    # robust per-frame normalization (reveals dim cells)
    lo, hi = np.percentile(ds, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((ds - lo) / (hi - lo), 0, None)

    if dog_scales:
        dog = None
        for (s_um, l_um) in dog_scales:
            resp = (gaussian_filter(norm, sigma=s_um / eff)
                    - gaussian_filter(norm, sigma=l_um / eff))
            dog = resp if dog is None else np.maximum(dog, resp)
    else:
        s_small = dog_small_um / eff
        s_large = dog_large_um / eff
        g1 = gaussian_filter(norm, sigma=s_small)
        g2 = gaussian_filter(norm, sigma=s_large)
        dog = g1 - g2  # bright blobs ~ positive response

    footprint = _ball_footprint(min_distance_um, eff)
    mx = maximum_filter(dog, footprint=footprint, mode="nearest")
    thr = max(rel_threshold, 0.0)
    abs_thr = np.percentile(norm, abs_percentile)
    peaks = (dog == mx) & (dog >= thr) & (norm >= abs_thr)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    vals = dog[peaks]
    order = np.argsort(vals)[::-1]
    coords = coords[order]
    if max_peaks is not None and len(coords) > max_peaks:
        coords = coords[:max_peaks]

    out = coords.astype(np.float64)
    out[:, 1] *= xy_downsample
    out[:, 2] *= xy_downsample
    return out

def _ball_footprint(radius_um: float, eff_spacing: np.ndarray) -> np.ndarray:
    """Ellipsoidal footprint covering radius_um in physical space on the eff grid."""
    rad_vox = np.maximum(1, np.round(radius_um / eff_spacing).astype(int))
    zz, yy, xx = np.ogrid[-rad_vox[0]:rad_vox[0] + 1,
                          -rad_vox[1]:rad_vox[1] + 1,
                          -rad_vox[2]:rad_vox[2] + 1]
    d = ((zz * eff_spacing[0]) ** 2 + (yy * eff_spacing[1]) ** 2 +
         (xx * eff_spacing[2]) ** 2)
    return d <= radius_um ** 2

def refine_centroids(vol: np.ndarray, coords: np.ndarray, win=(1, 3, 3)) -> np.ndarray:
    """Intensity-weighted local centre of mass refinement on original resolution."""
    if len(coords) == 0:
        return coords
    Z, Y, X = vol.shape
    out = coords.copy().astype(np.float64)
    wz, wy, wx = win
    for i, (z, y, x) in enumerate(coords):
        z, y, x = int(round(z)), int(round(y)), int(round(x))
        z0, z1 = max(0, z - wz), min(Z, z + wz + 1)
        y0, y1 = max(0, y - wy), min(Y, y + wy + 1)
        x0, x1 = max(0, x - wx), min(X, x + wx + 1)
        patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
        s = patch.sum()
        if s <= 0:
            continue
        zz = np.arange(z0, z1)[:, None, None]
        yy = np.arange(y0, y1)[None, :, None]
        xx = np.arange(x0, x1)[None, None, :]
        out[i, 0] = (patch * zz).sum() / s
        out[i, 1] = (patch * yy).sum() / s
        out[i, 2] = (patch * xx).sum() / s
    return out

def link_frames(frames: list, max_link_um: float = 10.0,
                allow_divisions: bool = False,
                division_max_um: float = 6.0) -> TrackGraph:
    """Link detections across consecutive frames.

    frames: list over t of (M_t, 3) arrays of (z, y, x) voxel coords.
    Returns a predicted TrackGraph with unique node ids.
    """
    node_ids = []
    node_t = []
    node_z = []
    node_y = []
    node_x = []
    frame_ids = []  # per frame: list of assigned global node ids
    nid = 1
    for t, coords in enumerate(frames):
        ids_t = []
        for (z, y, x) in coords:
            node_ids.append(nid)
            node_t.append(t)
            node_z.append(z)
            node_y.append(y)
            node_x.append(x)
            ids_t.append(nid)
            nid += 1
        frame_ids.append(ids_t)

    edges = []
    for t in range(len(frames) - 1):
        a = frames[t]
        b = frames[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        ap = a * SCALE
        bp = b * SCALE
        d = np.sqrt(((ap[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))
        big = max_link_um * 1000.0 + 1.0
        cost = np.where(d <= max_link_um, d, big)
        ri, ci = linear_sum_assignment(cost)
        matched_b = set()
        matched_a = set()
        for r, c in zip(ri, ci):
            if d[r, c] <= max_link_um:
                edges.append((frame_ids[t][r], frame_ids[t + 1][c]))
                matched_a.add(r)
                matched_b.add(c)

        if allow_divisions:
            # try to add a second daughter for parents, from unmatched b nodes
            unmatched_b = [j for j in range(len(b)) if j not in matched_b]
            for r in list(matched_a):
                # the already-linked daughter
                # find nearest unmatched b within division_max_um of parent
                if not unmatched_b:
                    break
                pr = ap[r]
                dd = np.sqrt(((pr[None, :] - bp[unmatched_b]) ** 2).sum(axis=1))
                j = int(np.argmin(dd))
                if dd[j] <= division_max_um:
                    bj = unmatched_b[j]
                    edges.append((frame_ids[t][r], frame_ids[t + 1][bj]))
                    unmatched_b.remove(bj)

    g = TrackGraph(
        node_t=np.array(node_t, dtype=np.int64),
        node_z=np.array(node_z, dtype=np.float64),
        node_y=np.array(node_y, dtype=np.float64),
        node_x=np.array(node_x, dtype=np.float64),
        node_ids=np.array(node_ids, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
        meta={},
    )
    return g

def close_gaps(frames: list, g: TrackGraph, max_gap: int = 1,
               gap_dist_um: float = 8.0) -> TrackGraph:
    """Insert interpolated nodes to bridge single-frame detection gaps.

    For track-ends at frame t with no successor and track-starts at frame t+2
    with no predecessor, if they are within gap_dist_um*2, insert an interpolated
    node at t+1 (midpoint) and connect end -> interp -> start. Recovers cells
    missed in one frame (all GT edges are dt=1, so this yields 2 matchable edges).
    """
    if g.n_edges == 0:
        return g
    coords = {int(nid): (int(g.node_t[i]), g.node_z[i], g.node_y[i], g.node_x[i])
              for i, nid in enumerate(g.node_ids)}
    has_out = set(int(s) for s, _ in g.edges)
    has_in = set(int(t) for _, t in g.edges)
    # candidates by frame
    ends_by_t = {}   # t -> list of node ids with no outgoing edge
    starts_by_t = {}  # t -> list of node ids with no incoming edge
    for nid, (t, z, y, x) in coords.items():
        if nid not in has_out:
            ends_by_t.setdefault(t, []).append(nid)
        if nid not in has_in:
            starts_by_t.setdefault(t, []).append(nid)

    new_nodes = []  # (t, z, y, x)
    new_edges = []
    next_id = int(g.node_ids.max()) + 1 if g.n_nodes else 1
    for gap in range(1, max_gap + 1):
        for t, ends in ends_by_t.items():
            starts = starts_by_t.get(t + gap + 1)
            if not starts:
                continue
            ec = np.array([[coords[e][1], coords[e][2], coords[e][3]] for e in ends]) * SCALE
            sc = np.array([[coords[s][1], coords[s][2], coords[s][3]] for s in starts]) * SCALE
            d = np.sqrt(((ec[:, None, :] - sc[None, :, :]) ** 2).sum(axis=2))
            thr = gap_dist_um * (gap + 1)
            big = thr * 1000 + 1
            cost = np.where(d <= thr, d, big)
            ri, ci = linear_sum_assignment(cost)
            used_s = set()
            for r, c in zip(ri, ci):
                if d[r, c] > thr or ends[r] in has_out or starts[c] in used_s:
                    continue
                e_id, s_id = ends[r], starts[c]
                te, ze, ye, xe = coords[e_id]
                ts, zs, ys, xs = coords[s_id]
                prev = e_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    zi = ze + (zs - ze) * frac
                    yi = ye + (ys - ye) * frac
                    xi = xe + (xs - xe) * frac
                    nid = next_id
                    next_id += 1
                    new_nodes.append((te + k, zi, yi, xi, nid))
                    new_edges.append((prev, nid))
                    prev = nid
                new_edges.append((prev, s_id))
                has_out.add(e_id)
                used_s.add(c)
    if not new_nodes:
        return g
    nt = np.concatenate([g.node_t, np.array([n[0] for n in new_nodes], dtype=np.int64)])
    nz = np.concatenate([g.node_z, np.array([n[1] for n in new_nodes])])
    ny = np.concatenate([g.node_y, np.array([n[2] for n in new_nodes])])
    nx = np.concatenate([g.node_x, np.array([n[3] for n in new_nodes])])
    nid = np.concatenate([g.node_ids, np.array([n[4] for n in new_nodes], dtype=np.int64)])
    edges = np.concatenate([g.edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)])
    return TrackGraph(node_t=nt, node_z=nz, node_y=ny, node_x=nx, node_ids=nid,
                      edges=edges, meta=g.meta)

def prune_isolated(g: TrackGraph) -> TrackGraph:
    """Remove nodes not referenced by any edge."""
    if g.n_edges == 0:
        return g
    used = set(int(x) for x in g.edges.reshape(-1))
    keep = np.array([i for i, nid in enumerate(g.node_ids) if int(nid) in used])
    if len(keep) == len(g.node_ids):
        return g
    return TrackGraph(
        node_t=g.node_t[keep], node_z=g.node_z[keep], node_y=g.node_y[keep],
        node_x=g.node_x[keep], node_ids=g.node_ids[keep], edges=g.edges, meta=g.meta,
    )

@dataclass
class Config:
    xy_downsample: int = 4
    dog_small_um: float = 1.5
    dog_large_um: float = 4.0
    dog_scales: list | None = None
    rel_threshold: float = 0.02
    abs_percentile: float = 50.0
    min_distance_um: float = 2.5
    max_peaks: int | None = 40000
    refine: bool = True
    max_link_um: float = 10.0
    allow_divisions: bool = False
    division_max_um: float = 6.0
    close_gaps: bool = False
    max_gap: int = 1
    gap_dist_um: float = 8.0
    prune_isolated: bool = True


# settings of the density router; the track gate lowers rel_threshold
ROUTER_CONFIG = {'dog_scales': [[1.5, 4.0], [2.2, 5.5]], 'rel_threshold': 0.045, 'min_distance_um': 3.2, 'max_peaks': 40000,
                 'max_link_um': 8.0, 'close_gaps': True, 'max_gap': 1, 'gap_dist_um': 6.0}


def track_movie(zarr_path: str, cfg: Config) -> TrackGraph:
    vol_meta = open_image(zarr_path)
    frames = []
    for t in range(vol_meta.n_t):
        vol = vol_meta.frame(t)
        coords = detect_blobs(vol, xy_downsample=cfg.xy_downsample, dog_small_um=cfg.dog_small_um, dog_large_um=cfg.dog_large_um,
                              min_distance_um=cfg.min_distance_um, rel_threshold=cfg.rel_threshold,
                              abs_percentile=cfg.abs_percentile, max_peaks=cfg.max_peaks, dog_scales=cfg.dog_scales)
        if cfg.refine and len(coords) > 0:
            coords = refine_centroids(vol, coords)
        frames.append(coords)
    g = link_frames(frames, max_link_um=cfg.max_link_um, allow_divisions=cfg.allow_divisions, division_max_um=cfg.division_max_um)
    if cfg.close_gaps:
        g = close_gaps(frames, g, max_gap=cfg.max_gap, gap_dist_um=cfg.gap_dist_um)
    if cfg.prune_isolated:
        g = prune_isolated(g)
    return g


def movie_spacing_um(zarr_path: str, n_nodes: int) -> float:
    """Mean nucleus spacing (um) implied by the average number of nodes per frame."""
    vol = open_image(zarr_path)
    T = int(vol.shape[0])
    z, y, x = vol.shape[1], vol.shape[2], vol.shape[3]
    vol_um3 = (z * SCALE[0]) * (y * SCALE[1]) * (x * SCALE[2])
    per_frame = max(1.0, n_nodes / max(1, T))
    return float((vol_um3 / per_frame) ** (1.0 / 3.0))
