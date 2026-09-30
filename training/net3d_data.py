"""Training data for the 3D Net: frames, labels, sparse-label targets.

* Frames are read per time point, XY mean-pooled by `pool` (2 for the 128 models, 4 for the 64 model) and normalised
  (p50 -> 0, p99.5 -> 1, clipped to [-0.5, 6]). The network input is the window (t-1, t, t+1), clamped at the ends.
* Labels are the annotated nodes (.geff) of a movie, optionally merged with pseudo-labels (<movie>.npz from
  external_data/make_pseudo_labels.py); a pseudo-label within 4 um of an annotated node is dropped.
* Targets (on the middle frame, grid resolution) have three tiers:
    region     voxels owned by a label: nearest label within 5.5 um, among the brightest 5% of the frame, plus a
               2 um core that is always kept. Vector target = unit vector to the owning label, probability target 1.
    background voxels darker than 0.4 x the local maximum within 3.5 um (and not in a region). Probability target 0.
    ignored    everything else, i.e. mostly unannotated cells. Probability target 0 with a very small weight.
  Each tier's weight is divided by its voxel count, so w_pos : w_bg : w_ignore are the importances of whole tiers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], np.float32)     # full-resolution um / voxel
MAX_LABELS = 320                                                   # labels per frame; more are subsampled at random


def grid_scale(pool: int) -> np.ndarray:
    return (SCALE_ZYX * np.array([1, pool, pool], np.float32)).astype(np.float32)


def normalize(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol, np.float32)
    lo, hi = np.percentile(v, [50.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(v, np.float32)
    return np.clip((v - lo) / (hi - lo), -0.5, 6.0).astype(np.float32)


def pool_xy(vol: np.ndarray, f: int) -> np.ndarray:
    z, y, x = vol.shape
    return vol.reshape(z, y // f, f, x // f, f).mean(axis=(2, 4))


class Frames:
    """Lazy per-movie reader with a small LRU cache of normalised, pooled frames."""

    def __init__(self, data_dir: Path, pool: int, cache: int = 16):
        self.dir, self.pool, self.cache_n = Path(data_dir), pool, cache
        self._arr: dict[str, object] = {}
        self._cache: dict = {}

    def array(self, name: str):
        if name not in self._arr:
            self._arr[name] = zarr.open_group(str(self.dir / f'{name}.zarr'), mode='r')['0']
        return self._arr[name]

    def frame(self, name: str, t: int) -> np.ndarray:
        key = (name, int(t))
        if key in self._cache:
            v = self._cache.pop(key); self._cache[key] = v
            return v
        v = normalize(pool_xy(np.asarray(self.array(name)[t], np.float32), self.pool))
        self._cache[key] = v
        while len(self._cache) > self.cache_n:
            self._cache.pop(next(iter(self._cache)))
        return v

    def window(self, name: str, t: int, half: int = 1) -> np.ndarray:
        T = int(self.array(name).shape[0])
        return np.stack([self.frame(name, min(max(t + d, 0), T - 1)) for d in range(-half, half + 1)])


# ----------------------------------------------------------------------------------------------------------- labels
def annotated_nodes(geff: Path) -> dict[int, np.ndarray]:
    """t -> (N, 3) annotated nuclei in full-resolution voxels; {} if the movie has no labels."""
    if not geff.exists():
        return {}
    p = zarr.open_group(str(geff), mode='r')['nodes/props']
    t, z, y, x = (np.asarray(p[f'{k}/values']) for k in 'tzyx')
    out: dict[int, list] = {}
    for ti, zi, yi, xi in zip(t, z, y, x):
        out.setdefault(int(ti), []).append((zi, yi, xi))
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def merge_labels(names, data_dir: Path, pseudo_dir: Path | None, dedup_um: float = 4.0) -> dict[str, dict[int, np.ndarray]]:
    """movie -> t -> (N, 4) [z, y, x, 1] full-resolution voxels: annotated nodes first, then pseudo-labels that are
    more than dedup_um away from every annotated node."""
    out, n_gt, n_ps, n_drop = {}, 0, 0, 0
    for name in names:
        gt = annotated_nodes(Path(data_dir) / f'{name}.geff')
        ps: dict[int, list] = {}
        f = Path(pseudo_dir) / f'{name}.npz' if pseudo_dir else None
        if f is not None and f.exists():
            for row in np.asarray(np.load(f)['nodes'], np.float64).reshape(-1, 4):
                ps.setdefault(int(row[0]), []).append((row[1], row[2], row[3], 1.0))
        by_t = {}
        for t in set(gt) | set(ps):
            g = np.asarray(gt.get(t, np.empty((0, 3))), np.float32).reshape(-1, 3)
            p = np.asarray(ps.get(t, []), np.float32).reshape(-1, 4)
            if len(g) and len(p):
                d, _ = cKDTree(g * SCALE_ZYX[None]).query(p[:, :3] * SCALE_ZYX[None], k=1)
                n_drop += int((d <= dedup_um).sum()); p = p[d > dedup_um]
            n_gt += len(g); n_ps += len(p)
            rows = ([np.concatenate([g, np.ones((len(g), 1), np.float32)], 1)] if len(g) else []) + ([p] if len(p) else [])
            if rows:
                by_t[int(t)] = np.concatenate(rows, 0).astype(np.float32)
        out[name] = by_t
    print(f'labels: {n_gt} annotated + {n_ps} pseudo ({n_drop} pseudo within {dedup_um} um of an annotated node dropped)', flush=True)
    return out


def frame_index(labels) -> list[tuple[str, int]]:
    return sorted((n, t) for n, by_t in labels.items() for t, c in sorted(by_t.items()) if len(c) >= 1)


def epoch_items(index, rng: np.random.Generator, frames_per_movie: int) -> list[tuple[str, int]]:
    """Per epoch, min(frames_per_movie, n) distinct labelled frames of every movie, shuffled."""
    by: dict[str, list[int]] = {}
    for n, t in index:
        by.setdefault(n, []).append(t)
    out = []
    for n, ts in by.items():
        k = min(max(1, frames_per_movie), len(ts))
        for i in rng.choice(len(ts), size=k, replace=False):
            out.append((n, int(ts[i])))
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------------------------------------- targets
@dataclass(frozen=True)
class TargetConfig:
    r_max_um: float = 5.5
    core_um: float = 2.0
    fg_quantile: float = 0.95
    bg_alpha: float = 0.40
    bg_local_um: float = 3.5
    w_pos: float = 12.0
    w_bg: float = 1.0
    w_ignore: float = 0.05


def build_targets(img: np.ndarray, labels_full: np.ndarray, pool: int, cfg: TargetConfig = TargetConfig()):
    """img (Z, Y, X) normalised grid frame, labels_full (K, 4) [z, y, x, conf] full-res voxels ->
    owner (Z, Y, X) int32 (-1 = no region), weight (Z, Y, X) float32, centre_um (K, 3)."""
    gs = grid_scale(pool)
    Z, Y, X = img.shape
    c = np.asarray(labels_full, np.float32).reshape(-1, 4)
    K = len(c)
    centre_um = (np.stack([c[:, 0], c[:, 1] / float(pool), c[:, 2] / float(pool)], 1) * gs[None]).astype(np.float32)
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    pos = np.stack([zz * gs[0], yy * gs[1], xx * gs[2]]).astype(np.float32).reshape(3, -1).T
    if K:
        dist, idx = cKDTree(centre_um).query(pos, k=1)
    else:
        dist, idx = np.full(pos.shape[0], np.inf), np.zeros(pos.shape[0], np.int64)
    dist = dist.reshape(Z, Y, X).astype(np.float32); idx = idx.reshape(Z, Y, X).astype(np.int32)
    thr = float(np.quantile(img, cfg.fg_quantile)) if img.size else 0.0
    region = (dist <= cfg.r_max_um) & ((img >= thr) | (dist <= cfg.core_um))
    owner = np.where(region, idx, -1).astype(np.int32)
    rad = np.maximum(1, np.round(cfg.bg_local_um / gs).astype(int))
    bg = (img < cfg.bg_alpha * maximum_filter(img, size=tuple(2 * rad + 1), mode='nearest')) & ~region
    weight = np.zeros((Z, Y, X), np.float32)
    for mask, importance in ((region, cfg.w_pos), (bg, cfg.w_bg), (~region & ~bg, cfg.w_ignore)):
        n = int(mask.sum())
        if n:
            weight[mask] = importance / n
    return owner, weight, centre_um


class FrameDataset(Dataset):
    """One sample = window (t-1, t, t+1) + targets of frame t, with a random Y and X flip (Z is never flipped)."""

    def __init__(self, items, labels, frames: Frames, pool: int, seed: int, half: int = 1, train: bool = True,
                 cfg: TargetConfig = TargetConfig()):
        self.items, self.labels, self.frames, self.pool = list(items), labels, frames, pool
        self.seed, self.half, self.train, self.cfg = seed, half, train, cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, t = self.items[i]
        rng = np.random.default_rng((self.seed * 1000003 + i) & 0x7FFFFFFF)
        img = self.frames.window(name, t, self.half)
        c = np.asarray(self.labels[name][t], np.float32).reshape(-1, 4).copy()
        if self.train:                        # flip the image and the labels together, before building the targets
            _, _, Y, X = img.shape
            if rng.random() < 0.5:
                img = img[:, :, ::-1]; c[:, 1] = (Y * self.pool - 1) - c[:, 1]
            if rng.random() < 0.5:
                img = img[:, :, :, ::-1]; c[:, 2] = (X * self.pool - 1) - c[:, 2]
            img = np.ascontiguousarray(img)
        if len(c) > MAX_LABELS:
            c = c[rng.choice(len(c), MAX_LABELS, replace=False)]
        owner, weight, centre_um = build_targets(img[self.half], c, self.pool, self.cfg)
        pad = np.zeros((MAX_LABELS, 3), np.float32); pad[:len(centre_um)] = centre_um[:MAX_LABELS]
        return dict(img=torch.from_numpy(img.astype(np.float32)), owner=torch.from_numpy(owner.astype(np.int64)),
                    weight=torch.from_numpy(weight), centre_um=torch.from_numpy(pad))


def load_folds(path: Path) -> dict[str, list[str]]:
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if k.startswith('fold')}
