#!/usr/bin/env python
"""Fork verification after the ILP. For every node with two children, both daughters are traced in the raw images from 3
frames before to 5 frames after the fork (motion-compensated DoG peak search around the predicted position). From the
traces we build 117 separation / peak-quality / displacement features and 40 separation-trend features, and score them
with CatBoost verifiers (raw score -> per-model sigmoid calibration -> mean). The weaker daughter link (lower learned
probability) is removed when the score is below --threshold, unless both daughters clearly move apart along the fork axis
(median slope >= --gate um/frame). Nodes are never changed.
usage: fork_verify.py --graph-dir DIR --data-dir ZARR_DIR --models models/fork_verifier/a,models/fork_verifier/b --out-csv pred.csv"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from common import CSV_HEADER, ZarrFrames, submission_rows

SCALE = np.array([1.625, .40625, .40625], np.float32)


def peak(image, center):
    """Best DoG peak within 5 um of `center` (um), refined by a weighted centroid -> (position, 4 quality stats)."""
    center = np.asarray(center, float); rounded = np.rint(center / SCALE).astype(int)
    shape = np.array([13, 41, 41]); axes = [rounded[k] + np.arange(shape[k]) - shape[k] // 2 for k in range(3)]
    valid = [(v >= 0) & (v < image.shape[k]) for k, v in enumerate(axes)]
    axes = [v.clip(0, image.shape[k] - 1) for k, v in enumerate(axes)]
    raw = image[np.ix_(*axes)].astype(np.float32)
    mask = valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :]
    if not mask.any():
        return center, np.zeros(4, np.float32)
    raw[~mask] = np.median(raw[mask])
    dog = gaussian_filter(raw, 1.2 / SCALE) - gaussian_filter(raw, 2.4 / SCALE)
    median = float(np.median(dog[mask])); noise = max(float(np.median(np.abs(dog[mask] - median))) * 1.4826, 1e-4)
    grid = np.stack(np.meshgrid(*[(np.arange(s) - s // 2) * sc for s, sc in zip(shape, SCALE)], indexing='ij'), -1)
    offsets = grid + rounded * SCALE - center; dist2 = np.sum(offsets ** 2, axis=-1)
    inside = mask & (dist2 <= 25)
    if not inside.any():
        return center, np.zeros(4, np.float32)
    merit = np.where(inside, (dog - median) / noise - dist2 / 18., -np.inf)
    at = np.unravel_index(np.argmax(merit), merit.shape)
    sl = tuple(slice(max(0, k - 1), min(s, k + 2)) for k, s in zip(at, shape))
    weights = np.maximum(dog[sl] - median, 0) * mask[sl]
    delta = (offsets[sl] * weights[..., None]).sum((0, 1, 2)) / max(float(weights.sum()), 1e-6)
    if weights.sum() <= 0:
        delta = offsets[at]
    stats = np.array([(dog[at] - median) / noise, np.linalg.norm(delta), (raw[at] - np.median(raw[mask])) / max(float(np.std(raw[mask])), 1e-4), float(mask.mean())], np.float32)
    return center + delta, stats


def assemble(positions, quality, valid, shift, times):
    stable = positions - shift[times][:, :, None, :]
    separation = np.linalg.norm(stable[:, :, 0] - stable[:, :, 1], axis=-1)
    features = [separation, quality.sum(2).reshape(len(positions), 36), np.abs(quality[:, :, 0] - quality[:, :, 1]).reshape(len(positions), 36), valid.astype(float)]
    for left, right in [(0, 4), (1, 4), (2, 4), (3, 4), (4, 5), (4, 6), (4, 7), (4, 8)]:
        disp = np.linalg.norm(stable[:, left] - stable[:, right], axis=-1)
        features.extend([disp.sum(1)[:, None], np.abs(disp[:, 0] - disp[:, 1])[:, None], (separation[:, left] - separation[:, right])[:, None]])
    before = separation[:, :4]; after = separation[:, 4:]
    features.extend([np.min(before, axis=1)[:, None], np.median(before, axis=1)[:, None], np.median(after, axis=1)[:, None]])
    return np.concatenate(features, axis=1).astype(np.float32)


class Frames:
    def __init__(self, zarr_path):
        self.arr = ZarrFrames(zarr_path); self.T = int(self.arr.shape[0]); self.cache = {}

    def get(self, t):
        t = int(t)
        if t not in self.cache:
            self.cache[t] = np.asarray(self.arr[t])
        return self.cache[t]


def trace(frames, shift, parent_times, daughter_positions):
    T = frames.T; times = parent_times[:, None] + np.arange(-3, 6)
    valid = (times >= 0) & (times < T); times = times.clip(0, T - 1)
    pos = np.zeros((len(parent_times), 9, 2, 3), np.float32); quality = np.zeros((len(parent_times), 9, 2, 4), np.float32)
    pos[:, 4] = daughter_positions
    for tt in np.unique(parent_times):
        rows = np.flatnonzero(parent_times == tt)
        for view in [4, 3, 2, 1, 0, 5, 6, 7, 8]:                 # outwards from the fork frame
            actual = int(times[rows[0], view]); im = frames.get(actual)
            for k in rows:
                ref = 4 if view == 4 else view + 1 if view < 4 else view - 1
                center = pos[k, ref] + shift[actual] - shift[times[k, ref]]
                if not valid[k, view]:
                    pos[k, view] = center; continue
                for branch in [0, 1]:
                    pos[k, view, branch], quality[k, view, branch] = peak(im, center[branch])
    return assemble(pos, quality, valid, shift, times), pos, valid


def trend(distance, valid):
    distance = np.asarray(distance, float); valid = np.asarray(valid, bool); result = np.zeros((len(distance), 12), np.float32)
    for row, (d, mask) in enumerate(zip(distance, valid)):
        t = np.flatnonzero(mask); v = d[t]; result[row, 0] = len(t)
        if len(t) < 2:
            continue
        pairs = np.triu_indices(len(t), 1); rates = (v[pairs[1]] - v[pairs[0]]) / (t[pairs[1]] - t[pairs[0]])
        slope = np.median(rates); fit = np.median(v - slope * t) + slope * t; steps = np.diff(v) / np.diff(t); drawdown = np.maximum.accumulate(v) - v
        result[row, 1:] = [slope, np.mean(rates > 0), v[-1] - v[0], np.max(drawdown), np.sum(np.maximum(-np.diff(v), 0)), np.median(np.abs(v - fit)),
                           (v[-1] - v[0]) / (1 + v[0]), np.mean(v[1:] > v[0]), steps.min(), steps.max(), t[-1] - t[0]]
    return result


def trend_features(x):
    x = np.asarray(x, np.float32); assert x.ndim == 2 and x.shape[1] == 117
    sep = x[:, :9]; valid = x[:, 81:90] > .5
    pre = trend(sep[:, 1:4], valid[:, 1:4]); early = trend(sep[:, 4:7], valid[:, 4:7]); post = trend(sep[:, 4:9], valid[:, 4:9])
    quality = (x[:, 9:45].reshape(-1, 9, 4) - x[:, 45:81].reshape(-1, 9, 4)) / 2
    observed = valid[:, 4:9]; weak = np.where(observed, quality[:, 4:9, 0], np.inf).min(1); weak = np.where(np.isfinite(weak), weak, 0)
    out = np.column_stack([pre, early, post, post[:, 1] - pre[:, 1], post[:, 3] - pre[:, 3], weak, observed.sum(1)])
    assert out.shape == (len(x), 40) and np.isfinite(out).all()
    return out.astype(np.float32)


def slopes(pos, valid):
    """median speed (um/frame) of each daughter away from the other along the fork axis, and the smaller of the two"""
    out = np.zeros((len(pos), 3), np.float32)
    for k, (p, m) in enumerate(zip(pos, valid)):
        axis = p[4, 1] - p[4, 0]; norm = np.linalg.norm(axis)
        if norm < 1e-6:
            continue
        u = axis / norm; q = np.flatnonzero(m[4:9]) + 4
        if len(q) < 3:
            continue
        vals = []
        for branch, sign in [(0, -1), (1, 1)]:
            v = np.sum((p[:, branch] - p[4, branch]) * u, axis=1) * sign; vv = v[q]
            den = q[None, :] - q[:, None]; num = vv[None, :] - vv[:, None]; ok = np.triu(den > 0, 1)
            vals.append(float(np.median((num / np.where(ok, den, 1))[ok])))
        out[k] = [vals[0], vals[1], min(vals)]
    return out


def forks_of(edges, prob):
    """(mother, strong daughter, weak daughter) for every two-child node; strong = higher link probability"""
    succ = {}; ep = {}
    for (s, d), p in zip(edges, prob):
        succ.setdefault(int(s), []).append(int(d)); ep[int(s), int(d)] = float(p)
    T = []
    for s in sorted(s for s, ch in succ.items() if len(ch) == 2):
        a, b = sorted(succ[s], key=lambda d: ep[s, d], reverse=True); T.append([s, a, b])
    return np.array(T, np.int64).reshape(-1, 3)


def load_models(spec):
    from catboost import CatBoostClassifier
    ms = []
    for d in spec.split(','):
        m = CatBoostClassifier(); m.load_model(os.path.join(d, 'model.cbm')); cal = json.load(open(os.path.join(d, 'calibration.json')))
        ms.append((m, float(cal['coef']), float(cal['intercept'])))
    return ms


def verify(nodes, edges, prob, velocity, frames, models, threshold, gate):
    T = forks_of(edges, prob); stats = dict(forks=len(T), dropped=0, protected=0)
    if len(T) == 0:
        return edges, stats
    shift = np.cumsum(velocity, axis=0).astype(np.float32)
    x, pos, valid = trace(frames, shift, nodes[T[:, 0], 0].astype(int), nodes[T[:, 1:], 1:].astype(np.float32))
    X = np.column_stack([x, trend_features(x)])
    p = np.mean([1.0 / (1.0 + np.exp(-(coef * m.predict(X, prediction_type='RawFormulaVal') + icpt))) for m, coef, icpt in models], axis=0).astype(np.float32)
    drop = p < threshold
    if gate >= 0:
        d = slopes(pos, valid); protect = (d[:, 0] >= gate) & (d[:, 1] >= gate) & (d[:, 2] >= gate)
        stats['protected'] = int((drop & protect).sum()); drop = drop & ~protect
    n = len(nodes); keys = edges[:, 0] * n + edges[:, 1]; remove = T[:, [0, 2]][drop]
    keep = ~np.isin(keys, remove[:, 0] * n + remove[:, 1]); stats['dropped'] = int((~keep).sum())
    return edges[keep], stats


_MODELS = {}


def _movie_task(args):
    f, data_dir, spec, threshold, gate = args
    if spec not in _MODELS:
        _MODELS[spec] = load_models(spec)
    name = os.path.basename(f)[:-4]; z = np.load(f); nodes, edges = z['nodes'], z['E'].astype(np.int64)
    new_edges, st = verify(nodes, edges, z['p'], z['velocity'], Frames(os.path.join(data_dir, name + '.zarr')), _MODELS[spec], threshold, gate)
    return name, submission_rows(name, nodes, new_edges), st


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--graph-dir', required=True); ap.add_argument('--data-dir', required=True); ap.add_argument('--models', required=True)
    ap.add_argument('--out-csv', required=True); ap.add_argument('--threshold', type=float, default=0.25); ap.add_argument('--gate', type=float, default=0.5)
    ap.add_argument('--jobs', type=int, default=4); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.graph_dir, '*.npz'))); lines = [CSV_HEADER]; tot = dict(forks=0, dropped=0, protected=0)
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, rows, st in ex.map(_movie_task, [(f, a.data_dir, a.models, a.threshold, a.gate) for f in files]):
            for k in tot:
                tot[k] += st[k]
            lines += rows; print(f'  fork verifier {name}: forks {st["forks"]} dropped {st["dropped"]} protected {st["protected"]}', flush=True)
    Path(a.out_csv).write_text('\n'.join(lines) + '\n'); print(f'fork verifier: {tot} -> {a.out_csv}', flush=True)


if __name__ == '__main__':
    main()
