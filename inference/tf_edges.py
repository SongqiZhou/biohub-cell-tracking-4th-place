#!/usr/bin/env python
"""Transformer link probabilities for every candidate link. For each pair of consecutive frames the edge transformer
scores all node pairs in both directions; the logits are softmax-normalised over the sources of each target (forward)
and over the targets of each source (reverse), and combined by a weighted harmonic mean (harm). Writes
<out-root>/edges_tf_{fwd,rev,harm}/<movie>.npz (I, J, p).
usage: tf_edges.py --nodes DIR --cand DIR --test ZARR_DIR --weights models/edge_transformer/weights.pth --out-root DIR"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from edge_transformer import load_edge_transformer, position_embedding


class Movie:
    """Raw movie in RAM; frames are sampled on the transformer's grid and scaled by the stored 0.1/99.9% quantiles."""

    def __init__(self, zarr_path, ds):
        import zarr
        g = zarr.open_group(str(zarr_path), mode='r')
        self.raw = np.asarray(g['0'])
        q = g.attrs['image_statistics']['quantiles']
        self.q_low, self.q_high = float(q['0.001']), float(q['0.999'])
        self.T = int(self.raw.shape[0])
        self.ds = tuple(int(d) for d in ds)
        self.grid_shape = tuple(-(-int(s) // d) for s, d in zip(self.raw.shape[1:], self.ds))

    def frames(self, ts):
        dz, dy, dx = self.ds
        raw = self.raw[list(ts), ::dz, ::dy, ::dx].astype(np.float32)
        img = torch.from_numpy((raw - self.q_low) / (self.q_high - self.q_low + 1e-6)).clamp(0.0)
        if tuple(img.shape[1:]) != self.grid_shape:
            img = F.interpolate(img[:, None], size=list(self.grid_shape), mode='trilinear', align_corners=False)[:, 0]
        return img


@torch.no_grad()
def score_movie(model, W, ds, zarr_path, z, I, J, device, harm_w=0.15):
    mv = Movie(zarr_path, ds); t = z['t']; vox = z['vox'].astype(np.float64); T = mv.T
    ds_arr = np.asarray(ds, np.float32); ds_t = torch.from_numpy(ds_arr).to(device)
    grid = np.rint(vox / ds_arr[None]).astype(np.float32)             # node coordinates on the transformer grid
    idx = {tt: np.where(t == tt)[0] for tt in range(T)}
    byt = {}
    for q in range(len(I)):
        byt.setdefault(int(t[I[q]]), []).append(q)
    out = {k: np.zeros(len(I), np.float32) for k in ('fwd', 'rev', 'harm')}
    window_shape = (W,) + mv.grid_shape
    for ws in range(0, T - W + 1):
        ts = list(range(ws, ws + W))
        feats = model.encode(mv.frames(ts).unsqueeze(0).to(device))
        for f in range(W - 1):
            ta = ts[f]
            if f > 0 and ws > 0:
                continue                                               # every frame pair is scored once
            a = idx.get(ta, np.zeros(0, int)); b = idx.get(ts[f + 1], np.zeros(0, int)); qs = byt.get(ta, [])
            if not len(a) or not len(b) or not qs:
                continue
            ca = np.concatenate([np.full((len(a), 1), f, np.float32), grid[a]], 1); cb = np.concatenate([np.full((len(b), 1), f + 1, np.float32), grid[b]], 1)
            ga = torch.from_numpy(grid[a]).to(device); gb = torch.from_numpy(grid[b]).to(device)
            ea = torch.from_numpy(position_embedding(ca, window_shape)).unsqueeze(0).to(device); eb = torch.from_numpy(position_embedding(cb, window_shape)).unsqueeze(0).to(device)
            fa = model.sample(feats[0, f], ga); fb = model.sample(feats[0, f + 1], gb)
            pa = (ga * ds_t).unsqueeze(0); pb = (gb * ds_t).unsqueeze(0)
            fwd = model.score(fa, fb, pa, pb, ea, eb)[0].float()                  # (na, nb)
            rev = model.score(fb, fa, pb, pa, eb, ea)[0].float().T
            pf = torch.softmax(fwd, 0); pr = torch.softmax(rev, 0)
            ph = 1.0 / ((1 - harm_w) / pf.clamp_min(1e-8) + harm_w / pr.clamp_min(1e-8)); ph = ph / ph.sum(0, keepdim=True).clamp_min(1e-8)
            la = {n: k for k, n in enumerate(a)}; lb = {n: k for k, n in enumerate(b)}
            ii = np.array([la[int(I[q])] for q in qs]); jj = np.array([lb[int(J[q])] for q in qs]); qs_ = np.asarray(qs)
            for k, P in (('fwd', pf), ('rev', pr), ('harm', ph)):
                out[k][qs_] = P.cpu().numpy()[ii, jj]
        del feats
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--cand', required=True); ap.add_argument('--test', required=True)
    ap.add_argument('--weights', required=True); ap.add_argument('--out-root', required=True); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--movies', default='', help='comma-separated subset (default: all movies in --nodes)'); a = ap.parse_args()
    device = torch.device(f'cuda:{a.gpu}'); torch.cuda.set_device(device)
    model, W, ds = load_edge_transformer(Path(a.weights), device)
    outs = {k: Path(a.out_root) / f'edges_tf_{k}' for k in ('fwd', 'rev', 'harm')}
    for p in outs.values():
        p.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f'{a.nodes}/*.npz'))
    if a.movies:
        files = [f for f in files if Path(f).stem in set(a.movies.split(','))]
    for i, f in enumerate(files, 1):
        z = np.load(f); c = np.load(f'{a.cand}/{Path(f).name}'); name = Path(f).stem
        out = score_movie(model, W, ds, Path(a.test) / f'{name}.zarr', z, c['I'], c['J'], device)
        for k, p in out.items():
            np.savez_compressed(outs[k] / Path(f).name, I=c['I'], J=c['J'], p=p)
        print(f'  transformer links [{i}/{len(files)}] {name}', flush=True)
