#!/usr/bin/env python
"""Division CNN: every node gets the probability that it is a mother about to divide, from a 5-frame (t-2..t+2) raw patch of
9 x 33 x 33 voxels around it. Ensemble of the networks matching --model (glob), each averaged over 4 flips.
Writes <out-dir>/<movie>.divscore.npz (keys (t, z, y, x) voxels, score).
usage: division_cnn.py --cand-dir DIR --data-dir ZARR_DIR --out-dir DIR --model 'models/division_cnn/*.pt' [--half]"""
import argparse
import glob
import json
from pathlib import Path

import blosc2
import numpy as np
import torch
import torch.nn as nn

HZ, HY, HX = 4, 16, 16
TW = 2


class DivNet(nn.Module):
    def __init__(self, cin=5, ch=48):
        super().__init__()

        def blk(i, o, pool):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.SiLU(),
                                 nn.Conv3d(o, o, 3, padding=1), nn.BatchNorm3d(o), nn.SiLU(), nn.MaxPool3d(pool))
        self.b1 = blk(cin, ch, (1, 2, 2)); self.b2 = blk(ch, ch * 2, (2, 2, 2)); self.b3 = blk(ch * 2, ch * 3, (2, 2, 2))
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(ch * 3 * 2, 128), nn.SiLU(), nn.Dropout(0.25), nn.Linear(128, 1))

    def forward(self, x):
        h = self.b3(self.b2(self.b1(x)))
        return self.head(torch.cat([h.mean(dim=(2, 3, 4)), h.amax(dim=(2, 3, 4))], 1)).squeeze(1)


def read_frame(zp, t, shape, dtype, cache):
    """One frame straight from its zarr v3 chunk (the movies store one chunk per time point)."""
    if t in cache:
        return cache[t]
    chunk = zp / '0' / 'c' / str(t) / '0' / '0' / '0'
    if not chunk.exists():
        return None
    arr = np.frombuffer(blosc2.decompress(chunk.read_bytes()), dtype=dtype).reshape(shape[1:])
    if len(cache) > 6:
        cache.pop(next(iter(cache)))
    cache[t] = arr
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand-dir', required=True); ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model', required=True); ap.add_argument('--tta', type=int, default=4); ap.add_argument('--chunk', type=int, default=384)
    ap.add_argument('--half', action='store_true', help='fp16 inference')
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nets = []
    for p in sorted(glob.glob(a.model)):
        n = DivNet().to(dev); n.load_state_dict(torch.load(p, map_location=dev)); n.eval()
        nets.append(n.half() if a.half else n)
    print(f'division CNN: ensemble of {len(nets)}, {a.tta} flips', flush=True)
    oz = torch.arange(-HZ, HZ + 1, device=dev); oy = torch.arange(-HY, HY + 1, device=dev); ox = torch.arange(-HX, HX + 1, device=dev)
    for cp in sorted(Path(a.cand_dir).glob('*.candidates.npz')):
        ds = cp.name.replace('.candidates.npz', '')
        zp = Path(a.data_dir) / f'{ds}.zarr'
        meta = json.loads((zp / '0' / 'zarr.json').read_text())
        shape = tuple(meta['shape']); dtype = np.dtype(meta['data_type'])
        try:
            q = json.loads((zp / 'zarr.json').read_text()).get('attributes', {}).get('image_statistics', {}).get('quantiles', {})
            qlo, qhi = float(q.get('0.001', 0)), float(q.get('0.999', 1))
        except Exception:
            qlo, qhi = 0.0, 1.0
        coords = np.load(cp)['coords']
        by_t = {}
        for t, z, y, x in coords:
            by_t.setdefault(int(round(float(t))), []).append((int(round(float(z))), int(round(float(y))), int(round(float(x)))))
        cache = {}; keys, scores = [], []
        for t in sorted(by_t):
            frames = [read_frame(zp, min(max(t + dt, 0), shape[0] - 1), shape, dtype, cache) for dt in range(-TW, TW + 1)]
            if any(f is None for f in frames):
                continue
            fr = torch.as_tensor(np.stack(frames).astype(np.int32)).to(dev)
            Z, Y, X = fr.shape[1:]
            pts = np.asarray(by_t[t], dtype=np.int64)
            for s0 in range(0, len(pts), a.chunk):
                p_ = torch.as_tensor(pts[s0:s0 + a.chunk]).to(dev)
                zr = p_[:, 0:1] + oz; yr = p_[:, 1:2] + oy; xr = p_[:, 2:3] + ox
                vz = (zr >= 0) & (zr < Z); vy = (yr >= 0) & (yr < Y); vx = (xr >= 0) & (xr < X)
                zi = zr.clamp(0, Z - 1); yi = yr.clamp(0, Y - 1); xi = xr.clamp(0, X - 1)
                g = fr[:, zi[:, :, None, None], yi[:, None, :, None], xi[:, None, None, :]].permute(1, 0, 2, 3, 4)
                m = vz[:, :, None, None] & vy[:, None, :, None] & vx[:, None, None, :]
                xb = ((g * m[:, None]).float() - qlo) / max(qhi - qlo, 1e-6)
                if a.half:
                    xb = xb.half()
                views = [xb]
                if a.tta >= 2:
                    views.append(xb.flip(-1))
                if a.tta >= 4:
                    views += [xb.flip(-2), xb.flip(-1).flip(-2)]
                with torch.no_grad():
                    p = torch.stack([torch.sigmoid(n(v)) for n in nets for v in views]).mean(0)
                scores.append(p.float().cpu().numpy())
            keys += [(t, z, y, x) for (z, y, x) in by_t[t]]
        sc = np.concatenate(scores) if scores else np.zeros(0, np.float32)
        np.savez_compressed(out / f'{ds}.divscore.npz', keys=np.array(keys, dtype=np.int32), score=sc.astype(np.float32))
        print(f'  division CNN {ds}: {len(keys)} nodes', flush=True)


if __name__ == '__main__':
    main()
