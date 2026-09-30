#!/usr/bin/env python
"""Contrastive cell-identity embedding. A small 3D CNN, trained so that the same nucleus in consecutive frames maps to
nearby unit vectors, encodes a (16, 32, 32) raw patch around every node. Each candidate link gets
p = (cos(e_i, e_j) + 1) / 2. Writes <out>/<movie>.npz (I, J, p) and the node embeddings <emb-out>/<movie>.npy.
usage: cell_embedding.py --nodes DIR --cand DIR --test ZARR_DIR --model models/cell_embedding.pt --out DIR --emb-out DIR"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

PZ, PYX = 16, 32


class Encoder(nn.Module):
    def __init__(self, emb, w=1, in_ch=1):
        super().__init__(); c1, c2, c3, c4 = 16 * w, 32 * w, 64 * w, 96 * w
        self.net = nn.Sequential(nn.Conv3d(in_ch, c1, (3, 5, 5), 1, (1, 2, 2)), nn.BatchNorm3d(c1), nn.GELU(),
                                 nn.Conv3d(c1, c2, 3, (1, 2, 2), 1), nn.BatchNorm3d(c2), nn.GELU(),
                                 nn.Conv3d(c2, c3, 3, 2, 1), nn.BatchNorm3d(c3), nn.GELU(),
                                 nn.Conv3d(c3, c4, 3, 1, 1), nn.BatchNorm3d(c4), nn.GELU(),
                                 nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(c4, emb))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def norm_patch(p):
    """(n, Z, Y, X) raw patches -> (n, 1, Z, Y, X), log1p then per-patch z-score."""
    if p.dim() == 4:
        p = p.unsqueeze(1)
    x = torch.log1p(p.float()); mu = x.mean(dim=(1, 2, 3, 4), keepdim=True); sd = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-3
    return (x - mu) / sd


def crop(vol, vox, dev):
    """vol (Z, Y, X) int32 on the device, vox (n, 3) -> (n, PZ, PYX, PYX), zero outside the volume."""
    Z, Y, X = vol.shape; hz, hy = PZ // 2, PYX // 2; v = torch.as_tensor(vox, device=dev)
    kz = torch.arange(-hz, PZ - hz, device=dev); ky = torch.arange(-hy, PYX - hy, device=dev)
    zi = v[:, 0:1] + kz[None]; yi = v[:, 1:2] + ky[None]; xi = v[:, 2:3] + ky[None]
    valid = ((zi >= 0) & (zi < Z))[:, :, None, None] & ((yi >= 0) & (yi < Y))[:, None, :, None] & ((xi >= 0) & (xi < X))[:, None, None, :]
    return vol[zi.clamp(0, Z - 1)[:, :, None, None], yi.clamp(0, Y - 1)[:, None, :, None], xi.clamp(0, X - 1)[:, None, None, :]] * valid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--cand', required=True); ap.add_argument('--test', required=True)
    ap.add_argument('--model', required=True); ap.add_argument('--out', required=True); ap.add_argument('--emb-out', required=True)
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--batch', type=int, default=4096); A = ap.parse_args()
    dev = torch.device(f'cuda:{A.gpu}' if torch.cuda.is_available() else 'cpu'); ck = torch.load(A.model, map_location=dev)
    enc = Encoder(int(ck['emb']), int(ck.get('width', 1)), int(ck['enc']['net.0.weight'].shape[1])).to(dev); enc.load_state_dict(ck['enc']); enc.eval()
    os.makedirs(A.out, exist_ok=True); os.makedirs(A.emb_out, exist_ok=True); t0 = time.time()
    for f in sorted(glob.glob(os.path.join(A.nodes, '*.npz'))):
        m = os.path.basename(f)[:-4]; z = np.load(f); t = z['t'].astype(int); vox = np.rint(z['vox']).astype(np.int64); N = len(t)
        c = np.load(os.path.join(A.cand, m + '.npz')); I, J = c['I'].astype(np.int64), c['J'].astype(np.int64)
        arr = zarr.open(os.path.join(A.test, m + '.zarr'), mode='r')['0']
        E = torch.zeros(N, int(ck['emb']), device=dev)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dev.type == 'cuda')):
            for tt in np.unique(t):
                vol = torch.as_tensor(np.asarray(arr[int(tt)]).astype(np.int32), device=dev)
                ks = np.where(t == tt)[0]
                for s in range(0, len(ks), A.batch):
                    kk = ks[s:s + A.batch]
                    E[kk] = enc(norm_patch(crop(vol, vox[kk], dev))).float()
            cos = (E[torch.as_tensor(I, device=dev)] * E[torch.as_tensor(J, device=dev)]).sum(1)
        p = ((cos + 1) / 2).clamp(0, 1).cpu().numpy().astype(np.float32)
        np.savez_compressed(os.path.join(A.out, m + '.npz'), I=I, J=J, p=p)
        np.save(os.path.join(A.emb_out, m + '.npy'), E.cpu().numpy().astype(np.float32))
        print(f'  cell embedding {m}: {N} nodes, {len(I)} links ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
