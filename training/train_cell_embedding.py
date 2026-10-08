#!/usr/bin/env python
"""Contrastive cell-identity embedding (a small 3D CNN on a 16 x 32 x 32 raw patch around each node).

Training pairs are the labelled candidate links of the out-of-fold node set: y = 1 (the annotated link) is a positive
pair, y = 0 (another candidate of the same source) a hard negative. The loss is a symmetric InfoNCE over the batch of
positives, with up to 3 hard negatives per positive as extra columns.

    # 1. patches of all nodes that take part in a labelled pair
    python training/train_cell_embedding.py pairs --nodes work/link/nodes --cand work/link/cand_pool --labels work/link/labels \
        --data data/train --out work/link/pairs
    # 2. two out-of-fold encoders (every second movie by sorted name); each scores its held-out half:
    #    <out>/<movie>.npz (I, J, p = (cos + 1) / 2) and <emb-out>/<movie>.npy (node embeddings, float16)
    python training/train_cell_embedding.py train --pairs work/link/pairs --fold 0 --nodes work/link/nodes --cand work/link/cand_pool \
        --data data/train --out work/link/edges_embed --emb-out work/link/embeddings
    python training/train_cell_embedding.py train ... --fold 1 ...
    # 3. the deployed encoder, trained on all movies
    python training/train_cell_embedding.py train --pairs work/link/pairs --all models/cell_embedding.pt
"""
import argparse
import collections
import glob
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from cell_embedding import Encoder, norm_patch  # noqa: E402
from common import ZarrFrames  # noqa: E402

PZ, PYX = 16, 32


def crop_batch(vol, vox):
    """vol (Z, Y, X) uint16, vox (n, 3) int -> (n, PZ, PYX, PYX) uint16 centred on the voxel, zero outside the volume"""
    Z, Y, X = vol.shape; hz, hy = PZ // 2, PYX // 2
    kz = np.arange(-hz, PZ - hz); ky = np.arange(-hy, PYX - hy)
    zi = vox[:, 0:1] + kz[None]; yi = vox[:, 1:2] + ky[None]; xi = vox[:, 2:3] + ky[None]
    vz = (zi >= 0) & (zi < Z); vy = (yi >= 0) & (yi < Y); vx = (xi >= 0) & (xi < X)
    out = vol[np.clip(zi, 0, Z - 1)[:, :, None, None], np.clip(yi, 0, Y - 1)[:, None, :, None], np.clip(xi, 0, X - 1)[:, None, None, :]]
    out *= (vz[:, :, None, None] & vy[:, None, :, None] & vx[:, None, None, :])
    return out


# ------------------------------------------------------------------------------------------------------------ pairs
def build_pairs(args):
    m, a = args
    out = Path(a.out) / f'{m}.npz'
    z = np.load(Path(a.nodes) / f'{m}.npz'); t = z['t'].astype(int); vox = np.rint(z['vox']).astype(int)
    c = np.load(Path(a.cand) / f'{m}.npz'); y = np.load(Path(a.labels) / f'{m}.npz')['y'].astype(int)
    I, J = c['I'].astype(int), c['J'].astype(int)
    sel = np.where(y >= 0)[0]
    if len(sel) == 0:
        np.savez_compressed(out, node_id=np.zeros(0, int), patch=np.zeros((0, PZ, PYX, PYX), np.uint16), pair_a=np.zeros(0, int),
                            pair_b=np.zeros(0, int), y=np.zeros(0, int))
        return m, 0, 0
    I, J, y = I[sel], J[sel], y[sel]
    nodes = np.unique(np.concatenate([I, J])); pos = {int(n): k for k, n in enumerate(nodes)}
    arr = ZarrFrames(Path(a.data) / f'{m}.zarr')
    patches = np.zeros((len(nodes), PZ, PYX, PYX), np.uint16)
    for tt in np.unique(t[nodes]):
        ks = np.where(t[nodes] == tt)[0]
        patches[ks] = crop_batch(np.asarray(arr[int(tt)]), vox[nodes[ks]])
    np.savez_compressed(out, node_id=nodes, patch=patches, pair_a=np.array([pos[int(i)] for i in I]),
                        pair_b=np.array([pos[int(j)] for j in J]), y=y)
    return m, len(nodes), len(y)


def load_pairs(files):
    P, PA, PB, Y, off = [], [], [], [], 0
    for f in files:
        z = np.load(f)
        if len(z['y']) == 0:
            continue
        P.append(z['patch']); PA.append(z['pair_a'] + off); PB.append(z['pair_b'] + off); Y.append(z['y']); off += len(z['node_id'])
    return np.concatenate(P), np.concatenate(PA), np.concatenate(PB), np.concatenate(Y)


# ------------------------------------------------------------------------------------------------------------ train
def train_encoder(P, PA, PB, Y, rng, dev, epochs=8, emb=64, bs=256, lr=1e-3, tau=0.07, neg_per_pos=3):
    pos_idx = np.where(Y == 1)[0]; negs = collections.defaultdict(list)
    for q in np.where(Y == 0)[0]:
        negs[int(PA[q])].append(int(PB[q]))
    Pt = torch.from_numpy(P.astype(np.int32)); enc = Encoder(emb).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(pos_idx) / bs))
    for ep in range(epochs):
        enc.train(); perm = rng.permutation(pos_idx); tl = 0.0; nb = 0; t0 = time.time()
        for c in range(0, len(perm), bs):
            q = perm[c:c + bs]; a, b = PA[q], PB[q]; hn = []
            for s in a:
                cand = negs.get(int(s), [])
                if cand:
                    hn.extend(rng.choice(cand, size=min(neg_per_pos, len(cand)), replace=False).tolist())
            hn = np.array(hn, int) if hn else np.zeros(0, int)
            ea = enc(norm_patch(Pt[a].to(dev))); eb = enc(norm_patch(Pt[np.concatenate([b, hn])].to(dev)))
            logits = ea @ eb.T / tau; target = torch.arange(len(a), device=dev)
            loss = F.cross_entropy(logits, target) + F.cross_entropy(logits[:, :len(a)].T, target)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item(); nb += 1
        print(f'  epoch {ep + 1}/{epochs}: loss {tl / max(nb, 1):.4f} ({time.time() - t0:.0f}s)', flush=True)
    return enc.eval()


@torch.no_grad()
def score_movie(enc, m, a, dev, emb=64):
    """embeddings of all nodes (fp32) -> p for every candidate link, and the node embeddings"""
    z = np.load(Path(a.nodes) / f'{m}.npz'); t = z['t'].astype(int); vox = np.rint(z['vox']).astype(int)
    c = np.load(Path(a.cand) / f'{m}.npz'); I, J = c['I'].astype(int), c['J'].astype(int)
    arr = ZarrFrames(Path(a.data) / f'{m}.zarr'); E = torch.zeros(len(t), emb, device=dev)
    for tt in np.unique(t):
        ks = np.where(t == tt)[0]; vol = np.asarray(arr[int(tt)])
        for s in range(0, len(ks), 2048):
            kk = ks[s:s + 2048]
            E[kk] = enc(norm_patch(torch.from_numpy(crop_batch(vol, vox[kk]).astype(np.int32)).to(dev)))
    cos = (E[torch.from_numpy(I).to(dev)] * E[torch.from_numpy(J).to(dev)]).sum(1)
    return I, J, ((cos + 1) / 2).cpu().numpy().astype(np.float32), E.cpu().numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['pairs', 'train'])
    ap.add_argument('--nodes'); ap.add_argument('--cand'); ap.add_argument('--labels'); ap.add_argument('--data'); ap.add_argument('--out')
    ap.add_argument('--pairs'); ap.add_argument('--fold', type=int, default=-1); ap.add_argument('--all', default='', help='train on all movies and save the encoder here')
    ap.add_argument('--emb-out'); ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'pairs':
        os.makedirs(a.out, exist_ok=True)
        movies = sorted(Path(f).stem for f in glob.glob(f'{a.nodes}/*.npz'))
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(build_pairs, [(m, a) for m in movies]))
        print(f'pairs: {len(res)} movies, {sum(r[1] for r in res)} patches, {sum(r[2] for r in res)} labelled pairs', flush=True)
        return
    torch.manual_seed(0); np.random.seed(0); dev = torch.device(f'cuda:{a.gpu}')
    files = sorted(glob.glob(f'{a.pairs}/*.npz')); halves = [files[0::2], files[1::2]]
    if a.all:
        enc = train_encoder(*load_pairs(files), np.random.default_rng(1), dev, epochs=a.epochs)
        torch.save(dict(enc=enc.state_dict(), emb=64), a.all); print('saved', a.all, flush=True)
        return
    k = a.fold
    enc = train_encoder(*load_pairs(halves[1 - k]), np.random.default_rng(k), dev, epochs=a.epochs)
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.emb_out, exist_ok=True)
    torch.save(dict(enc=enc.state_dict(), emb=64), Path(a.out) / f'model_fold{k}.pt')
    for f in halves[k]:
        m = Path(f).stem; I, J, p, E = score_movie(enc, m, a, dev)
        np.savez_compressed(Path(a.out) / f'{m}.npz', I=I, J=J, p=p); np.save(Path(a.emb_out) / f'{m}.npy', E)
    print(f'fold {k}: scored {len(halves[k])} held-out movies', flush=True)


if __name__ == '__main__':
    main()
