#!/usr/bin/env python
"""Train a 3D Net (cell probability + unit vector field).

    # 3D Net-128, ground truth only, all 199 movies (or --fold k to leave fold k out)
    python training/train_net3d.py --data data/train --out runs/net3d_128 --pool 2 --epochs 10
    # 3D Net-128-PL: competition movies + external windows, with pseudo-labels (see external_data/)
    python training/train_net3d.py --data data/mix/data --pseudo data/mix/pseudo --out runs/net3d_128_pl --pool 2 --epochs 10
    # 3D Net-64
    python training/train_net3d.py --data data/train --out runs/net3d_64 --pool 4 --epochs 50

Every epoch is saved as <out>/ep<k>.pt; average_checkpoints.py turns them into the deployed weights.
Recipe: Adam, lr 3e-4, batch 4, 16 random labelled frames per movie and epoch, bf16 autocast, gradient clipping at 10.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'inference'))
from net3d import Net3D  # noqa: E402
from net3d_data import FrameDataset, Frames, epoch_items, frame_index, grid_scale, load_folds, merge_labels  # noqa: E402


def position_um(shape, gs, device):
    Z, Y, X = shape
    axes = [torch.arange(n, device=device, dtype=torch.float32) * float(s) for n, s in zip((Z, Y, X), gs)]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'))                     # (3, Z, Y, X)


def net3d_loss(out, owner, weight, centre_um, pos_um):
    """L1 between the predicted and the target unit vectors on region voxels, plus the tier-weighted BCE of the
    cell probability (target 1 on region voxels, 0 elsewhere)."""
    flow, prob = out['flow'], out['prob']
    B = flow.shape[0]
    own = owner.view(B, -1)
    sel = (own >= 0).reshape(-1).nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        l_flow = flow.sum() * 0.0
    else:
        K = centre_um.shape[1]
        gid = (own + (torch.arange(B, device=flow.device) * K).view(B, 1)).clamp_min(0).reshape(-1)[sel]
        p = pos_um.unsqueeze(0).expand(B, -1, -1, -1, -1).permute(0, 2, 3, 4, 1).reshape(-1, 3)[sel]
        d = centre_um.reshape(-1, 3)[gid] - p
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
        l_flow = (flow.permute(0, 2, 3, 4, 1).reshape(-1, 3)[sel] - d).abs().sum(-1).mean()
    w = weight.reshape(-1)
    target = (own >= 0).to(prob.dtype).reshape(-1)
    l_prob = (F.binary_cross_entropy_with_logits(prob.reshape(-1).float(), target, reduction='none') * w).sum() / w.sum().clamp_min(1e-6)
    return l_flow + l_prob, l_flow, l_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='directory with <movie>.zarr (+ <movie>.geff labels)')
    ap.add_argument('--pseudo', default='', help='directory with <movie>.npz pseudo-labels')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pool', type=int, default=2, help='XY pooling of the input grid: 2 -> 3D Net-128, 4 -> 3D Net-64')
    ap.add_argument('--fold', type=int, default=-1, help='leave this fold out (training/folds.json); -1 = train on all movies')
    ap.add_argument('--folds', default=str(HERE / 'folds.json'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--frames-per-movie', type=int, default=16)
    ap.add_argument('--seed', type=int, default=314159)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f'cuda:{a.gpu}')
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    names = sorted(p.name[:-5] for p in Path(a.data).iterdir() if p.name.endswith('.zarr'))
    if a.fold >= 0:
        held_out = set(load_folds(Path(a.folds))[f'fold{a.fold}'])
        names = [n for n in names if n not in held_out]
    labels = merge_labels(names, Path(a.data), Path(a.pseudo) if a.pseudo else None)
    index = frame_index(labels)
    config = dict(c=32, heads=4, blocks=2, pool_xy=a.pool, window_half=1, tag=out.name)
    model = Net3D(config['c'], config['heads'], config['blocks']).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gs = grid_scale(a.pool)
    frames = Frames(Path(a.data), a.pool)
    print(f'{len(names)} movies, {len(index)} labelled frames, {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters', flush=True)
    pos_um = None
    for ep in range(1, a.epochs + 1):
        items = epoch_items(index, np.random.default_rng(a.seed + ep), a.frames_per_movie)
        ds = FrameDataset(items, labels, frames, a.pool, seed=a.seed + ep)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
        model.train(); t0, tot, nb = time.time(), 0.0, 0
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            if pos_um is None:
                pos_um = position_um(b['img'].shape[-3:], gs, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(b['img'])
            o = {k: v.float() for k, v in o.items()}
            loss, _, _ = net3d_loss(o, b['owner'], b['weight'], b['centre_um'], pos_um)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item(); nb += 1
            if nb % 50 == 0:
                print(f'  ep{ep} step {nb}/{len(dl)} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)', flush=True)
        torch.save({'model': model.state_dict(), 'config': config, 'epoch': ep}, out / f'ep{ep}.pt')
        print(f'epoch {ep}/{a.epochs}: loss {tot / max(nb, 1):.4f} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
