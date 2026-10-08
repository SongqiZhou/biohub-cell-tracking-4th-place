#!/usr/bin/env python
"""Train the division CNN (inference/division_cnn.py) on the patches of division_patches.py.

Divisions are rare (151 of 128,732 patches), so every batch of 56 holds 10 divisions drawn with replacement and 46
non-divisions. 60 epochs of 150 batches, AdamW (lr 1.2e-3, weight decay 3e-4) with a one-cycle schedule, BCE with
positive weight 2.5. Augmentation: random flips of X, Y and Z, X/Y transposition, a shift of up to 2 voxels, intensity
scaling 0.8-1.2 and Gaussian noise.

--paste adds 12 synthetic divisions per real one: the bright voxels of a division patch (a 5 um ball around the mother
in frames t-2..t, a 9 um ball around the daughters in t+1..t+2, above the 85% intensity quantile of the ball) are
max-blended, scaled by 0.8-1.2, into a random non-division patch of another movie.

    # deployed: three seeds each, trained on all movies
    python training/train_division_cnn.py --patches work/div/patches --out models/division_cnn
    python training/train_division_cnn.py --patches work/div/patches --out models/division_cnn --paste
    # out-of-fold scores for the division CatBoost: leave half k out, two seeds
    python training/train_division_cnn.py --patches work/div/patches --out work/div/cnn_oof --half 0 --seeds 2
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'inference'))
from division_cnn import DivNet  # noqa: E402

UM = (1.625, 0.40625, 0.40625)


def load_patches(d):
    files = sorted(glob.glob(f'{d}/shard*.npz'), key=lambda f: int(Path(f).stem[5:]))
    parts = [np.load(f) for f in files]
    X = np.concatenate([p['X'] for p in parts]); y = np.concatenate([p['y'] for p in parts]).astype(np.int64)
    movie = np.array([m.split('|')[0] for p in parts for m in p['meta']])
    return X, y, movie


def augment(xb):
    if torch.rand(1).item() < 0.5:
        xb = xb.flip(-1)
    if torch.rand(1).item() < 0.5:
        xb = xb.flip(-2)
    if torch.rand(1).item() < 0.35:
        xb = xb.flip(-3)
    if torch.rand(1).item() < 0.5:
        xb = xb.transpose(-1, -2)
    if torch.rand(1).item() < 0.3:
        sy, sx = int(torch.randint(-2, 3, (1,))), int(torch.randint(-2, 3, (1,)))
        xb = torch.roll(xb, shifts=(sy, sx), dims=(-2, -1))
    xb = xb * (0.8 + 0.4 * torch.rand(xb.shape[0], 1, 1, 1, 1, device=xb.device))
    return xb + 0.02 * torch.randn_like(xb)


def train(X, y, seed, dev, epochs=60, steps=150, batch=56, pos_per_batch=10):
    torch.manual_seed(seed)
    net = DivNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1.2e-3, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.2e-3, total_steps=epochs * steps)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    pw = torch.tensor(2.5, device=dev)
    for ep in range(epochs):
        net.train(); tot = 0.0
        for _ in range(steps):
            idx = np.concatenate([rng.choice(pos, pos_per_batch, replace=True), rng.choice(neg, batch - pos_per_batch, replace=False)])
            xb = torch.as_tensor(X[idx].astype(np.float32)).to(dev); yb = torch.as_tensor(y[idx].astype(np.float32)).to(dev)
            loss = F.binary_cross_entropy_with_logits(net(augment(xb)), yb, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f'  seed {seed} epoch {ep + 1}/{epochs}: loss {tot / steps:.4f}', flush=True)
    return net


def paste_divisions(X, y, movie, keep, rng, per_division=12, q=0.85):
    """synthetic division patches built from the training patches `keep` (bool mask)"""
    T, Z, Y, Xd = X.shape[1:]
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(Xd), indexing='ij')
    r = np.sqrt(((zz - Z // 2) * UM[0]) ** 2 + ((yy - Y // 2) * UM[1]) ** 2 + ((xx - Xd // 2) * UM[2]) ** 2)
    ball = np.stack([r <= 5.0] * 3 + [r <= 9.0] * (T - 3), 0)

    def blend(P, N):
        P = P.astype(np.float32); N = N.astype(np.float32); M = np.zeros_like(P, dtype=bool)
        for f in range(T):
            M[f] = ball[f] & (P[f] > np.quantile(P[f][ball[f]], q))
        s = rng.uniform(0.8, 1.2); o = N.copy(); o[M] = np.maximum(o[M], s * P[M])
        return o.astype(np.float16)

    pos, neg = np.flatnonzero(keep & (y == 1)), np.flatnonzero(keep & (y == 0))
    out = []
    for i in pos:
        others = neg[movie[neg] != movie[i]]
        for _ in range(per_division):
            out.append(blend(X[i], X[rng.choice(others)]))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patches', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--paste', action='store_true'); ap.add_argument('--half', type=int, default=-1, choices=[-1, 0, 1],
                                                                        help='leave out this half of the movies (training/fold_movies.py --half)')
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}'); torch.cuda.set_device(dev)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X, y, movie = load_patches(a.patches)
    keep = np.ones(len(y), bool)
    if a.half >= 0:
        held = subprocess.run([sys.executable, str(HERE / 'fold_movies.py'), '--half', str(a.half)], capture_output=True, text=True, check=True).stdout.strip().split(',')
        keep = ~np.isin(movie, held)
    print(f'{len(y)} patches, {int(y.sum())} divisions; training on {int(keep.sum())} ({int(y[keep].sum())} divisions)', flush=True)
    Xt, yt = X[keep], y[keep]
    if a.paste:
        syn = paste_divisions(X, y, movie, keep, np.random.default_rng(7 if a.half < 0 else 100 + a.half))
        Xt = np.concatenate([Xt, syn]); yt = np.concatenate([yt, np.ones(len(syn), np.int64)])
        print(f'+ {len(syn)} synthetic divisions', flush=True)
    del X
    name = ('cnn_copypaste' if a.paste else 'cnn') + (f'_half{a.half}' if a.half >= 0 else '')
    for s in range(a.seeds):
        net = train(Xt, yt, s, dev, epochs=a.epochs)
        torch.save(net.state_dict(), out / f'{name}_s{s}.pt'); print('saved', out / f'{name}_s{s}.pt', flush=True)


if __name__ == '__main__':
    main()
