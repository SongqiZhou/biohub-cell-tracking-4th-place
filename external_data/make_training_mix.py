#!/usr/bin/env python
"""Assemble the training set of the pseudo-label 3D Net (3D Net-128-PL): competition movies + external windows.

Creates symlinks only (no copies):
  <out>/data/    the 199 competition training movies (<id>.zarr + <id>.geff) and the external windows (<id>.zarr, no labels)
  <out>/pseudo/  pseudo-labels of both parts (<id>.npz from make_pseudo_labels.py)

    python external_data/make_training_mix.py --train data/train --external data/ultrack_windows \
        --pseudo-train data/pseudo_labels/pseudo_oof --pseudo-external data/pseudo_labels/pseudo_ultrack \
        --out data/mix
"""
import argparse
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--train', required=True, help='competition training data (*.zarr, *.geff)')
ap.add_argument('--external', required=True, help='external windows (*.zarr)')
ap.add_argument('--pseudo-train', required=True, help='pseudo-labels of the competition movies (out-of-fold detections)')
ap.add_argument('--pseudo-external', required=True, help='pseudo-labels of the external windows')
ap.add_argument('--out', required=True)
a = ap.parse_args()


def link(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    if not dst.exists():
        os.symlink(src.resolve(), dst)


out = Path(a.out)
(out / 'data').mkdir(parents=True, exist_ok=True)
(out / 'pseudo').mkdir(parents=True, exist_ok=True)
movies = sorted(p for p in Path(a.train).iterdir() if p.suffix in ('.zarr', '.geff'))
windows = sorted(p for p in Path(a.external).iterdir() if p.suffix == '.zarr')
for p in movies + windows:
    link(p, out / 'data')
pseudo = sorted(Path(a.pseudo_train).glob('*.npz')) + sorted(Path(a.pseudo_external).glob('*.npz'))
for p in pseudo:
    link(p, out / 'pseudo')
n_train = sum(p.suffix == '.zarr' for p in movies)
print(f'{n_train} competition movies + {len(windows)} external windows -> {out / "data"}; {len(pseudo)} pseudo-label files -> {out / "pseudo"}')
