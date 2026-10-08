#!/usr/bin/env python
"""Candidate graphs for the division models: nodes plus the candidate links with probability >= 0.02, in the format the
division CNN and the division features read (<movie>.candidates.npz: coords (N, 4) t, z, y, x voxels; edges (M, 4) i, j,
p, distance in voxels). The same graph is built inside inference/downstream.py from the deployed edge model.

    python training/candidate_graphs.py --nodes work/div/nodes --edges work/div/edges_oof --out work/div/cand_npz
    python training/candidate_graphs.py ... --half 0 --out work/div/cand_half0      # only the movies of one half
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--edges', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--p-min', type=float, default=0.02); ap.add_argument('--half', type=int, default=-1, choices=[-1, 0, 1]); a = ap.parse_args()
keep = None
if a.half >= 0:
    keep = set(subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'fold_movies.py'), '--half', str(a.half)],
                              capture_output=True, text=True, check=True).stdout.strip().split(','))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = 0
for f in sorted(glob.glob(f'{a.nodes}/*.npz')):
    if keep is not None and Path(f).stem not in keep:
        continue
    z = np.load(f); e = np.load(Path(a.edges) / Path(f).name); m = e['p'] >= a.p_min
    I, J, p = e['I'][m], e['J'][m], e['p'][m]
    np.savez_compressed(out / f'{Path(f).stem}.candidates.npz', coords=np.concatenate([z['t'][:, None], np.rint(z['vox'])], 1).astype(np.int32),
                        edges=np.stack([I, J, p, np.linalg.norm(z['vox'][J] - z['vox'][I], axis=1)], 1).astype(np.float64)); n += 1
print(f'candidate graphs: {n} movies -> {out}', flush=True)
