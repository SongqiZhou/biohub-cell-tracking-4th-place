#!/usr/bin/env python
"""Everything after detection: nodes -> candidate links -> link probabilities -> division prior -> ILP -> fork verifier
-> gap filling -> submission.csv.

usage: downstream.py --nodes DET_DIR --test ZARR_DIR --out submission.csv --work DIR --route route.json [--gpus 0,1]
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M = HERE / 'models'

ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--test', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--work', default='ds'); ap.add_argument('--route', required=True); ap.add_argument('--gpus', default='0'); ap.add_argument('--jobs', type=int, default=3)
ap.add_argument('--div-costs', default='6,16,0.5', help='ILP division cost base,gain,relax'); ap.add_argument('--sparse-div-costs', default='5,16,0.5', help='the same for sparse movies')
ap.add_argument('--fork-thr', default='0.25'); a = ap.parse_args()
W = Path(a.work); W.mkdir(parents=True, exist_ok=True); PY = sys.executable
env = dict(os.environ, POLARS_MAX_THREADS='2', OMP_NUM_THREADS='4')
gpus = [g for g in a.gpus.split(',') if g]
VISIBLE = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d]


def run(script, *args, **kw):
    cmd = [PY, str(HERE / script), *[str(x) for x in args]]
    print('>>', ' '.join(cmd), flush=True); subprocess.run(cmd, check=True, env=kw.get('env', env))


# 1. nodes, candidate links and their features
run('nodes_table.py', '--nodes', a.nodes, '--out', W / 'nodes')
run('candidates.py', '--nodes', W / 'nodes', '--out', W / 'cand_pool', '--r', 10, '--k', 5, '--k-in', 3)
run('edge_features.py', '--nodes', W / 'nodes', '--cand', W / 'cand_pool', '--out', W / 'feat')
run('tf_edges.py', '--nodes', W / 'nodes', '--cand', W / 'cand_pool', '--test', a.test, '--weights', M / 'edge_transformer' / 'weights.pth', '--out-root', W, '--gpu', gpus[0])
run('cell_embedding.py', '--nodes', W / 'nodes', '--cand', W / 'cand_pool', '--test', a.test, '--model', M / 'cell_embedding.pt', '--out', W / 'edges_embed',
    '--emb-out', W / 'embeddings', '--gpu', gpus[0])
# 2. learned link probability
run('edge_model.py', '--model', M / 'edge_lgbm', '--feat', W / 'feat', '--root', W, '--out', W / 'edges_learned')

# 3. division prior: CNN -> CatBoost node prior -> pair model, MAX-combined
cand = W / 'cand_npz'; cand.mkdir(exist_ok=True)
for f in sorted(glob.glob(str(W / 'nodes' / '*.npz'))):
    z = np.load(f); e = np.load(W / 'edges_learned' / Path(f).name); keep = e['p'] >= 0.02
    I, J, p = e['I'][keep], e['J'][keep], e['p'][keep]
    np.savez_compressed(cand / f'{Path(f).stem}.candidates.npz', coords=np.concatenate([z['t'][:, None], np.rint(z['vox'])], 1).astype(np.int32),
                        edges=np.stack([I, J, p, np.linalg.norm(z['vox'][J] - z['vox'][I], axis=1)], 1).astype(np.float64))
files = sorted(glob.glob(str(cand / '*.candidates.npz')), key=lambda f: len(np.load(f)['coords']), reverse=True)
bins, load = [[] for _ in gpus], [0] * len(gpus)                  # one CNN process per GPU, movies dealt by node count
for f in files:
    i = load.index(min(load)); bins[i].append(f); load[i] += len(np.load(f)['coords'])
procs = []
for g, b in zip(gpus, bins):
    if not b:
        continue
    d = W / f'cand_npz_gpu{g}'; d.mkdir(exist_ok=True)
    for f in b:
        if not (d / Path(f).name).exists():
            os.symlink(os.path.abspath(f), d / Path(f).name)
    cmd = [PY, str(HERE / 'division_cnn.py'), '--cand-dir', str(d), '--data-dir', a.test, '--out-dir', str(W / 'div_cnn'), '--model', str(M / 'division_cnn' / '*.pt'), '--tta', '4', '--half']
    print(f'>> [gpu {g}, {len(b)} movies]', ' '.join(cmd), flush=True)
    procs.append(subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=VISIBLE[int(g)] if VISIBLE else g)))
assert all(p.wait() == 0 for p in procs), 'division CNN failed'
run('division_cat.py', '--model', M / 'division_cat.cbm', '--cand', cand, '--div-dir', W / 'div_cnn', '--out', W / 'div_cat')
run('division_pair.py', '--nodes', W / 'nodes', '--edges', W / 'edges_learned', '--harm', W / 'edges_tf_harm', '--emb', W / 'embeddings', '--div-dir', W / 'div_cnn',
    '--cat', W / 'div_cat', '--test', a.test, '--models', M / 'division_pair', '--out', W / 'div_prior', '--jobs', max(2, a.jobs))

# 4. tracking ILP, fork verification, gap filling
pred = W / 'pred.csv'
run('ilp_solve.py', '--nodes', W / 'nodes', '--edges', W / 'edges_learned', '--div-dir', W / 'div_prior', '--alt-dir', W / 'edges_tf_harm', '--alt-min-divs', 0.2,
    '--div-costs', a.div_costs, '--sparse-div-costs', a.sparse_div_costs, '--route', a.route, '--out', pred, '--graph-out', W / 'graph', '--jobs', a.jobs)
run('fork_verify.py', '--graph-dir', W / 'graph', '--data-dir', a.test, '--models', f'{M / "fork_verifier" / "a"},{M / "fork_verifier" / "b"}', '--out-csv', pred,
    '--threshold', a.fork_thr, '--gate', 0.5)
run('gapfill.py', '--csv', pred, '--out', pred, '--max-gap', 2, '--jobs', max(2, a.jobs))

lines = pred.read_text().splitlines(); out = ['id,' + lines[0]] + [f'{i},{l}' for i, l in enumerate(lines[1:])]
Path(a.out).write_text('\n'.join(out) + '\n'); print(f'wrote {a.out}: {len(out) - 1} rows', flush=True)
