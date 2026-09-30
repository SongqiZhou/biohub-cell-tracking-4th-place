#!/usr/bin/env python
"""Fit the mother-daughter pair model of the division prior (inference/division_pair.py) on the out-of-fold node set.

Training triplets (mother, daughter a, daughter b) are built exactly as at inference, but only for two kinds of mothers:
  * nodes matched to an annotated mother with two annotated children (one-to-one matching per frame within 7 um, maximising
    the sum of 1 / (1 + distance)): a triplet is positive when both daughters lie within 5 um of the two annotated
    daughters; its other triplets are left out, as the annotation cannot tell whether they are wrong;
  * 200 random other nodes per movie (not in the first or last frame): all their triplets are negatives.
Features as at inference, from out-of-fold inputs:
  * 53 graph features on the edge model's links (p >= 0.02) with the plain division CNN's scores (--div-dir),
  * 12 embedding features on the links of the edge model without the embedding column (p >= 0.02) (--edges-noemb),
  * 22 link features from the edge model's links, the transformer's harmonic probability and the embeddings.
LightGBM (500 rounds, lr 0.03, 15 leaves) and CatBoost (600 trees, depth 6) are fitted on all rows.

    python training/train_division_pair.py --nodes work/link/nodes --gt data/train --edges work/link/edges_oof \
        --edges-noemb work/link/edges_oof_noemb --harm work/link/edges_tf_harm --emb work/link/embeddings \
        --div-dir work/link/div_cnn_plain --test data/train --out models/division_pair
"""
import argparse
import collections
import glob
import subprocess
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'inference'))
from division_pair import candidate_triplets, embedding_features, graph_features, link_features, tables  # noqa: E402
from link_labels import annotated_graph  # noqa: E402

A = None


def negative_seed(name):
    return zlib.crc32(name.encode())


def match_annotated(t, um, gt_t, gt_um, r=7.0):
    """annotated node -> detected node (or -1), one-to-one per frame, maximising sum 1 / (1 + d) over pairs with d <= r"""
    match = -np.ones(len(gt_t), int)
    for tt in np.unique(gt_t):
        gs = np.where(gt_t == tt)[0]; ks = np.where(t == tt)[0]
        if len(ks) == 0:
            continue
        D = np.linalg.norm(gt_um[gs][:, None, :] - um[ks][None, :, :], axis=2); W = np.where(D <= r, 1.0 / (1.0 + D), 0.0)
        ri, ci = linear_sum_assignment(-W)
        for i, j in zip(ri, ci):
            if W[i, j] > 0:
                match[gs[i]] = ks[j]
    return match


def movie_rows(name):
    z = np.load(Path(A.nodes) / f'{name}.npz'); t = z['t'].astype(int); um = z['um'].astype(np.float64); vox = z['vox']; N = len(t); T = int(t.max()) + 1
    gt_t, gt_um, gt_e = annotated_graph(Path(A.gt) / f'{name}.geff')
    match = match_annotated(t, um, gt_t, gt_um)
    children = collections.defaultdict(list)
    for p, c in gt_e:
        children[int(p)].append(int(c))
    mothers_gt = {int(match[p]): (gt_um[cs[0]], gt_um[cs[1]]) for p, cs in children.items() if len(cs) == 2 and match[p] >= 0}
    pool = np.where((t < T - 1) & (t >= 1))[0]; pool = pool[~np.isin(pool, list(mothers_gt))]
    neg = np.random.default_rng(negative_seed(name)).choice(pool, min(A.neg, len(pool)), replace=False)
    mothers = np.array(sorted(set(mothers_gt) | set(neg.tolist())), int)
    tab, tabn = tables(str(Path(A.test) / f'{name}.zarr'), t, um)
    mothers, R, X = candidate_triplets(t, um, tab.astype(np.float64), tabn.astype(np.float64), mothers)
    if len(R) == 0:
        return name, np.zeros((0,), np.int8), None
    e = np.load(Path(A.edges) / f'{name}.npz'); I, J, P = e['I'].astype(np.int64), e['J'].astype(np.int64), e['p'].astype(np.float64)
    e2 = np.load(Path(A.edges_noemb) / f'{name}.npz'); keep = e2['p'] >= 0.02
    h = np.load(Path(A.harm) / f'{name}.npz'); HK = h['I'].astype(np.int64) * N + h['J'].astype(np.int64); ho = np.argsort(HK); HK = HK[ho]; HP = h['p'].astype(np.float64)[ho]
    E = np.load(Path(A.emb) / f'{name}.npy').astype(np.float64)
    XP, _ = graph_features(name, t, vox, I, J, P, A.div_dir)
    XE = embedding_features(N, e2['I'][keep].astype(np.int64), e2['J'][keep].astype(np.int64), e2['p'][keep].astype(np.float64), E.astype(np.float32))
    u = mothers[R[:, 0]]
    F = np.concatenate([X, XP[u], XE[u], link_features(N, I, J, P, HK, HP, E, u, R[:, 1], R[:, 2])], 1)
    y = np.zeros(len(R), np.int8)
    for k, (mr, v, w) in enumerate(R):
        m = int(mothers[mr])
        if m in mothers_gt:
            g1, g2 = mothers_gt[m]; pv, pw = um[v], um[w]
            ok = (np.linalg.norm(pv - g1) <= 5 and np.linalg.norm(pw - g2) <= 5) or (np.linalg.norm(pv - g2) <= 5 and np.linalg.norm(pw - g1) <= 5)
            y[k] = 1 if ok else -1
    return name, y, F


def main():
    global A
    ap = argparse.ArgumentParser()
    for k in ('nodes', 'gt', 'edges', 'edges-noemb', 'harm', 'emb', 'div-dir', 'test', 'out'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--neg', type=int, default=200); ap.add_argument('--jobs', type=int, default=24)
    ap.add_argument('--half', type=int, default=-1, choices=[-1, 0, 1], help='leave out this half of the movies (training/fold_movies.py --half)')
    A = ap.parse_args()
    names = sorted(Path(f).stem for f in glob.glob(f'{A.nodes}/*.npz'))
    if A.half >= 0:
        held = set(subprocess.run([sys.executable, str(HERE / 'fold_movies.py'), '--half', str(A.half)], capture_output=True, text=True, check=True).stdout.strip().split(','))
        names = [n for n in names if n not in held]
    with ProcessPoolExecutor(A.jobs) as ex:
        res = [(y, F) for _, y, F in ex.map(movie_rows, names) if F is not None]
    y = np.concatenate([r[0] for r in res]); X = np.nan_to_num(np.concatenate([r[1] for r in res]).astype(np.float64), nan=-1.0)
    print(f'triplets {len(y)}: positive {int((y == 1).sum())}, left out {int((y == -1).sum())}, negative {int((y == 0).sum())}; {X.shape[1]} features', flush=True)
    use = y >= 0; X, y = X[use], y[use].astype(int)

    import lightgbm as lgb
    from catboost import CatBoostClassifier
    prm = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=15, feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1,
               lambda_l2=10.0, scale_pos_weight=float((y == 0).sum() / max(y.sum(), 1)) ** 0.5, verbose=-1, num_threads=12, seed=0)
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    lgb.train(prm, lgb.Dataset(X, y), 500).save_model(str(out / 'lgb.txt'))
    CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, auto_class_weights='SqrtBalanced', random_seed=0, verbose=0, thread_count=12,
                       allow_writing_files=False).fit(X, y).save_model(str(out / 'cat.cbm'))
    print(f'division pair model -> {out}', flush=True)


if __name__ == '__main__':
    main()
