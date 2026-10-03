#!/usr/bin/env python
"""Fit the two fork verifiers (inference/fork_verify.py) on the rows of fork_verifier_rows.py or fork_rows.py.

Verifier a is trained on the movies of half 1 and verifier b on half 0 (training/fold_movies.py --half), each without every
fifth movie of its half. CatBoost, 400 trees of depth 4, learning rate 0.03, L2 8, on the fork and event rows of the
training movies. Its raw score is calibrated by a logistic regression (C = 0.5) on out-of-fold scores of the fork rows:
three inner models, each trained without a third of the training movies, score the forks of that third.
Writes <out>/{a,b}/model.cbm and calibration.json.

--all-movies trains both verifiers on all movies instead (a and b differ only in the random seed), with the calibration
from three inner models on thirds of the movies. Without hand labels this is the better choice (use drop threshold 0.40).

    python training/train_fork_verifier.py --rows work/fork/rows --out models/fork_verifier
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

PARAMS = dict(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=8, random_seed=20260920, verbose=False, thread_count=3, allow_writing_files=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--rows', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--all-movies', action='store_true'); a = ap.parse_args()
    names = sorted(Path(f).stem for f in glob.glob(f'{a.rows}/*.npz'))
    rows = {}
    for n in names:
        z = np.load(Path(a.rows) / f'{n}.npz'); X, y, kind = z['X'], z['y'], z['kind']; m = y >= 0
        rows[n] = ((X[m & (kind == 0)], y[m & (kind == 0)]), (X[m & (kind == 1)], y[m & (kind == 1)]))

    def fit(movies, seed=PARAMS['random_seed']):
        xs, ys = [], []
        for n in movies:
            for x, y in rows[n]:
                xs.append(x); ys.append(y)
        return CatBoostClassifier(**{**PARAMS, 'random_seed': seed}).fit(np.concatenate(xs), np.concatenate(ys))

    for k_, (tag, half) in enumerate((('a', 1), ('b', 0))):
        seed = PARAMS['random_seed'] + (k_ if a.all_movies else 0)
        if a.all_movies:
            fit_movies = names
        else:
            movies = names[half::2]; fit_movies = [n for n in movies if n not in set(movies[::5])]
        logits, labels = [], []
        for k in range(3):
            val = fit_movies[k::3]; model = fit([n for n in fit_movies if n not in set(val)], seed)
            for n in val:
                x, y = rows[n][0]
                if len(y):
                    logits.append(model.predict(x, prediction_type='RawFormulaVal')); labels.append(y)
        cal = LogisticRegression(C=.5).fit(np.concatenate(logits)[:, None], np.concatenate(labels))
        out = Path(a.out) / tag; out.mkdir(parents=True, exist_ok=True)
        fit(fit_movies, seed).save_model(str(out / 'model.cbm'))
        json.dump(dict(coef=float(cal.coef_[0, 0]), intercept=float(cal.intercept_[0])), open(out / 'calibration.json', 'w'), indent=1)
        n_fork = sum(len(rows[n][0][1]) for n in fit_movies); n_event = sum(len(rows[n][1][1]) for n in fit_movies)
        print(f'verifier {tag}: {len(fit_movies)} movies, {n_fork} forks + {n_event} events; calibration on {sum(len(l) for l in labels)} '
              f'out-of-fold forks: coef {cal.coef_[0, 0]:.3f}, intercept {cal.intercept_[0]:.3f} -> {out}', flush=True)


if __name__ == '__main__':
    main()
