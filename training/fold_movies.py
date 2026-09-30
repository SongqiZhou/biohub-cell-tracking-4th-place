#!/usr/bin/env python
"""Print the held-out movies of a fold as a comma-separated list.

    python training/fold_movies.py 0                                          # all held-out movies of fold 0
    python training/fold_movies.py 0 --dog work/dog_train.json --sparse       # only those with DoG spacing >= 19.0 um
    python training/fold_movies.py 0 --dog work/dog_train.json --dense        # only those below 19.0 um
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('fold', type=int)
ap.add_argument('--dog', default='', help='dog_route.py output for the training movies')
ap.add_argument('--split-um', type=float, default=19.0)
g = ap.add_mutually_exclusive_group()
g.add_argument('--sparse', action='store_true')
g.add_argument('--dense', action='store_true')
a = ap.parse_args()
movies = json.loads((Path(__file__).resolve().parent / 'folds.json').read_text())[f'fold{a.fold}']
if a.sparse or a.dense:
    sp = {m: v['dog_sp'] for m, v in json.loads(Path(a.dog).read_text()).items()}
    movies = [m for m in movies if (sp[m] >= a.split_um) == a.sparse]
print(','.join(movies))
