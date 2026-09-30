"""Per-movie nucleus spacing from the DoG tracker (independent of the learned detectors).

Writes {movie: {"dog_sp": spacing_um, "dog_nodes": n}} to --out. Movies with spacing >= 19.7 um are treated as sparse.
usage: dog_route.py --test <dir with *.zarr> --out dog_route.json [--procs N]"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--test', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--procs', type=int, default=0)
a = ap.parse_args()


def one(name):
    import dog_tracker as dt
    zp = os.path.join(a.test, name + '.zarr')
    g = dt.track_movie(zp, dt.Config(**dt.ROUTER_CONFIG))
    return name, float(dt.movie_spacing_um(zp, g.n_nodes)), int(g.n_nodes)


if __name__ == '__main__':
    t0 = time.time()
    names = sorted(d[:-5] for d in os.listdir(a.test) if d.endswith('.zarr'))
    n = a.procs or max(1, min(len(names), os.cpu_count() or 2))
    with Pool(n) as p:
        res = p.map(one, names)
    json.dump({m: {'dog_sp': sp, 'dog_nodes': k} for m, sp, k in res}, open(a.out + '.tmp', 'w'))
    os.replace(a.out + '.tmp', a.out)
    print(f'dog_route: {len(res)} movies in {time.time() - t0:.0f}s with {n} procs -> {a.out}', flush=True)
