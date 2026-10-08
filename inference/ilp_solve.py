#!/usr/bin/env python
"""Tracking ILP (tracksdata + SCIP) on the fixed node set.

  minimise  sum_links -p_ij * y_ij  + app * appear_i + dis * disappear_i + div_i * divide_i   (tracksdata sign convention)

Link weights are the learned link probabilities; only links with p above a threshold enter the graph. The division cost is
per node, div_i = base - gain * s_i, where s_i is the division prior, and the link threshold of a likely mother is relaxed
to max(thr - relax * s_i, 0). For links that are not their source's best link and whose source has s_i >= --alt-min-divs,
p is replaced by max(p, harmonic transformer probability), which rescues the weaker second daughter.
Afterwards: components with fewer than --L nodes and no division are dropped, and coordinates are smoothed along tracks
(motion-compensated line fit).

Division costs can differ per density band: movies marked sparse in --route use --sparse-div-costs.
Writes the submission-format CSV and, per movie, the solved graph (for the fork verifier).
usage: ilp_solve.py --nodes DIR --edges DIR --div-dir DIR --alt-dir DIR --out pred.csv --graph-out DIR
                    [--div-costs 6,16,0.5 --sparse-div-costs 5,16,0.5 --route route.json]
"""
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import CSV_HEADER, filter_short_tracks, stabilized_linefit, submission_rows


def solve_one(args):
    f, edges_dir, thr, app, dis, div_dir, costs, alt_dir, alt_min_divs, timeout = args
    import polars as pl
    import tracksdata as td
    divbase, divgain, div_relax = costs
    z = np.load(f); name = Path(f).stem; e = np.load(f'{edges_dir}/{name}.npz'); I, J, p = e['I'], e['J'], e['p'].astype(np.float64)
    t = z['t']; um = z['um'].astype(np.float64); N = len(t)
    div_s = None
    dp = Path(div_dir) / f'{name}.divscore.npz'
    if dp.exists():
        dd = np.load(dp); smap = {tuple(int(v) for v in k): float(s) for k, s in zip(dd['keys'], dd['score'])}
        vox = np.rint(z['vox']).astype(int); div_s = np.array([smap.get((int(t[k]), vox[k, 0], vox[k, 1], vox[k, 2]), 0.0) for k in range(N)])
    if alt_dir:                                       # second-daughter rescue
        e2 = np.load(f'{alt_dir}/{name}.npz'); alt = {(int(i), int(j)): float(q) for i, j, q in zip(e2['I'], e2['J'], e2['p'])}
        pa = np.array([alt.get((int(i), int(j)), 0.0) for i, j in zip(I, J)])
        top_row = np.zeros(len(I), bool); seen = set()
        for q in np.lexsort((-p, I)):
            if int(I[q]) not in seen:
                seen.add(int(I[q])); top_row[q] = True
        sel = ~top_row
        if alt_min_divs > 0 and div_s is not None:
            sel = sel & (div_s[I] >= alt_min_divs)
        p = np.where(sel, np.maximum(p, pa), p)
    g = td.graph.InMemoryGraph()
    for key in ('z', 'y', 'x', 'app_cost', 'dis_cost'):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    if div_s is not None:
        g.add_node_attr_key('div_cost', pl.Float64, float(divbase))
        ids = g.bulk_add_nodes([{'t': int(t[k]), 'z': float(um[k, 0]), 'y': float(um[k, 1]), 'x': float(um[k, 2]), 'div_cost': float(divbase - divgain * div_s[k]),
                                 'app_cost': float(app), 'dis_cost': float(dis)} for k in range(N)])
    else:
        ids = g.bulk_add_nodes([{'t': int(t[k]), 'z': float(um[k, 0]), 'y': float(um[k, 1]), 'x': float(um[k, 2]), 'app_cost': float(app), 'dis_cost': float(dis)} for k in range(N)])
    ids = np.asarray(ids)
    eff = np.full(len(I), thr) if div_s is None or div_relax <= 0 else np.maximum(thr - div_relax * div_s[I], 0.0)
    keep = p > eff
    g.add_edge_attr_key('edge_prob', pl.Float64, 0.0)
    if keep.any():
        g.bulk_add_edges([{'source_id': int(ids[i]), 'target_id': int(ids[j]), 'edge_prob': float(q)} for i, j, q in zip(I[keep], J[keep], p[keep])])
    solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr('edge_prob'), node_weight=0.0, appearance_weight=td.NodeAttr('app_cost'),
                                  disappearance_weight=td.NodeAttr('dis_cost'), division_weight=(td.NodeAttr('div_cost') if div_s is not None else 1.0),
                                  num_threads=1, timeout=timeout)
    sol = solver.solve(g)
    ed = sol.edge_attrs(attr_keys=[]).select(['source_id', 'target_id']).to_numpy(); id2k = {int(v): k for k, v in enumerate(ids)}
    return name, np.array([[id2k[int(a)], id2k[int(b)]] for a, b in ed], np.int64).reshape(-1, 2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--nodes', required=True); ap.add_argument('--edges', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--div-dir', required=True); ap.add_argument('--div-costs', default='6,16,0.5'); ap.add_argument('--sparse-div-costs', default=''); ap.add_argument('--route', default='')
    ap.add_argument('--alt-dir', default=''); ap.add_argument('--alt-min-divs', type=float, default=0.2)
    ap.add_argument('--thr', type=float, default=0.3); ap.add_argument('--app', type=float, default=0.5); ap.add_argument('--dis', type=float, default=3.2)
    ap.add_argument('--L', type=int, default=2); ap.add_argument('--linefit', type=int, default=3); ap.add_argument('--graph-out', required=True)
    ap.add_argument('--jobs', type=int, default=3); ap.add_argument('--timeout', type=float, default=600); a = ap.parse_args()
    dense = tuple(float(v) for v in a.div_costs.split(','))
    sparse_movies = set()
    if a.sparse_div_costs:
        sparse = tuple(float(v) for v in a.sparse_div_costs.split(','))
        sparse_movies = {m for m, r in json.load(open(a.route)).items() if r.get('sparse')}
    files = sorted(glob.glob(f'{a.nodes}/*.npz'))
    print(f'ILP: {len(files)} movies, division costs {a.div_costs} (dense) / {a.sparse_div_costs or a.div_costs} ({len(sparse_movies)} sparse)', flush=True)
    tasks = [(f, a.edges, a.thr, a.app, a.dis, a.div_dir, sparse if Path(f).stem in sparse_movies else dense, a.alt_dir, a.alt_min_divs, a.timeout) for f in files]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(solve_one, tasks))
    lines = [CSV_HEADER]; Path(a.graph_out).mkdir(parents=True, exist_ok=True)
    for f in files:
        z = np.load(f); name = Path(f).stem; nodes = np.concatenate([z['t'][:, None].astype(np.float32), z['um'].astype(np.float32)], 1)
        E0 = res[name]; nodes, E, keep = filter_short_tracks(nodes, E0, a.L)
        ee = np.load(f'{a.edges}/{name}.npz'); n0 = len(z['t']); keys = ee['I'].astype(np.int64) * n0 + ee['J'].astype(np.int64); order = np.argsort(keys)
        loc = np.clip(np.searchsorted(keys[order], E0[:, 0] * n0 + E0[:, 1]), 0, len(order) - 1); hit = keys[order[loc]] == E0[:, 0] * n0 + E0[:, 1]
        conf = np.where(hit, ee['p'][order[loc]], 0.0)[keep[E0[:, 0]] & keep[E0[:, 1]]]
        nodes, velocity = stabilized_linefit(nodes, E, conf, a.linefit, 0.8)
        np.savez_compressed(Path(a.graph_out) / f'{name}.npz', nodes=nodes.astype(np.float32), E=E, p=conf.astype(np.float32), velocity=velocity.astype(np.float32),
                            node_index=np.flatnonzero(keep))
        lines += submission_rows(name, nodes, E)
    Path(a.out).write_text('\n'.join(lines) + '\n'); print('wrote', a.out, flush=True)
