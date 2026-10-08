#!/usr/bin/env python
"""Extract 104 um windows from the public Ultrack `zebrafish_embryo` light-sheet volume in the competition format.

* No resampling: level 0 of this volume already has the competition voxel size (1.625, 0.40625, 0.40625) um, so a
  104 um window is a plain (100, 64, 256, 256) crop (100 frames).
* No labels: the volume ships without tracks, so every window is image-only; labels come later from our own detector
  (make_pseudo_labels.py).
* The output matches the competition layout (zarr v3, one chunk per frame, zstd + bitshuffle, OME multiscales 0.5,
  the same image-statistics quantile keys), so the windows load with the same readers as the competition movies.

Level 0 stores a whole YX plane per chunk, so the download cost is per (time span, z-chunk), not per window. The window
list (ultrack_windows.json) is therefore confined to the first z-chunk and three time spans.

    python external_data/extract_ultrack_windows.py --out-dir data/ultrack_windows --workers 6
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from zarr_io import COMPETITION_SCALE, multiscale_info, open_group, save_json

URL = "https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr"
BOX_UM = 104.0
TARGET_SHAPE = (64, 256, 256)
QUANTILE_KEYS = (0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 1.0)


def create_zarr(path: Path, n_t: int):
    path = Path(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    grp = zarr.open_group(path, mode="w", zarr_format=3)
    from zarr.codecs import BloscCodec, BloscShuffle

    arr = grp.create_array(
        "0", shape=(n_t, *TARGET_SHAPE), chunks=(1, *TARGET_SHAPE), dtype="uint16",
        compressors=[BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.bitshuffle, typesize=2)],
    )
    return grp, arr


def finalize(grp, sample: np.ndarray, extra: dict) -> None:
    quant = {str(q): float(np.quantile(sample, q)) for q in QUANTILE_KEYS}
    grp.attrs.update({
        "multiscales": [{
            "version": "0.5", "name": "0",
            "axes": [{"name": "T", "type": "time", "unit": "second"},
                     {"name": "Z", "type": "space", "unit": "micrometer"},
                     {"name": "Y", "type": "space", "unit": "micrometer"},
                     {"name": "X", "type": "space", "unit": "micrometer"}],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, *map(float, COMPETITION_SCALE)]}]}],
        }],
        "image_statistics": {"quantiles": quant},
        "ultrack_window": extra,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--windows", default=str(Path(__file__).resolve().parent / "ultrack_windows.json"))
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wins = json.loads(Path(args.windows).read_text())

    grp = open_group(URL)
    lv = multiscale_info(grp)[0]
    arr = grp[lv["path"]]
    scale = np.array(lv["scale_zyx_um"], dtype=float)
    shp = np.array(arr.shape[-3:])
    assert np.allclose(scale, COMPETITION_SCALE, rtol=1e-3), f"expected competition grid, got {scale}"

    groups = defaultdict(list)
    for w in wins:
        groups[w["t0"]].append(w)
    print(f"extracting {len(wins)} windows in {len(groups)} time spans -> {out_dir}")

    for t0, gw in sorted(groups.items()):
        todo = [w for w in gw if args.no_skip or not (out_dir / f"{w['id']}.json").exists()]
        if not todo:
            print(f"  [t0={t0}] all {len(gw)} done")
            continue
        t_start = time.time()
        plans = {}
        for w in todo:
            o = np.array(w["origin_um"], float)
            i0 = np.clip(np.floor(o / scale).astype(int), 0, shp - np.array(TARGET_SHAPE))
            plans[w["id"]] = {"i0": i0, "i1": i0 + np.array(TARGET_SHAPE)}
        u0 = np.min([p["i0"] for p in plans.values()], axis=0)
        u1 = np.max([p["i1"] for p in plans.values()], axis=0)
        frames_t = [t for t in range(t0, t0 + 100) if t < arr.shape[0]]
        print(f"  [t0={t0}] {len(todo)} windows, {len(frames_t)} frames, "
              f"union {(u1-u0).tolist()} = {np.prod(u1-u0)*2/1e6:.0f} MB/frame")

        grps, arrs, samples = {}, {}, {}
        for w in todo:
            g_, a_ = create_zarr(out_dir / f"{w['id']}.zarr", len(frames_t))
            grps[w["id"]], arrs[w["id"]], samples[w["id"]] = g_, a_, []
        lock = __import__("threading").Lock()

        def _read(t, tries=6):
            """Read one frame block, retrying with backoff (remote reads occasionally time out)."""
            for k in range(tries):
                try:
                    a = arr if k < tries - 2 else open_group(URL)[lv["path"]]
                    return np.asarray(a[t, 0, u0[0]:u1[0], u0[1]:u1[1], u0[2]:u1[2]])
                except Exception as e:
                    if k == tries - 1:
                        raise
                    print(f"    [t={t}] read failed ({type(e).__name__}), "
                          f"retry {k+1}/{tries-1} in {2**k}s", flush=True)
                    time.sleep(2 ** k)

        def _one(j_t):
            j, t = j_t
            blk = _read(t)
            for w in todo:
                p = plans[w["id"]]
                a, b = p["i0"] - u0, p["i1"] - u0
                vol = blk[a[0]:b[0], a[1]:b[1], a[2]:b[2]].astype(np.uint16)
                arrs[w["id"]][j] = vol
                with lock:
                    samples[w["id"]].append(vol.ravel()[::200].copy())
            return j

        with ThreadPoolExecutor(args.workers) as ex:
            for _ in ex.map(_one, list(enumerate(frames_t))):
                pass

        for w in todo:
            finalize(grps[w["id"]], np.concatenate(samples[w["id"]]), w)
            info = {**w, "source": "ultrack/zebrafish_embryo", "source_level": lv["path"],
                    "source_scale_zyx_um": scale.tolist(),
                    "source_index_lo": plans[w["id"]]["i0"].tolist(),
                    "crop_shape": [len(frames_t), *TARGET_SHAPE], "frames_t": frames_t,
                    "target_scale_zyx_um": list(COMPETITION_SCALE),
                    "n_nodes": 0, "n_edges": 0, "n_divisions": 0, "labelled": False}
            save_json(info, out_dir / f"{w['id']}.json")
            del grps[w["id"]], arrs[w["id"]], samples[w["id"]]
        print(f"  [t0={t0}] done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
