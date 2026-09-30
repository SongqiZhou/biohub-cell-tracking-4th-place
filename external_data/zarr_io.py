"""Small helpers for reading remote OME-Zarr volumes and writing JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import zarr

COMPETITION_SCALE = (1.625, 0.40625, 0.40625)          # competition voxel size (z, y, x) in um


def open_group(url: str) -> zarr.Group:
    """Open an OME-Zarr group from an http(s) URL or a local path. ZARR_HTTP_TIMEOUT=<s> sets a per-read timeout."""
    if url.startswith("http://") or url.startswith("https://"):
        opts = {}
        if os.environ.get("ZARR_HTTP_TIMEOUT"):
            import aiohttp
            opts = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=None, sock_connect=60,
                                                                      sock_read=float(os.environ["ZARR_HTTP_TIMEOUT"]))}}
        store = zarr.storage.FsspecStore.from_url(url, read_only=True, storage_options=opts or None)
        return zarr.open_group(store, mode="r")
    return zarr.open_group(url, mode="r")


def multiscale_info(grp: zarr.Group) -> list[dict]:
    """One dict per pyramid level: path, scale (z, y, x) in um, shape, chunks, dtype."""
    ms = grp.attrs["multiscales"][0]
    out = []
    for ds in ms["datasets"]:
        scale = ds["coordinateTransformations"][0]["scale"]
        arr = grp[ds["path"]]
        out.append({"path": ds["path"], "scale_zyx_um": [float(s) for s in scale[-3:]], "shape": list(arr.shape),
                    "chunks": list(arr.chunks), "dtype": str(arr.dtype)})
    return out


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
