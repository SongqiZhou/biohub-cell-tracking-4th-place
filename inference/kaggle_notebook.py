# %% [markdown]
# # Biohub cell tracking: 4th place solution (inference)
#
# This notebook reproduces our final submission (public 0.969 / private 0.962). All code and weights are in the dataset
# **biohub-4th-place-artifacts**; the Python packages that the Kaggle image lacks (zarr, tracksdata, SCIP, ...) are
# installed offline from the public **biohub-tracking-support-pack** wheels.
#
# Pipeline:
# 1. **Density router.** A classical DoG blob tracker estimates each movie's nucleus spacing. Movies with spacing
#    >= 19.7 um are treated as sparse (runs on CPU, in the background).
# 2. **Detection with the 3D Net.** The net predicts a cell probability and a unit vector field pointing to the nucleus
#    centre; voxels above the seed threshold are advected along the field and clustered. A second 3D Net trained with
#    pseudo-labels refines the nodes and is trusted only where the first one agrees. Sparse movies use a stricter seed
#    threshold (0.99 instead of 0.97).
# 3. **Linking.** Candidate links get geometric features, transformer link probabilities and the similarity of a
#    contrastive cell embedding; a LightGBM model turns them into one link probability.
# 4. **Division prior.** A division CNN, a CatBoost node model and a mother-daughter pair model give every node a
#    division score s; in the ILP each node's division cost is 6 - 16 s (5 - 16 s in sparse movies).
# 5. **ILP** (tracksdata + SCIP), motion-compensated smoothing, fork verification and gap filling.
# 6. **Track-support gate** (sparse movies only): track segments whose nodes a low-resolution 3D Net (3D Net-64) and a
#    sensitive DoG detector do not confirm often enough are removed.

# %%
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# final configuration
THR_DENSE, THR_SPARSE = 0.97, 0.99        # 3D Net seed threshold (sparse = DoG spacing >= SPARSE_UM)
SPARSE_UM = 19.7
NMS_ADAPT = "4,7,10"                      # per-movie NMS radius: r_dense, r_sparse, nearest-neighbour cut (um)
REFINE_R, REFINE_SUPPORT = 6, 0.5         # pseudo-label 3D Net refinement
NET64_THR = 0.99
DIV_COSTS, SPARSE_DIV_COSTS = "6,16,0.5", "5,16,0.5"      # ILP division cost base, gain, link-threshold relax
FORK_THR = 0.25
GATE_SPEC = "19.7:25:or:0.4:4:5;25:99:and:0.2:5:5"          # track gate bands lo:hi:mode:theta:r_dog:r_net64

INPUT = Path(os.environ.get("BIOHUB_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("BIOHUB_WORK", "/kaggle/working"))
KEEP = os.environ.get("BIOHUB_KEEP", "0") == "1"           # keep intermediate files


def shallow(pattern):
    """glob at depths 1-3 below INPUT (a recursive walk of the competition data is slow)"""
    return sorted({p for d in ("*", "*/*", "*/*/*") for p in glob.glob(str(INPUT / d / pattern))})


CODE = Path(os.environ.get("BIOHUB_CODE") or Path(shallow("net3d.py")[0]).parent)
TEST = os.environ.get("BIOHUB_TEST") or next(p for p in shallow("test") if glob.glob(p + "/*.zarr"))
MOVIES = sorted(p.name[:-5] for p in Path(TEST).iterdir() if p.name.endswith(".zarr"))
print(f"code + models: {CODE}\ntest movies:   {TEST} ({len(MOVIES)})", flush=True)
T0 = time.time()

# %% [markdown]
# ## Offline packages

# %%
if os.environ.get("BIOHUB_SKIP_PIP", "0") != "1":
    wheels = shallow("wheels")
    force = ("polars>=1.36", "polars-runtime-32")          # the image's polars is too old for tracksdata
    pkgs = ("numcodecs", "donfig", "typing_extensions", "packaging", "google-crc32c", "zarr", "pyarrow", "bidict", "psygnal", "rich",
            "networkx", "pydantic", "annotated-types", "pydantic-core", "typing-inspection", "geff-spec", "geff", "rustworkx", "sqlalchemy",
            "click", "cloudpickle", "fsspec", "locket", "partd", "pyyaml", "toolz", "dask", "ndindex", "msgpack", "numexpr", "blosc2",
            "imagecodecs", "ilpy", "pyscipopt", "tracksdata")
    for pkg in force + pkgs:
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q"] + (["--force-reinstall"] if pkg in force else [])
        cmd += sum((["--find-links", w] for w in wheels), []) + [pkg]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  pip {pkg}: {'ok' if r.returncode == 0 else 'skipped'}", flush=True)
    if subprocess.run([sys.executable, "-c", "import catboost"], capture_output=True).returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", "--find-links", str(CODE / "wheels"), "catboost"], check=True)
r = subprocess.run([sys.executable, "-c", "import zarr, polars, tracksdata, pyscipopt, ilpy, lightgbm, catboost, blosc2; assert hasattr(polars, 'Float16'); print('packages OK')"],
                   capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[-1500:], flush=True)
assert r.returncode == 0, "package check failed"

# %% [markdown]
# ## 1-2. Density router and 3D Net detection
#
# The DoG router and the sensitive DoG detector (support for the track gate) run on the CPU while the GPUs detect.

# %%
import torch  # noqa: E402

NGPU = max(1, torch.cuda.device_count())
VISIBLE = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]


def gpu(i):
    return VISIBLE[i % NGPU] if VISIBLE else str(i % NGPU)


def py(script, *args):
    return [sys.executable, str(CODE / script), *[str(a) for a in args]]


def launch(cmd, gpu_index=None, tag=""):
    """start a child process; its output is forwarded line by line (so it also shows up in the notebook)"""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu(gpu_index)
    print(">> " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump():
        for line in p.stdout:
            print(tag + line, end="", flush=True)
    p.pump = threading.Thread(target=pump, daemon=True)
    p.pump.start()
    return p


def finish(*procs):
    for p in procs:
        rc = p.wait(); p.pump.join()
        assert rc == 0, f"failed ({rc}): {' '.join(p.args)}"


def run_parallel(cmds):
    """one command per GPU"""
    finish(*[launch(cmd, i, f"[gpu {i % NGPU}] ") for i, cmd in enumerate(cmds)])


dog_route = launch(py("dog_route.py", "--test", TEST, "--out", WORK / "dog_route.json", "--procs", 2), tag="[dog] ")
dog_nodes = launch(py("dog_nodes.py", "--test", TEST, "--route", WORK / "dog_route.json", "--out-dir", WORK / "dog_nodes",
                      "--min-sp", SPARSE_UM, "--rel", 0.01, "--procs", 2), tag="[dog] ")

NODES = WORK / "nodes"
det = ("--data-dir", TEST, "--out", NODES, "--nms-adapt", NMS_ADAPT, "--vec", CODE / "models" / "net3d_128.pt",
       "--refine-with", CODE / "models" / "net3d_128_pl.pt", "--refine-r", REFINE_R, "--support", REFINE_SUPPORT, "--gpu", 0)
run_parallel([py("detect.py", *det, "--thr", THR_DENSE, "--shard", f"{i}/{NGPU}") for i in range(NGPU)])

finish(dog_route)
spacing = {m: v["dog_sp"] for m, v in json.load(open(WORK / "dog_route.json")).items()}
sparse = [m for m in MOVIES if spacing[m] >= SPARSE_UM]
json.dump({m: {"dog_sp": spacing[m], "sparse": m in sparse} for m in MOVIES}, open(WORK / "route.json", "w"))
for m in MOVIES:
    print(f"  {m}: DoG spacing {spacing[m]:.2f} um -> {'sparse' if m in sparse else 'dense'}", flush=True)
if sparse:        # re-detect the sparse movies with the stricter seed threshold (their node files are replaced)
    run_parallel([py("detect.py", *det, "--thr", THR_SPARSE, "--movies", ",".join(sparse[i::NGPU])) for i in range(NGPU) if sparse[i::NGPU]])
    # 3D Net-64 support detections for the track gate
    run_parallel([py("detect.py", "--data-dir", TEST, "--out", WORK / "net64_nodes", "--nms-adapt", NMS_ADAPT, "--vec", CODE / "models" / "net3d_64.pt",
                     "--thr", NET64_THR, "--gpu", 0, "--movies", ",".join(sparse[i::NGPU])) for i in range(NGPU) if sparse[i::NGPU]])
print(f"detection done ({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 3-5. Linking, division prior, ILP, fork verification, gap filling

# %%
finish(launch(py("downstream.py", "--nodes", NODES, "--test", TEST, "--out", WORK / "submission.csv", "--work", WORK / "ds", "--route", WORK / "route.json",
                 "--gpus", ",".join(str(i) for i in range(NGPU)), "--div-costs", DIV_COSTS, "--sparse-div-costs", SPARSE_DIV_COSTS, "--fork-thr", FORK_THR)))
print(f"downstream done ({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 6. Track-support gate (sparse movies)

# %%
finish(dog_nodes)
if sparse:
    finish(launch(py("track_gate.py", "--sub", WORK / "submission.csv", "--route", WORK / "dog_route.json", "--net64-dir", WORK / "net64_nodes",
                     "--dog-dir", WORK / "dog_nodes", "--spec", GATE_SPEC)))

import pandas as pd  # noqa: E402

sub = pd.read_csv(WORK / "submission.csv")
assert sorted(sub["dataset"].unique()) == MOVIES
for m, g in sub.groupby("dataset"):
    print(f"  {m}: {(g.row_type == 'node').sum()} nodes, {(g.row_type == 'edge').sum()} edges", flush=True)
print(f"submission.csv: {len(sub)} rows ({time.time() - T0:.0f}s)", flush=True)
if not KEEP:
    for d in ("ds", "nodes", "net64_nodes", "dog_nodes"):
        shutil.rmtree(WORK / d, ignore_errors=True)
    for f in ("dog_route.json", "route.json"):
        (WORK / f).unlink(missing_ok=True)
