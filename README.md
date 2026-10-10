# Biohub – Cell Tracking During Development: 4th place solution

Code for our 4th-place solution (public 0.969 / private 0.962) in the Kaggle competition
[Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

**Pipeline.** A 3D Net detects nuclei (cell probability + a unit vector field pointing to the nucleus centre), a
LightGBM edge model scores candidate links, a learned division prior sets a per-node division cost inside a global
ILP, and a fork verifier, gap bridging and a track-support gate for sparse movies clean up the result.

## Repository layout

| folder | content |
|---|---|
| `inference/` | the complete inference pipeline (detection → linking → divisions → ILP → post-processing); see `inference/README.md` |
| `external_data/` | the external data we used, how much, and how it was downloaded, pseudo-labelled and merged; see `external_data/README.md` |
| `training/` | training code for every model (3D Nets, edge transformer, cell embedding, edge model, division models, fork verifier); `training/train_all.sh` runs it end to end; see `training/README.md` |
| `entry_points.md`, `SETTINGS.json`, `directory_structure.txt` | the commands from raw data to submission, the paths the scripts use, and the folder layout |

## Quick start (inference)

```bash
pip install -r requirements.txt
# put the models/ folder of the Kaggle dataset songqizhou/biohub-4th-place-artifacts into inference/models/
python inference/run_pipeline.py --test <dir with the *.zarr movies> --work work/run   # -> work/run/submission.csv
```

On Kaggle (2 × T4) the whole pipeline takes about 30 minutes on the 4 visible test movies. All commands, from the raw
competition data to a submission, are in `entry_points.md`.

## Environment and run time

| run | hardware | time |
|---|---|---|
| competition (all models of the final submission) | Ubuntu 24.04, 2 × AMD EPYC 9554 (128 cores), 1 TB RAM, 8 × NVIDIA L40S (48 GB); 3D Net-64 on one NVIDIA H100 | — |
| from-scratch retrain with this repository (`GPUS="0 1 2 3" bash training/train_all.sh`) | Linux, 4 × NVIDIA H100 | about 15 h |
| inference (the Kaggle notebook) | Kaggle, 2 × T4 | about 30 min for the 4 visible test movies |

- **Software.** Python 3.12, PyTorch 2.13 with the CUDA 12.6 wheels (NVIDIA driver for CUDA ≥ 12.6), the versions in
  `requirements.txt`. The 3D Nets train with bf16 autocast (batch 4); GPUs with less than 48 GB were not tested.
- **Disk.** Competition training data 81 GB, external windows 39 GB, checkpoints about 7 GB, plus intermediate files in
  `work/`.
- **Network (training only).** The edge transformer step clones the official baseline code from GitHub, and the 3D Net
  step downloads the 102 external windows from the public Ultrack bucket (about 115 GB of reads). Inference runs
  offline.
- **Kaggle image.** The notebook runs on the Kaggle image pinned in its metadata,
  `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.
- Run time per training step: `training/README.md`, "Run time".
- **Side effects.** Training writes `data/` (training sets, external windows), `work/`, `runs/` and `models/` at the
  repository root, overwriting earlier outputs there, clones the official baseline into `baseline/` and installs it into
  the active Python environment (`pip install -e`). Inference writes only to its `--work` directory.
- **Key assumptions.** All commands run from the repository root; the competition training movies (`*.zarr` + `*.geff`)
  are in `data/train`, test movies (`*.zarr`) in the directory given to `--test`, and the trained models in
  `inference/models/` in the layout of the Kaggle dataset; a Linux machine with an NVIDIA driver for CUDA ≥ 12.6.

`tracksdata` is used through its ILP interface. The Kaggle notebook installs it, together with `zarr`, SCIP and a recent
`polars`, from the public `biohub-tracking-support-pack` wheels (a development build, 0.1.0rc6.dev3); locally we used
0.1.0rc9 (`requirements.txt`), which gives byte-identical output.

## External data in one paragraph

One external embryo, the public Ultrack `zebrafish_embryo` light-sheet volume (no labels): **102 windows** of
64 × 256 × 256 voxels × 100 frames (**10,200 frames**, 39 GB), pseudo-labelled by our ground-truth-only 3D Net
(**2.02 M** pseudo-label nodes). It is used by one model only, the pseudo-label 3D Net that refines the nodes, where
it makes up 34% of the training windows and 30% of the pseudo-label nodes. All other models use the 199 competition
training movies only. Details in `external_data/README.md`. The pseudo-labels of that model (competition movies and
external windows, 6.8 M nodes) are released as the Kaggle dataset `songqizhou/biohub-4th-place-pseudo-labels`, and
training uses them by default: regenerated ones come from retrained detectors that are never bit-identical, and the
pseudo-label 3D Net is sensitive to that (`training/README.md`, section 1).

## Reproducing the scores

The inference pipeline with the released weights reproduces our final submission. Retraining follows the same recipes
but not to the bit (GPU non-determinism), and single retrained models move the leaderboard score by a few thousandths:
in our late submissions with retrained models the scores were 0.959–0.971 public and 0.959–0.964 private. Details in
`training/README.md`, section 6.

## Hand labels

The fork verifier of the final submission is also trained on 409 hand labels: candidate divisions in 39 training movies,
proposed by our models and checked by eye (274 divisions, 135 rejections). They and the verifier's other training events
are in `training/fork_verifier/`. A verifier trained from the official annotations only scores about 0.0017 lower on the
public and slightly higher on the private leaderboard, with its own drop threshold (0.40 instead of 0.25). Details in
`training/README.md`, section 5.

## Acknowledgements and licences

Our code is released under the MIT License (`LICENSE`). The parts derived from the official baseline keep its
BSD 3-Clause License (see below).

- The 3D Net backbone is adapted from a dual-encoder 3D U-Net with temporal attention from the literature; the unit
  vector field output and the advection decoding are in the spirit of Cellpose.
- `inference/edge_transformer.py` re-implements, for inference, the linker architecture of the competition's official
  baseline code (tracking-cellmot, BSD 3-Clause License, Copyright (c) 2026 Thibaut Goldsborough); see
  `inference/LICENSE-baseline.txt`. `training/edge_transformer/baseline.patch` is a patch to that code.
- The ILP uses [tracksdata](https://github.com/royerlab/tracksdata) with the SCIP solver.
- External imaging data: Royer Lab, CZ Biohub (Ultrack `zebrafish_embryo`).
