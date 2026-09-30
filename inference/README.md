# Inference

Everything needed to produce a submission from the trained models. The same code runs in the Kaggle notebook
(`kaggle_notebook.py`) and locally (`run_pipeline.py`).

**Weights.** Download the Kaggle dataset `songqizhou/biohub-4th-place-artifacts` and put its `models/` folder here
(`inference/models/`). Then:

```bash
python inference/run_pipeline.py --test <dir with the *.zarr movies> --work work/run
```

## Pipeline and files

| step | script | what it does |
|---|---|---|
| density router | `dog_tracker.py`, `dog_route.py` | classical DoG blob tracker; nucleus spacing per movie, sparse if >= 19.7 um |
| detection | `net3d.py`, `detect.py` | 3D Net: cell probability + unit vector field to the nucleus centre, voxel advection + clustering, adaptive NMS; refinement with the pseudo-label model |
| node table | `nodes_table.py` | detections -> per-movie node table |
| candidate links | `candidates.py`, `edge_features.py` | links within 10 um + k nearest; 26 geometric / motion / competition features |
| link evidence | `edge_transformer.py`, `tf_edges.py`, `cell_embedding.py` | transformer link probabilities (forward, reverse, harmonic); contrastive cell-embedding similarity |
| link probability | `edge_model.py` | LightGBM on all of the above |
| division prior | `division_cnn.py`, `division_features.py`, `division_cat.py`, `division_pair.py` | 5-frame division CNN; CatBoost node model; mother-daughter pair model (LightGBM + CatBoost); prior = MAX(pair, node) |
| tracking | `ilp_solve.py` | tracksdata ILP with a per-node division cost `base - 16 * s` (base 6, sparse movies 5), then smoothing |
| post-processing | `fork_verify.py`, `gapfill.py` | image-based fork verification; bridging of 1-2 frame gaps |
| track gate | `dog_nodes.py`, `track_gate.py` | sparse movies: drop track segments not supported by 3D Net-64 / a sensitive DoG detector |
| orchestration | `downstream.py` | everything after detection |
| shared | `common.py` | constants, greedy linking, track smoothing, CSV rows |

## Models (`inference/models/`)

| file | model | training data |
|---|---|---|
| `net3d_128.pt` | 3D Net, XY-pooled by 2 (128 x 128 grid) | ground-truth nodes only; weights averaged over epochs 7-10 |
| `net3d_128_pl.pt` | same architecture | ground truth + pseudo-labels (out-of-fold detections, plus an external Zebrahub embryo) |
| `net3d_64.pt` | same architecture, XY-pooled by 4 | ground truth only; independent support detector for sparse movies |
| `edge_transformer/` | temporal U-Net + node transformer link scorer | all training movies |
| `cell_embedding.pt` | small 3D CNN, contrastive (InfoNCE) | pairs of the same nucleus in consecutive frames |
| `edge_lgbm/` | LightGBM link model | out-of-fold detections |
| `division_cnn/` | 6 division CNNs (3 seeds, with and without copy-paste augmentation) | annotated mothers vs. annotated non-dividing cells |
| `division_cat.cbm` | CatBoost node division model | out-of-fold graphs |
| `division_pair/` | LightGBM + CatBoost pair model | out-of-fold graphs |
| `fork_verifier/a`, `b` | two CatBoost fork verifiers (averaged) + sigmoid calibration | forks of the out-of-fold ILP solution |

Division labels come only from annotated tracks (two children = positive, one child = negative); unannotated nodes are
never used as negatives, because most real divisions are unannotated.

## Acknowledgements and licences

- The 3D Net backbone is adapted from a dual-encoder 3D U-Net with temporal attention from the literature; the unit
  vector field output and the advection decoding are in the spirit of Cellpose.
- `edge_transformer.py` re-implements, for inference, the linker architecture of the competition's official baseline
  code (tracking-cellmot, BSD 3-Clause License, Copyright (c) 2026 Thibaut Goldsborough); see `LICENSE-baseline.txt`.
- The ILP uses [tracksdata](https://github.com/royerlab/tracksdata) with the SCIP solver.
