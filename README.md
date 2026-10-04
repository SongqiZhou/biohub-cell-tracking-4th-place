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
| `training/` | training code for every model: 3D Nets, edge transformer, cell embedding, edge model, division models, fork verifier; see `training/README.md` |

## Quick start (inference)

```bash
pip install -r requirements.txt
# put the models/ folder of the Kaggle dataset songqizhou/biohub-4th-place-artifacts into inference/models/
python inference/run_pipeline.py --test <dir with the *.zarr movies> --work work/run   # -> work/run/submission.csv
```

On Kaggle (2 × T4) the whole pipeline takes about 31 minutes on the 4 visible test movies.

`tracksdata` is used through its ILP interface; we used a development build (0.1.0rc9). The Kaggle notebook installs
it, together with `zarr`, SCIP and a recent `polars`, from the public `biohub-tracking-support-pack` wheels.

## External data in one paragraph

One external embryo, the public Ultrack `zebrafish_embryo` light-sheet volume (no labels): **102 windows** of
64 × 256 × 256 voxels × 100 frames (**10,200 frames**, 39 GB), pseudo-labelled by our ground-truth-only 3D Net
(**2.02 M** pseudo-label nodes). It is used by one model only, the pseudo-label 3D Net that refines the nodes, where
it makes up 34% of the training windows and 30% of the pseudo-label nodes. All other models use the 199 competition
training movies only. Details in `external_data/README.md`.

## Hand labels

The fork verifier of the final submission is also trained on 409 hand labels: candidate divisions in 40 training movies,
proposed by our models and checked by eye (274 divisions, 135 rejections). They and the verifier's other training events
are in `training/fork_verifier/`. A verifier trained from the official annotations only scores about 0.0017 lower on the
public and slightly higher on the private leaderboard, with its own drop threshold (0.40 instead of 0.25). Details in
`training/README.md`, section 5.

## Acknowledgements and licences

- The 3D Net backbone is adapted from a dual-encoder 3D U-Net with temporal attention from the literature; the unit
  vector field output and the advection decoding are in the spirit of Cellpose.
- `inference/edge_transformer.py` re-implements, for inference, the linker architecture of the competition's official
  baseline code (tracking-cellmot, BSD 3-Clause License, Copyright (c) 2026 Thibaut Goldsborough); see
  `inference/LICENSE-baseline.txt`.
- The ILP uses [tracksdata](https://github.com/royerlab/tracksdata) with the SCIP solver.
- External imaging data: Royer Lab, CZ Biohub (Ultrack `zebrafish_embryo`).
