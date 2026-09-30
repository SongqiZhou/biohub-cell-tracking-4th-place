# Training

All models are trained on the 199 competition training movies (`data/train`); the pseudo-label 3D Net additionally
uses 102 external windows (see `external_data/README.md`). Out-of-fold predictions use the five embryo-stratified
folds in `folds.json`.

## 1. The three 3D Nets

`run_net3d.sh` runs the whole sequence:

| step | what | output |
|---|---|---|
| 1 | 3D Net-128 on ground truth: five fold models + the final model, 10 epochs, average of epochs 7–10 | `net3d_128_fold{0..4}.pt`, `net3d_128.pt` |
| 2 | out-of-fold detections of the 199 movies with the fold models → pseudo-labels | `work/pseudo_oof` |
| 3 | external windows labelled by `net3d_128.pt` → combined training set | `data/mix` |
| 4 | 3D Net-128-PL on `data/mix` (annotated nodes + pseudo-labels), 10 epochs, average of epochs 7–10 | `net3d_128_pl.pt` |
| 5 | 3D Net-64 on ground truth (XY pooled by 4), 50 epochs, last epoch, fp16-safe scaling | `net3d_64.pt` |

Files:

| file | content |
|---|---|
| `train_net3d.py` | training loop (Adam, lr 3e-4, batch 4, 16 random labelled frames per movie and epoch, bf16 autocast, gradient clipping 10) |
| `net3d_data.py` | frame reading and normalisation, label merging, the three-tier sparse-label targets, random Y/X flips |
| `average_checkpoints.py` | epoch averaging into the deployed weights (and the fp16-safe scaling of the 64 model) |
| `folds.json`, `fold_movies.py` | the five folds |

The network itself is `inference/net3d.py`; checkpoints are written in the format `inference/detect.py` loads.

**Checks.** Against the implementation we used during the competition, on the same frames and the same weights:
the labels, the input windows and the three-tier targets are identical (including frames with more than 320 labels,
which are subsampled), and so are the loss and the gradient norm. The out-of-fold detection and pseudo-label step
reproduces our files exactly, up to one node in 60,000 that moves by one voxel through non-deterministic GPU
reductions. Retraining will not give bit-identical weights (GPU non-determinism), only the same recipe.

## 2. Downstream models

Being written up: edge transformer, cell embedding, edge model, division CNN, division graph and pair models, fork
verifier.
