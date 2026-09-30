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

## 2. Linking: cell embedding and edge model

`run_link.sh` builds the training data of the linking stage and fits its models:

| step | what | output |
|---|---|---|
| 1 | out-of-fold node set of the 199 movies: fold 3D Net-128 refined by 3D Net-128-PL; DoG spacing >= 19.0 um -> seed threshold 0.99 | `work/link/det` |
| 2 | candidate links, 26 edge features, ground-truth link labels (`link_labels.py`) | `work/link/{nodes,cand_pool,feat,labels}` |
| 3 | out-of-fold transformer link probabilities from the five fold edge transformers | `work/link/edges_tf_*` |
| 4 | cell embedding: labelled pairs, two out-of-fold encoders (movies split by sorted name into halves), and the deployed encoder on all movies (`train_cell_embedding.py`) | `work/link/edges_embed`, `cell_embedding.pt` |
| 5 | edge model: LightGBM, 600 trees, learning rate 0.05, 63 leaves (`fit_edge_model.py`) | `edge_lgbm/` |

**Checks.** Each step reproduces the files we trained with: the out-of-fold nodes and scores, the candidate links,
features and labels, the out-of-fold transformer and embedding columns, the pair patches and the node embeddings are
identical. Refitting the edge model on our original training table gives the same 639,118 labelled links and
predictions that agree with the deployed model to 3e-14.

## 3. Edge transformer

The linker of the competition baseline (royerlab/kaggle-cell-tracking-competition, commit `075fc5f`, BSD-3), retrained
with our folds. `edge_transformer/run.sh` clones the baseline, applies `edge_transformer/baseline.patch` and trains six
models (6 epochs each, lr 1e-4, batch 8, window 2, 5 um pooling, brightness + flip augmentation, seed 314159):

| split in `edge_transformer/splits.json` | training movies | output |
|---|---|---|
| 4 | all 199 | `models/edge_transformer/` (deployed) |
| 6 + k, k = 0..4 | all but fold k | `runs/edge_transformer_fold<k>/` (out-of-fold link probabilities, `run_link.sh` step 3) |

The weights of the last epoch are used. The patch changes nothing in the model or the loss; it only
* seeds the global torch / numpy generators as well (model initialisation and dropout), not only the data order,
* saves the weights of every epoch as `edge_predictor_last.pth` next to the baseline's "best" checkpoint,
* adds `--seed` and `--augment` to the command line.

Run this before `run_link.sh`.

## 4. Still being written up

The division CNN, the division graph and pair models, and the fork verifier.
