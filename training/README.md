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

## 4. Division models

`run_division.sh` trains the three parts of the division prior. It needs the fold 3D Nets, the fold edge transformers and
the linking stage's work directory (`work/link`).

| step | what | output |
|---|---|---|
| 1 | division CNN: 5-frame patches at the annotated nodes with at least one child (128,732 patches, 151 divisions; `division_patches.py`); three plain and three copy-paste networks on all movies, and two of each per half of the movies (`train_division_cnn.py`) | `division_cnn/`, `work/div/cnn_oof` |
| 2 | a second out-of-fold node set with the seed threshold 0.97 in every movie, its candidate links, features, labels and out-of-fold transformer probabilities; out-of-fold link probabilities of an edge model without the embedding column (two halves, `fit_edge_model.py --oof`) | `work/div` |
| 3 | division CatBoost: each half scored by the plain and the copy-paste CNNs that did not see it (averaged), division features on the candidate graphs (links with p >= 0.02; `candidate_graphs.py`), fit (`fit_division_cat.py`) | `division_cat.cbm` |
| 4 | pair model, on `work/link`: out-of-fold link probabilities of the edge model with and without the embedding column, out-of-fold scores of the plain CNNs, 200 random non-mother nodes per movie as negatives (`train_division_pair.py`) | `division_pair/` |

The training inputs listed above are the ones the deployed models were trained with, including where they differ from the
inference inputs (the 0.97 node set and the edge model without embedding for the CatBoost; plain-CNN scores for the pair
model's graph features). `inference/division_pair.py` provides the triplet, embedding and link features to the trainer.

**Checks.** The extracted patches are identical to ours. With deterministic cuDNN, the CNN training code gives the same
weights as our original trainer, and the copy-paste synthesis the same patches; the deployed networks were trained without
deterministic cuDNN, so a retrained CNN follows the same recipe but has different weights. Given our inputs, the
out-of-fold link probabilities (with `--n-jobs 48`, as used), the candidate graphs, the CNN scores, their average and the
division features are identical to ours, and refitting the division CatBoost reproduces the deployed model exactly. For
the pair model, the training table (labels and all 261 features) is identical to ours for the same negatives. The
negatives of the deployed model were drawn with a per-movie seed taken from Python's randomised string hash, which
cannot be recovered; the script seeds them with the CRC32 of the movie name instead, so a retrained pair model differs
from the deployed one by its negative sample.

## 5. Still being written up

The fork verifier.
