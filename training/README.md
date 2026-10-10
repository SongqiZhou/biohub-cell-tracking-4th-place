# Training

All models are trained on the 199 competition training movies (`data/train`); the pseudo-label 3D Net additionally
uses 102 external windows (see `external_data/README.md`) and our pseudo-labels (`data/pseudo_labels`, section 1).
Out-of-fold predictions of the 3D Nets and the edge transformers use the five embryo-stratified folds in `folds.json`;
the second-level models use two halves of the movies (every second movie by sorted name, `fold_movies.py --half`).

## End to end

`train_all.sh` trains every model of the final pipeline in order (sections 1–5) and writes `models/` in the layout
`inference/` loads (`cp -r models/. inference/models/`, or `MODELS=inference/models`):

```bash
kaggle datasets download songqizhou/biohub-4th-place-pseudo-labels -p data/pseudo_labels --unzip   # 61 MB
bash training/train_all.sh                               # MODELS=models by default
WITH_OFFICIAL_VERIFIER=1 bash training/train_all.sh      # also the annotations-only fork verifier (section 5b)
```

Fold and half models are trained only where the pipeline needs out-of-fold inputs, because the downstream models were
trained on predictions for movies the upstream model had not seen:

| extra models | used for |
|---|---|
| five fold 3D Net-128 models | out-of-fold detections: the node set all downstream models are trained on (and the pseudo-labels of 3D Net-128-PL with `PSEUDO=regenerate`) |
| five fold edge transformers | the out-of-fold transformer columns of the edge model's training data |
| two half models each: cell embedding, division CNNs, edge model | out-of-fold columns of the edge model and of the division models' training data |

No local evaluation is run. `GPUS="0 1 2 3" bash training/train_all.sh` spreads the independent jobs of each step (the
fold models, the edge transformers, the per-fold detections, the division CNNs, ...) over these GPUs, one job per GPU at a
time; the default is GPU 0 only.

**Run time.** Wall clock of a from-scratch run on 4 × H100 (`GPUS="0 1 2 3"`), about 15 h in total. That run regenerated
the pseudo-labels (`PSEUDO=regenerate`, which adds the detections for them to step 1); with the released pseudo-labels
step 1 is shorter.

| step | time |
|---|---|
| 1. 3D Nets (the external windows download meanwhile, about 4 h) | 6.7 h |
| 2. edge transformers | 4.0 h |
| 3. linking stage | 1.5 h |
| 4. division models | 3.0 h |
| 5. fork verifier | < 1 min |

Single 3D Nets on one L40S: 3D Net-128 about 45 min (fold model) to 50 min (all movies) for 10 epochs, 3D Net-64 about
1.8 h for 50 epochs, 3D Net-128-PL about 2 h for 10 epochs.

## 1. The three 3D Nets

`run_net3d.sh` runs the whole sequence:

| step | what | output |
|---|---|---|
| 1 | 3D Net-128 on ground truth: five fold models + the final model, 10 epochs, average of epochs 7–10 | `net3d_128_fold{0..4}.pt`, `net3d_128.pt` |
| 2 | 3D Net-64 on ground truth (XY pooled by 4), 50 epochs, last epoch, fp16-safe scaling | `net3d_64.pt` |
| 3 | combined training set: the 199 movies and the 102 external windows with our pseudo-labels (`data/pseudo_labels`, checked against `pseudo_labels.sha256`) | `data/mix` |
| 4 | 3D Net-128-PL on `data/mix` (annotated nodes + pseudo-labels), 10 epochs, average of epochs 7–10 | `net3d_128_pl.pt` |

Meanwhile the 102 external windows are downloaded and extracted in the background
(`external_data/extract_ultrack_windows.py`: about 115 GB of reads, 39 GB on disk in `data/ultrack_windows`).

**Pseudo-labels.** Ours are released as the Kaggle dataset `songqizhou/biohub-4th-place-pseudo-labels` (199 + 102 files,
6.8 M nodes): the out-of-fold detections of the five fold models on the competition movies and the detections of the
final 3D Net-128 on the external windows, after `external_data/make_pseudo_labels.py`. `PSEUDO=regenerate bash
training/run_net3d.sh` rebuilds them from the newly trained 3D Nets instead (out-of-fold detections → `work/pseudo_oof`,
external windows → `work/pseudo_ultrack`, training set `data/mix_regenerated`). Given the same detector weights this
reproduces our files exactly, but retrained detectors are never bit-identical, and 3D Net-128-PL is sensitive to the
difference. In a from-scratch run with regenerated pseudo-labels (all other models retrained as well) the private score
was unchanged (0.96251 vs 0.96234) while the public score dropped from 0.96987 to 0.96093. Swapping one model group at a
time into the released weights put the drop in the 3D Nets; end-to-end runs on 28 windows of four unseen Zebrahub
embryos, scored against their Ultrack tracks, put it in 3D Net-128-PL: trained on regenerated pseudo-labels it scores
0.002 lower than the released model, trained with the same code on the released pseudo-labels it matches it (0.6015 vs
0.6013). This is why the released pseudo-labels are the default. Even then a retrained 3D Net-128-PL moves the leaderboard
score by a few thousandths (section 6).

Files:

| file | content |
|---|---|
| `train_net3d.py` | training loop (Adam, lr 3e-4, batch 4, 16 random labelled frames per movie and epoch, bf16 autocast, gradient clipping 10) |
| `net3d_data.py` | frame reading and normalisation, label merging, the three-tier sparse-label targets, random Y/X flips |
| `average_checkpoints.py` | epoch averaging into the deployed weights (and the fp16-safe scaling of the 64 model) |
| `folds.json`, `fold_movies.py` | the five folds |
| `pseudo_labels.sha256` | checksums of the released pseudo-labels |

The network itself is `inference/net3d.py`; checkpoints are written in the format `inference/detect.py` loads.

**Checks.** Against the implementation we used during the competition, on the same frames and the same weights, for
both grids (the 128 models with the 3D Net-128 weights, the 64 model with the 3D Net-64 weights): the labels, the input
windows and the three-tier targets are identical (including frames with more than 320 labels, which are subsampled),
and so are the loss and the gradient norm. The training settings of all three nets are those we used; the 64 model
differs from the 128 models only in the grid and the number of epochs. The out-of-fold detection and pseudo-label step
reproduces our files exactly, up to one node in 60,000 that moves by one voxel through non-deterministic GPU
reductions. Retraining will not give bit-identical weights (GPU non-determinism), only the same recipe. On the released
pseudo-labels, 3D Net-128-PL trained with this code follows the loss curve of ours (0.6843 / 0.5661 / … / 0.5039 per
epoch against 0.6864 / 0.5688 / … / 0.5052), and so does a rerun of the training code we used during the competition, with
the same settings and seed (0.6850 / 0.5683 / … / 0.5047).

## 2. Edge transformer

The linker of the competition baseline (royerlab/kaggle-cell-tracking-competition, commit `075fc5f`, BSD-3), retrained
with our folds. `edge_transformer/run.sh` clones the baseline, applies `edge_transformer/baseline.patch` and trains six
models (6 epochs each, lr 1e-4, batch 8, window 2, 5 um pooling, brightness + flip augmentation, seed 314159):

| split in `edge_transformer/splits.json` | training movies | output |
|---|---|---|
| 4 | all 199 | `models/edge_transformer/` (deployed) |
| 6 + k, k = 0..4 | all but fold k | `runs/edge_transformer_fold<k>/` (out-of-fold link probabilities, `run_link.sh` step 3, section 3) |

The weights of the last epoch are used. The patch changes nothing in the model or the loss; it only
* seeds the global torch / numpy generators as well (model initialisation and dropout), not only the data order,
* saves the weights of every epoch as `edge_predictor_last.pth` next to the baseline's "best" checkpoint,
* adds `--seed` and `--augment` to the command line.

Run this before `run_link.sh` (section 3).

## 3. Linking: cell embedding and edge model

`run_link.sh` builds the training data of the linking stage and fits its models:

| step | what | output |
|---|---|---|
| 1 | out-of-fold node set of the 199 movies: fold 3D Net-128 refined by 3D Net-128-PL (which saw all movies, as at inference); DoG spacing >= 19.0 um -> seed threshold 0.99 | `work/link/det` |
| 2 | candidate links, 26 edge features, ground-truth link labels (`link_labels.py`) | `work/link/{nodes,cand_pool,feat,labels}` |
| 3 | out-of-fold transformer link probabilities from the five fold edge transformers | `work/link/edges_tf_*` |
| 4 | cell embedding: labelled pairs, two out-of-fold encoders (the two halves of the movies), and the deployed encoder on all movies (`train_cell_embedding.py`) | `work/link/edges_embed`, `cell_embedding.pt` |
| 5 | edge model: LightGBM, 600 trees, learning rate 0.05, 63 leaves (`fit_edge_model.py`) | `edge_lgbm/` |

**Checks.** Each step reproduces the files we trained with: the out-of-fold nodes and scores, the candidate links,
features and labels, the out-of-fold transformer and embedding columns, the pair patches and the node embeddings are
identical. Refitting the edge model on our original training table gives the same 639,118 labelled links and
predictions that agree with the deployed model to 3e-14.

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
from the deployed one by its negative sample (on the leaderboard: +0.0007 public, −0.0027 private; section 6).

## 5. Fork verifier

There are two ways to train it. The released weights (`models/fork_verifier`) come from the first one, which also uses
hand labels; the second one uses the official annotations only.

### 5a. Released verifier: annotations + hand labels

`training/fork_verifier/` holds the training events:

| file | content |
|---|---|
| `events.npz` | 152 forks of a tracking run on the out-of-fold node set, labelled by the competition's division score, and 2,034 candidate mother / daughter-pair events labelled by the annotations (mother frame, daughter positions, label) |
| `hand_labels.csv` | 409 hand labels in 39 training movies: our models proposed candidate divisions and we checked each candidate by eye; 274 are divisions (positives), 135 are not (negatives). Candidates we could not decide were left out. One of the movies (`44b6_0b24845f`) is also among the four example movies of the test folder, which are copies of training movies. No hidden test data was labelled. |
| `motion.npz` | per-movie frame-to-frame translation used to compensate common motion in the image traces |

```bash
python training/fork_verifier_rows.py --data data/train --out work/fork_released/rows       # image traces of all events
python training/train_fork_verifier.py --rows work/fork_released/rows --out models/fork_verifier
```

Each of the two verifiers is trained on one half of the movies, without every fifth movie of that half, and calibrated
on the out-of-fold scores of three inner models (each fitted on two thirds of the same movies). Use these weights with the
drop threshold 0.257 (`FORK_THR` in `inference/kaggle_notebook.py`, or `downstream.py --fork-thr 0.257`), which removes
as many forks as the released weights at 0.25.
`fork_verifier_rows.py --no-hand-labels` gives the same recipe without the hand labels.

### 5b. Official annotations only

`run_fork.sh` builds everything from the official annotations. Its training forks come from the tracking stage itself, run
on the training movies with out-of-fold inputs only:

| step | what | output |
|---|---|---|
| 1 | density bands of the training movies as at inference (DoG spacing >= 19.7 um is sparse) | `work/fork/route.json` |
| 2 | out-of-fold division prior on `work/link`: each half scored by the CNNs, the division CatBoost and the pair model trained without it (`--half`), MAX-combined | `work/fork/div_prior` |
| 3 | tracking ILP with the deployed settings | `work/fork/graph` |
| 4 | rows (`fork_rows.py`): every fork of the solved graphs, labelled by the competition's division score; and candidate events (the three pairs of the three most probable children of every node with prior >= 0.02), labelled by the annotations | `work/fork/rows` |
| 5 | two verifiers on all movies (`train_fork_verifier.py --all-movies`) | `fork_verifier_official/{a,b}` |

Use these weights with the drop threshold 0.40: copy `fork_verifier_official/{a,b}` over
`inference/models/fork_verifier/` and set `FORK_THR = 0.40` in `inference/kaggle_notebook.py` (or run `downstream.py`
with `--fork-thr 0.40`).

### Comparison

Late submissions, identical to the final submission except for the fork verifier:

| fork verifier | public | private |
|---|---|---|
| released: annotations + hand labels (5a), threshold 0.25 | 0.96987 | 0.96234 |
| annotations only, recipe of 5a (each model on half of the movies), threshold 0.385 | 0.96750 | 0.96203 |
| annotations only, 5b (models on all movies), threshold 0.40 | 0.96816 | 0.96297 |

Without hand labels the verifier is about 0.0017 lower on the public and slightly higher on the private leaderboard, once
its threshold is chosen for it; with half of the movies per model it is lower on both. The threshold does not carry over:
without hand labels the calibrated scores are higher, and 0.40 removes about as many forks as the released verifier at 0.25.

The verifier features are the image traces of `inference/fork_verify.py`
(117 separation / peak / displacement features and 40 separation-trend features); the official division score comes from
the baseline package installed for the edge transformer (`tracking_cellmot.division_metrics`).

## 6. What to expect from a retrain

Retraining follows our recipes but does not give our weights: GPU training is not bit-reproducible, and single models
change the leaderboard score by a few thousandths. Late submissions with retrained models, all run with the released
inference code and settings:

| models | public | private |
|---|---|---|
| released weights (final submission) | 0.96987 | 0.96234 |
| all retrained from scratch with this repository, pseudo-labels regenerated (`PSEUDO=regenerate`) | 0.96093 | 0.96251 |
| as above, but 3D Net-128-PL retrained on the released pseudo-labels (close to the default flow; the downstream models were still trained on nodes refined by the other 3D Net-128-PL) | 0.96335 | 0.96397 |
| released weights, the three 3D Nets replaced by the from-scratch ones | 0.95902 | 0.95946 |
| released weights, the linking models (edge transformer, cell embedding, edge model) replaced | 0.96925 | 0.96328 |
| released weights, the division models (CNNs, CatBoost, pair model) replaced | 0.96767 | 0.96087 |
| released weights, 3D Net-128-PL retrained with this repository on the released pseudo-labels | 0.96865 | 0.95955 |
| released weights, 3D Net-128-PL retrained with the training code we used during the competition, on the same pseudo-labels with the same settings and seed | 0.96346 | 0.96173 |
| released weights, pair model refitted on our training inputs (only the negative sample differs, section 4) | 0.97058 | 0.95966 |

- The two retrains of 3D Net-128-PL on the same data differ by 0.0052 on the public and 0.0022 on the private
  leaderboard, in opposite directions. Their loss curves both follow ours to within 0.003 per epoch, and on the unseen
  Zebrahub check of section 1 both match the released model (0.6015 and 0.6008 vs 0.6013). The competition code does not
  score better than this repository; the spread is training noise.
- A different random negative sample for the pair model alone moves the private score by 0.0027.
- Over the eight submissions with retrained models, the public score is on average 0.0045 below the final submission
  (lower in 7 of 8) and the private score 0.0010 below (higher in 3 of 8). The final submission was chosen by its public
  score, so that score is a favourable draw. A full retrain should land in the range above: about 0.959–0.971 public and
  0.959–0.964 private.
