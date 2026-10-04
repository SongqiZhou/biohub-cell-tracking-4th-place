#!/bin/bash
# Training data and models of the linking stage. Needs the 3D Nets from run_net3d.sh (fold models + the pseudo-label model)
# and the five fold edge transformers from training/edge_transformer/ (runs/edge_transformer_fold{0..4}).
# GPUS="0 1 2" runs the independent jobs of a step in parallel (training/gpus.sh).
set -euo pipefail
PY=${PY:-python}
M=${MODELS:-models}; W=work/link; mkdir -p $W
source training/gpus.sh

detect_fold() {   # out-of-fold nodes of fold k: dense movies at 0.97, sparse movies at 0.99
  for band in dense sparse; do
    thr=$([ $band = dense ] && echo 0.97 || echo 0.99)
    mv=$($PY training/fold_movies.py $1 --dog work/dog_train.json --$band)
    [ -z "$mv" ] || $PY inference/detect.py --data-dir data/train --vec $M/net3d_128_fold$1.pt --refine-with $M/net3d_128_pl.pt \
        --refine-r 6 --support 0.5 --thr $thr --nms-adapt 4,7,10 --movies "$mv" --out $W/det
  done
}

# 1. Out-of-fold node set of the 199 training movies: fold 3D Net-128 + refinement with 3D Net-128-PL. Movies with DoG spacing
#    >= 19.0 um use the stricter seed threshold (the training node set used 19.0 um; inference uses 19.7 um).
$PY inference/dog_route.py --test data/train --out work/dog_train.json --procs 8
for k in 0 1 2 3 4; do gpu_run detect_fold $k; done
gpu_wait

# 2. Candidate links, their features and ground-truth labels.
$PY inference/nodes_table.py --nodes $W/det --out $W/nodes
$PY inference/candidates.py --nodes $W/nodes --out $W/cand_pool --r 10 --k 5 --k-in 3
$PY inference/edge_features.py --nodes $W/nodes --cand $W/cand_pool --out $W/feat
$PY training/link_labels.py --nodes $W/nodes --cand $W/cand_pool --gt data/train --out $W/labels

# 3. Out-of-fold transformer link probabilities (each fold's movies scored by the edge transformer that did not see them).
for k in 0 1 2 3 4; do
  gpu_run $PY inference/tf_edges.py --nodes $W/nodes --cand $W/cand_pool --test data/train --weights runs/edge_transformer_fold$k/edge_predictor_last.pth \
      --out-root $W --movies "$($PY training/fold_movies.py $k)"
done
gpu_wait

# 4. Cell embedding: two out-of-fold encoders (for the edge model's training data) and the deployed encoder.
$PY training/train_cell_embedding.py pairs --nodes $W/nodes --cand $W/cand_pool --labels $W/labels --data data/train --out $W/pairs
for f in 0 1; do
  gpu_run $PY training/train_cell_embedding.py train --pairs $W/pairs --fold $f --nodes $W/nodes --cand $W/cand_pool --data data/train \
      --out $W/edges_embed --emb-out $W/embeddings
done
gpu_run $PY training/train_cell_embedding.py train --pairs $W/pairs --all $M/cell_embedding.pt
gpu_wait

# 5. Edge model.
$PY training/fit_edge_model.py --feat $W/feat --labels $W/labels --root $W --extra tf_fwd,tf_rev,tf_harm,embed --out $M/edge_lgbm
