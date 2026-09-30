#!/bin/bash
# Division models: the division CNNs, the per-node division CatBoost and the mother-daughter pair model.
# Needs run_net3d.sh (fold 3D Nets + the pseudo-label model), the fold edge transformers and run_link.sh (work/link).
set -euo pipefail
PY=${PY:-python}
M=${MODELS:-models}; L=work/link; D=work/div; mkdir -p $D

# 1. Division CNN: patches at the annotated nodes, three plain and three copy-paste networks on all movies (deployed), and
#    two of each per half of the movies for out-of-fold scores.
$PY training/division_patches.py --data data/train --out $D/patches
$PY training/train_division_cnn.py --patches $D/patches --out $M/division_cnn
$PY training/train_division_cnn.py --patches $D/patches --out $M/division_cnn --paste
for h in 0 1; do
  $PY training/train_division_cnn.py --patches $D/patches --out $D/cnn_oof --half $h --seeds 2
  $PY training/train_division_cnn.py --patches $D/patches --out $D/cnn_oof --half $h --seeds 2 --paste
done

# 2. Training data of the division CatBoost: the out-of-fold node set with the dense seed threshold 0.97 in every movie
#    (the movies below 19.0 um DoG spacing are the same as in work/link), its candidate links, out-of-fold transformer
#    probabilities and out-of-fold link probabilities of an edge model without the embedding column.
mkdir -p $D/det
for k in 0 1 2 3 4; do
  for m in $($PY training/fold_movies.py $k --dog work/dog_train.json --dense | tr ',' ' '); do
    ln -sfn "$(realpath $L/det/$m.nodes.npy)" $D/det/$m.nodes.npy; ln -sfn "$(realpath $L/det/$m.scores.npy)" $D/det/$m.scores.npy
  done
  mv=$($PY training/fold_movies.py $k --dog work/dog_train.json --sparse)
  [ -z "$mv" ] || $PY inference/detect.py --data-dir data/train --vec $M/net3d_128_fold$k.pt --refine-with $M/net3d_128_pl.pt \
      --refine-r 6 --support 0.5 --thr 0.97 --nms-adapt 4,7,10 --movies "$mv" --out $D/det
done
$PY inference/nodes_table.py --nodes $D/det --out $D/nodes
$PY inference/candidates.py --nodes $D/nodes --out $D/cand_pool --r 10 --k 5 --k-in 3
$PY inference/edge_features.py --nodes $D/nodes --cand $D/cand_pool --out $D/feat
$PY training/link_labels.py --nodes $D/nodes --cand $D/cand_pool --gt data/train --out $D/labels
for k in 0 1 2 3 4; do
  $PY inference/tf_edges.py --nodes $D/nodes --cand $D/cand_pool --test data/train --weights runs/edge_transformer_fold$k/edge_predictor_last.pth \
      --out-root $D --movies "$($PY training/fold_movies.py $k)"
done
$PY training/fit_edge_model.py --feat $D/feat --labels $D/labels --root $D --extra tf_fwd,tf_rev,tf_harm --oof $D/edges_oof_noemb --n-jobs 48

# 3. Division CatBoost: out-of-fold CNN scores (each half scored by the plain and the copy-paste networks that did not see
#    it, averaged) on the candidate graphs, then the fit.
$PY training/candidate_graphs.py --nodes $D/nodes --edges $D/edges_oof_noemb --out $D/cand_npz
for h in 0 1; do
  $PY training/candidate_graphs.py --nodes $D/nodes --edges $D/edges_oof_noemb --out $D/cand_half$h --half $h
  $PY inference/division_cnn.py --cand-dir $D/cand_half$h --data-dir data/train --out-dir $D/div_cnn_plain --model "$D/cnn_oof/cnn_half${h}_s*.pt" --tta 4 --half
  $PY inference/division_cnn.py --cand-dir $D/cand_half$h --data-dir data/train --out-dir $D/div_cnn_paste --model "$D/cnn_oof/cnn_copypaste_half${h}_s*.pt" --tta 4 --half
done
$PY training/mean_division_scores.py --inputs $D/div_cnn_plain,$D/div_cnn_paste --out $D/div_cnn
$PY training/fit_division_cat.py --cand $D/cand_npz --div-dir $D/div_cnn --labels $D/labels --gt data/train --out $M/division_cat.cbm

# 4. Pair model, on the node set of the edge model (work/link): out-of-fold link probabilities of the edge model with and
#    without the embedding column, and out-of-fold scores of the plain division CNNs.
$PY training/fit_edge_model.py --feat $L/feat --labels $L/labels --root $L --extra tf_fwd,tf_rev,tf_harm,embed --oof $L/edges_oof --n-jobs 48
$PY training/fit_edge_model.py --feat $L/feat --labels $L/labels --root $L --extra tf_fwd,tf_rev,tf_harm --oof $L/edges_oof_noemb --n-jobs 48
for h in 0 1; do
  $PY training/candidate_graphs.py --nodes $L/nodes --edges $L/edges_oof --out $L/cand_half$h --half $h
  $PY inference/division_cnn.py --cand-dir $L/cand_half$h --data-dir data/train --out-dir $L/div_cnn_plain --model "$D/cnn_oof/cnn_half${h}_s*.pt" --tta 4 --half
done
$PY training/train_division_pair.py --nodes $L/nodes --gt data/train --edges $L/edges_oof --edges-noemb $L/edges_oof_noemb --harm $L/edges_tf_harm \
    --emb $L/embeddings --div-dir $L/div_cnn_plain --test data/train --out $M/division_pair
