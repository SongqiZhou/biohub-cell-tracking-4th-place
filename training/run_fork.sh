#!/bin/bash
# Fork verifier from the official annotations only (no hand labels; the released verifier: fork_verifier_rows.py, see
# training/README.md). Its training forks come from the tracking stage run on the training movies with out-of-fold inputs only:
# out-of-fold link probabilities, CNN scores and a division prior whose two models are fitted on the other half of the
# movies. Needs run_link.sh and run_division.sh (work/link, work/div) and the baseline package (training/edge_transformer)
# for the official division score.
set -euo pipefail
PY=${PY:-python}
M=${MODELS:-models}; L=work/link; D=work/div; F=work/fork; mkdir -p $F

# 1. Density bands as at inference (DoG spacing >= 19.7 um is sparse).
$PY - <<'EOF'
import json
r = json.load(open('work/dog_train.json'))
json.dump({m: {'dog_sp': v['dog_sp'], 'sparse': v['dog_sp'] >= 19.7} for m, v in r.items()}, open('work/fork/route.json', 'w'))
EOF

# 2. Out-of-fold division prior of the node set of work/link: CNN ensemble (plain + copy-paste networks of the other half),
#    division CatBoost and pair model fitted on the other half (same recipes as the deployed models), MAX-combined.
$PY training/candidate_graphs.py --nodes $L/nodes --edges $L/edges_oof --out $L/cand_npz
for h in 0 1; do
  $PY inference/division_cnn.py --cand-dir $L/cand_half$h --data-dir data/train --out-dir $L/div_cnn_paste --model "$D/cnn_oof/cnn_copypaste_half${h}_s*.pt" --tta 4 --half
done
$PY training/mean_division_scores.py --inputs $L/div_cnn_plain,$L/div_cnn_paste --out $L/div_cnn
for h in 0 1; do
  mkdir -p $F/nodes_half$h
  for m in $($PY training/fold_movies.py --half $h | tr ',' ' '); do ln -sfn "$(realpath $L/nodes/$m.npz)" $F/nodes_half$h/$m.npz; done
  $PY training/fit_division_cat.py --cand $D/cand_npz --div-dir $D/div_cnn --labels $D/labels --gt data/train --half $h --out $F/division_cat_half$h.cbm
  $PY inference/division_cat.py --model $F/division_cat_half$h.cbm --cand $L/cand_half$h --div-dir $L/div_cnn --out $F/div_cat
  $PY training/train_division_pair.py --nodes $L/nodes --gt data/train --edges $L/edges_oof --edges-noemb $L/edges_oof_noemb --harm $L/edges_tf_harm \
      --emb $L/embeddings --div-dir $L/div_cnn_plain --test data/train --half $h --out $F/division_pair_half$h
  $PY inference/division_pair.py --nodes $F/nodes_half$h --edges $L/edges_oof --harm $L/edges_tf_harm --emb $L/embeddings --div-dir $L/div_cnn \
      --cat $F/div_cat --test data/train --models $F/division_pair_half$h --out $F/div_prior --jobs 16
done

# 3. Tracking ILP with the deployed settings.
$PY inference/ilp_solve.py --nodes $L/nodes --edges $L/edges_oof --div-dir $F/div_prior --alt-dir $L/edges_tf_harm --alt-min-divs 0.2 \
    --div-costs 6,16,0.5 --sparse-div-costs 5,16,0.5 --route $F/route.json --out $F/pred.csv --graph-out $F/graph --jobs 16

# 4. Rows and the two verifiers, both trained on all movies (use them with drop threshold 0.40: downstream.py --fork-thr 0.40).
$PY training/fork_rows.py --graph $F/graph --nodes $L/nodes --labels $L/labels --edges $L/edges_oof --prior $F/div_prior --gt data/train \
    --data data/train --out $F/rows
$PY training/train_fork_verifier.py --rows $F/rows --out $M/fork_verifier_official --all-movies
