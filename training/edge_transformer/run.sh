#!/bin/bash
# Edge transformer: the competition baseline's linker (temporal U-Net + node transformer), retrained with our splits.
# One model on all 199 movies (deployed) and five fold models (out-of-fold link probabilities for the edge model).
# Run from the repository root. DATA = absolute path of the competition training data.
set -euo pipefail
PY=${PY:-python}
DATA=$(realpath ${DATA:-data/train}); SPLITS=$(realpath training/edge_transformer/splits.json); PATCH=$(realpath training/edge_transformer/baseline.patch)
M=${MODELS:-models}

git clone https://github.com/royerlab/kaggle-cell-tracking-competition.git baseline
git -C baseline checkout 075fc5f
git -C baseline apply "$PATCH"
$PY -m pip install --no-deps -e baseline     # dependencies: requirements.txt (tracksdata pinned there)

ARGS="--data-dir $DATA --splits $SPLITS --epochs 6 --lr 1e-4 --batch-size 8 --num-workers 4 --window-size 2 --pool-kernel-um 5.0 \
      --det-loss-weight 1.0 --det-neg-weight 0.01 --seed 314159 --augment brightness,flip --single-gpu"
(cd baseline && $PY scripts/train_unet_transformer.py $ARGS --split 4 --method edge_transformer_all)
for k in 0 1 2 3 4; do
  (cd baseline && $PY scripts/train_unet_transformer.py $ARGS --split $((6 + k)) --method edge_transformer_fold$k)
done

# collect: the weights of the last (6th) epoch
mkdir -p $M/edge_transformer
cp baseline/weights/edge_transformer_all/split_4/config.json $M/edge_transformer/config.json
cp baseline/weights/edge_transformer_all/split_4/edge_predictor_last.pth $M/edge_transformer/weights.pth
for k in 0 1 2 3 4; do
  mkdir -p runs/edge_transformer_fold$k
  cp baseline/weights/edge_transformer_fold$k/split_$((6 + k))/{config.json,edge_predictor_last.pth} runs/edge_transformer_fold$k/
done
