#!/bin/bash
# The three 3D Nets, in order. Run from the repository root; data/train = competition training data.
# GPU: one per training run (the five fold runs are independent and can run in parallel).
set -euo pipefail
PY=${PY:-python}
M=${MODELS:-models}; mkdir -p "$M" work

# 1. 3D Net-128 (ground truth only): five fold models (for out-of-fold detections) and the final model.
for k in 0 1 2 3 4; do
  $PY training/train_net3d.py --data data/train --fold $k --out runs/net3d_128_fold$k --pool 2 --epochs 10
  $PY training/average_checkpoints.py --out $M/net3d_128_fold$k.pt runs/net3d_128_fold$k/ep{7,8,9,10}.pt
done
$PY training/train_net3d.py --data data/train --out runs/net3d_128 --pool 2 --epochs 10
$PY training/average_checkpoints.py --out $M/net3d_128.pt runs/net3d_128/ep{7,8,9,10}.pt

# 2. Out-of-fold detections of the 199 training movies -> pseudo-labels.
for k in 0 1 2 3 4; do
  $PY inference/detect.py --data-dir data/train --vec $M/net3d_128_fold$k.pt --thr 0.97 --nms-adapt 4,7,10 \
      --movies "$($PY training/fold_movies.py $k)" --out work/oof_nodes
done
$PY external_data/make_pseudo_labels.py work/oof_nodes work/pseudo_oof

# 3. External windows, labelled by the final 3D Net-128 (see external_data/README.md), then the combined training set.
$PY external_data/extract_ultrack_windows.py --out-dir data/ultrack_windows --workers 6
$PY inference/detect.py --data-dir data/ultrack_windows --vec $M/net3d_128.pt --thr 0.97 --nms-adapt 4,7,10 --out work/ultrack_nodes
$PY external_data/make_pseudo_labels.py work/ultrack_nodes work/pseudo_ultrack
$PY external_data/make_training_mix.py --train data/train --external data/ultrack_windows \
    --pseudo-train work/pseudo_oof --pseudo-external work/pseudo_ultrack --out data/mix

# 4. 3D Net-128-PL: all 199 movies + 102 external windows, annotated nodes + pseudo-labels.
$PY training/train_net3d.py --data data/mix/data --pseudo data/mix/pseudo --out runs/net3d_128_pl --pool 2 --epochs 10
$PY training/average_checkpoints.py --out $M/net3d_128_pl.pt runs/net3d_128_pl/ep{7,8,9,10}.pt

# 5. 3D Net-64 (ground truth only, 4x coarser XY grid): 50 epochs, last epoch, fp16-safe scaling for T4 inference.
$PY training/train_net3d.py --data data/train --out runs/net3d_64 --pool 4 --epochs 50
$PY training/average_checkpoints.py --out $M/net3d_64.pt --fp16-scale-exp 7 runs/net3d_64/ep50.pt
