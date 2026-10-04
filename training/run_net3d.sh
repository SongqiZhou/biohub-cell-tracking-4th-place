#!/bin/bash
# The three 3D Nets, in order. Run from the repository root; data/train = competition training data.
# GPUS="0 1 2" runs the independent jobs of a step in parallel (training/gpus.sh).
set -euo pipefail
PY=${PY:-python}
M=${MODELS:-models}; mkdir -p "$M" work
source training/gpus.sh

net3d_128() {   # fold k (or "all")
  if [ "$1" = all ]; then
    $PY training/train_net3d.py --data data/train --out runs/net3d_128 --pool 2 --epochs 10
    $PY training/average_checkpoints.py --out $M/net3d_128.pt runs/net3d_128/ep{7,8,9,10}.pt
  else
    $PY training/train_net3d.py --data data/train --fold $1 --out runs/net3d_128_fold$1 --pool 2 --epochs 10
    $PY training/average_checkpoints.py --out $M/net3d_128_fold$1.pt runs/net3d_128_fold$1/ep{7,8,9,10}.pt
  fi
}
net3d_64() {
  $PY training/train_net3d.py --data data/train --out runs/net3d_64 --pool 4 --epochs 50
  $PY training/average_checkpoints.py --out $M/net3d_64.pt --fp16-scale-exp 7 runs/net3d_64/ep50.pt
}

# The external windows (download + extraction, CPU / network only) are prepared in the background meanwhile.
$PY external_data/extract_ultrack_windows.py --out-dir data/ultrack_windows --workers 6 > work/extract_ultrack_windows.log 2>&1 &
EXTRACT=$!

# 1. 3D Net-128 (ground truth only): five fold models (for out-of-fold detections) and the final model; 3D Net-64
#    (ground truth only, 4x coarser XY grid: 50 epochs, last epoch, fp16-safe scaling for T4 inference), independent of them.
for k in 0 1 2 3 4; do gpu_run net3d_128 $k; done
gpu_run net3d_128 all
gpu_run net3d_64
gpu_wait

# 2. Out-of-fold detections of the 199 training movies -> pseudo-labels.
for k in 0 1 2 3 4; do
  gpu_run $PY inference/detect.py --data-dir data/train --vec $M/net3d_128_fold$k.pt --thr 0.97 --nms-adapt 4,7,10 \
      --movies "$($PY training/fold_movies.py $k)" --out work/oof_nodes
done
gpu_wait
$PY external_data/make_pseudo_labels.py work/oof_nodes work/pseudo_oof

# 3. External windows, labelled by the final 3D Net-128 (see external_data/README.md), then the combined training set.
wait $EXTRACT || { echo "external window extraction failed, see work/extract_ultrack_windows.log" >&2; exit 1; }
n=${#_GPUS[@]}
for i in $(seq 0 $((n - 1))); do
  gpu_run $PY inference/detect.py --data-dir data/ultrack_windows --vec $M/net3d_128.pt --thr 0.97 --nms-adapt 4,7,10 --shard $i/$n --out work/ultrack_nodes
done
gpu_wait
$PY external_data/make_pseudo_labels.py work/ultrack_nodes work/pseudo_ultrack
$PY external_data/make_training_mix.py --train data/train --external data/ultrack_windows \
    --pseudo-train work/pseudo_oof --pseudo-external work/pseudo_ultrack --out data/mix

# 4. 3D Net-128-PL: all 199 movies + 102 external windows, annotated nodes + pseudo-labels.
CUDA_VISIBLE_DEVICES=${_GPUS[0]} $PY training/train_net3d.py --data data/mix/data --pseudo data/mix/pseudo --out runs/net3d_128_pl --pool 2 --epochs 10
$PY training/average_checkpoints.py --out $M/net3d_128_pl.pt runs/net3d_128_pl/ep{7,8,9,10}.pt
