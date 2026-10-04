#!/bin/bash
# Train every model of the final pipeline, in order, from the competition training data. Run from the repository root with
# data/train = the competition training movies. The result is models/ in the layout inference/ loads
# (cp -r models inference/models, or MODELS=inference/models).
#
# Fold / half models are trained only where the pipeline needs out-of-fold inputs: the five fold 3D Net-128 models give
# the out-of-fold detections (pseudo-labels for 3D Net-128-PL, and the node set every downstream model is trained on), the
# five fold edge transformers the out-of-fold transformer columns of the edge model, the half models (cell embedding,
# division CNNs, edge model) the out-of-fold columns of the division models. No local evaluation is run.
#
# WITH_OFFICIAL_VERIFIER=1 also trains the fork verifier from the official annotations only (run_fork.sh, optional).
set -euo pipefail
export PY=${PY:-python} MODELS=${MODELS:-models}
log() { echo "[$(date '+%F %T')] $*"; }

log "1/5 3D Nets";               bash training/run_net3d.sh
log "2/5 edge transformers";     bash training/edge_transformer/run.sh
log "3/5 linking stage";         bash training/run_link.sh
log "4/5 division models";       bash training/run_division.sh
log "5/5 fork verifier (released recipe: annotations + hand labels)"
$PY training/fork_verifier_rows.py --data data/train --out work/fork_released/rows
$PY training/train_fork_verifier.py --rows work/fork_released/rows --out $MODELS/fork_verifier
if [ "${WITH_OFFICIAL_VERIFIER:-0}" = 1 ]; then
  log "optional: fork verifier from the official annotations only"; bash training/run_fork.sh
fi
log "done -> $MODELS"
