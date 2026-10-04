# GPU job pool for the training scripts (sourced). GPUS="0 1 2" spreads independent jobs over these GPUs, one job per GPU
# at a time; the default "0" runs everything one after another on GPU 0.
#   gpu_run <command...>   start the command (or a shell function) on the next free GPU
#   gpu_wait               wait for all started jobs; stop the script if one of them failed
read -r -a _GPUS <<< "${GPUS:-0}"
declare -a _PIDS=()

gpu_run() {
  while true; do
    for i in "${!_GPUS[@]}"; do
      local pid=${_PIDS[$i]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        if [ -n "$pid" ]; then wait "$pid" || { echo "GPU job failed: $pid" >&2; exit 1; }; fi
        CUDA_VISIBLE_DEVICES=${_GPUS[$i]} "$@" &
        _PIDS[$i]=$!
        return
      fi
    done
    sleep 2
  done
}

gpu_wait() {
  local pid
  for pid in "${_PIDS[@]}"; do
    if [ -n "$pid" ]; then wait "$pid" || { echo "GPU job failed: $pid" >&2; exit 1; }; fi
  done
  _PIDS=()
}
