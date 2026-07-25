#!/usr/bin/env bash
# Phase 1 fixed GPU schedule (two sequential workers per GPU):
#   GPU 4: main N=2 and main N=4
#   GPU 5: main N=8 and main N=16
#   GPU 6: main N=32 and N=8 guidance-buffer ablations
set -uo pipefail

cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

PY=/home/shkim-larr/miniconda3/envs/jax_gpu/bin/python
LOG_DIR=outputs/phase1_logs
mkdir -p "$LOG_DIR"

run_main_job() {
  local gpu=$1
  local n=$2
  local seed=$3
  local log_file="$LOG_DIR/N${n}_seed${seed}.log"

  echo "[phase1] start main N=$n seed=$seed gpu=$gpu log=$log_file"
  echo "[phase1] resume $(date --iso-8601=seconds) gpu=$gpu" >>"$log_file"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py \
    env_id=dm_control/dog-run \
    seed="$seed" \
    actor.num_train_action_samples="$n" \
    diag.enabled=true \
    wandb.mode=online \
    resume=latest >>"$log_file" 2>&1; then
    echo "[phase1] done  main N=$n seed=$seed gpu=$gpu"
  else
    local status=$?
    echo "[phase1] FAIL  main N=$n seed=$seed gpu=$gpu status=$status"
  fi
}

run_main_chain() {
  local gpu=$1
  local n=$2
  for seed in 0 1 2 3 4; do
    run_main_job "$gpu" "$n" "$seed"
  done
}

run_buffer_job() {
  local gpu=$1
  local size=$2
  local seed=$3
  local log_file="$LOG_DIR/N8_buffer${size}_seed${seed}.log"
  local ckpt_dir="checkpoints/dm_control/dog-run/N8/buffer${size}/seed${seed}"
  local run_name="dm_control/dog-run-N8-buffer${size}-seed${seed}"

  echo "[phase1] start buffer N=8 size=$size seed=$seed gpu=$gpu log=$log_file"
  echo "[phase1] resume $(date --iso-8601=seconds) gpu=$gpu" >>"$log_file"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py \
    env_id=dm_control/dog-run \
    seed="$seed" \
    actor.num_train_action_samples=8 \
    guidance_buffer.size="$size" \
    diag.enabled=true \
    diag.shadow_interval=0 \
    ckpt_dir="$ckpt_dir" \
    wandb.name="$run_name" \
    wandb.mode=online \
    resume=latest >>"$log_file" 2>&1; then
    echo "[phase1] done  buffer N=8 size=$size seed=$seed gpu=$gpu"
  else
    local status=$?
    echo "[phase1] FAIL  buffer N=8 size=$size seed=$seed gpu=$gpu status=$status"
  fi
}

run_buffer_chain() {
  local gpu=$1
  for size in 0 51200 102400 204800; do
    for seed in 0 1 2 3 4; do
      run_buffer_job "$gpu" "$size" "$seed"
    done
  done
}

pids=()
run_main_chain 4 2 &
pids+=("$!")
run_main_chain 4 4 &
pids+=("$!")
run_main_chain 5 8 &
pids+=("$!")
run_main_chain 5 16 &
pids+=("$!")
run_main_chain 6 32 &
pids+=("$!")
run_buffer_chain 6 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
