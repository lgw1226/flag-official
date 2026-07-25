#!/usr/bin/env bash
# Phase 1: dog-run, N={2,4,8,16,32}, seeds={0,1,2,3,4}.
set -euo pipefail

cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

PY=/home/shkim-larr/miniconda3/envs/jax_gpu/bin/python

for n in 2 4 8 16 32; do
  for seed in 0 1 2 3 4; do
    "$PY" train.py \
      env_id=dm_control/dog-run \
      seed="$seed" \
      actor.num_train_action_samples="$n" \
      diag.enabled=true \
      "$@"
  done
done
