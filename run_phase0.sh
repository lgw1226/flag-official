#!/usr/bin/env bash
# Phase 0 (plan.md §4): instrumentation validation — dog-run, default buffer, 1 seed
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl LD_LIBRARY_PATH=
PY=/home/shkim-larr/miniconda3/envs/jax_gpu/bin/python
exec "$PY" train.py env_id=dm_control/dog-run seed=0 diag.enabled=true \
  ckpt_interval=100000 wandb.name=phase0-dog-run-seed0 "$@"
