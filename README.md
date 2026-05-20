# FLAG

This repository contains a JAX/Flax implementation of a FLAG agent that uses a FlagActor with distributional critics. It is configured via Hydra.

## Environment setup

MyoSuite and Gymnasium v5 are not compatible in the same environment. Use two separate conda environments.

### `flag-dmc` (DMControl / Gymnasium v5)

```bash
conda create -n flag-dmc python=3.10 -y
conda activate flag-dmc
pip install -r requirements_dmc.txt
```

### `flag-myo` (MyoSuite)

```bash
conda create -n flag-myo python=3.10 -y
conda activate flag-myo
pip install -r requirements_myo.txt
```

### Stack

- **JAX + CUDA 12** (`jax`, `jax-cuda12-plugin`) — array library and JIT compiler. All forward/backward passes are compiled with `jax.jit` (via `@nnx.jit`). Random state is explicit: every sampling call takes a PRNG key derived from a `flax.nnx.Rngs` stream.
- **Flax NNX** (`flax.nnx`) — neural network library. Models are plain Python classes that subclass `nnx.Module`. Parameters are stored as `nnx.Param` attributes and updated in-place through `nnx.Optimizer`. This is the newer NNX API (Flax ≥ 0.8), not the older Linen (`flax.linen`) API.
19
## Run a single experiment

**DMC/Gym-v5**
```bash
MUJOCO_GL=egl LD_LIBRARY_PATH= CUDA_VISIBLE_DEVICES=0 python train.py env_id=dm_control/dog-trot seed=42
```

**MyoSuite**
```bash
MUJOCO_GL=egl LD_LIBRARY_PATH= CUDA_VISIBLE_DEVICES=0 python train.py env_id=myo-reach-hard seed=42
```

Hyperparameters are set via Hydra overrides. See `conf/config.yaml` for all available options.

## Supported environments

Wrappers live in `src/flag/utils/wrappers/`.

### DMControl (`src/flag/utils/wrappers/dmcontrol.py`)

Use `flag-dmc`. Env IDs follow the pattern `dm_control/<domain>-<task>-v0` (`dm_control/dog-trot-v0`, ...).

### Gymnasium MuJoCo (`src/flag/utils/wrappers/mujoco.py`)

Use `flag-dmc`. Standard Gymnasium v5 env IDs (`HalfCheetah-v5`, ...).

### MyoSuite (`src/flag/utils/wrappers/myosuite.py`)

Use `flag-myo` (`myo-reach-hard`, `myo-obj-hold-hard`, ...).

## Troubleshooting

**Device falls back to CPU**
Clear `LD_LIBRARY_PATH` before running:
```bash
export LD_LIBRARY_PATH=
```

**MuJoCo rendering errors (`egl`, `GLFWError`, `osmesa`)**
`train.py` sets `MUJOCO_GL=egl` unconditionally, which is correct for headless Linux servers but will fail on macOS or machines without EGL. Override it before running:
```bash
# macOS
MUJOCO_GL=glfw CUDA_VISIBLE_DEVICES=0 python train.py env_id=HalfCheetah-v5 seed=42

# Headless Linux without EGL (e.g. CPU-only node)
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=0 python train.py env_id=HalfCheetah-v5 seed=42
```
