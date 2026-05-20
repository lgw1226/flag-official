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

## Run a single experiment

```bash
CUDA_VISIBLE_DEVICES=0 python train.py env_id=dm_control/dog-trot seed=42
```

Hyperparameters are set via Hydra overrides. See `conf/config.yaml` for all available options.

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
