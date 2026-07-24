"""Pickle-based training checkpoints (modules + optimizers + replay buffers)."""

import glob
import json
import os
import pickle
from typing import Any

import jax
import numpy as np
import flax.nnx as nnx


def _buffer_state(buf: Any) -> dict:
    n = len(buf)  # only the written prefix; the rest is uninitialized np.empty
    return {
        "ptr": buf.ptr,
        "is_full": buf.is_full,
        "arrays": {k: v[:n].copy() for k, v in vars(buf).items() if k.endswith("_buffer")},
    }


def _restore_buffer(buf: Any, state: dict) -> None:
    for k, v in state["arrays"].items():
        getattr(buf, k)[: len(v)] = v
    buf.ptr, buf.is_full = state["ptr"], state["is_full"]


def save_checkpoint(
    ckpt_dir: str,
    step: int,
    modules: dict[str, Any],
    buffers: dict[str, Any],
    extra: dict[str, Any],
) -> str:
    """Write one checkpoint and append a manifest line."""
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": step,
        "modules": {k: jax.device_get(nnx.state(m)) for k, m in modules.items() if m is not None},
        "buffers": {k: _buffer_state(b) for k, b in buffers.items()},
        "extra": extra,
    }

    path = os.path.join(ckpt_dir, f"step_{step:09d}.pkl")
    with open(path + ".tmp", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)  # atomic: a crash mid-write never corrupts a checkpoint

    scalars = {k: v for k, v in extra.items() if isinstance(v, (int, float, str, type(None)))}
    with open(os.path.join(ckpt_dir, "manifest.jsonl"), "a") as f:
        f.write(json.dumps({"step": step, "path": path, **scalars}) + "\n")
    return path


def find_latest(ckpt_dir: str) -> str | None:
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pkl")))  # zero-padded -> lexicographic == numeric
    return ckpts[-1] if ckpts else None


def load_checkpoint(path: str, modules: dict[str, Any], buffers: dict[str, Any]) -> tuple[int, dict]:
    """Restore in place into freshly constructed modules/buffers. Returns (step, extra)."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    for k, state in payload["modules"].items():
        nnx.update(modules[k], state)
    for k, state in payload["buffers"].items():
        _restore_buffer(buffers[k], state)
    return payload["step"], payload["extra"]


def main() -> None:
    import optax
    import tempfile

    from flag.buffers import ReplayBuffer

    def make():
        model = nnx.Linear(3, 2, rngs=nnx.Rngs(0))
        optim = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        buf = ReplayBuffer(size=8, obs_dim=3, act_dim=2, batch_size=2)
        return model, optim, buf

    model, optim, buf = make()
    grads = jax.tree.map(lambda p: p + 1.0, nnx.state(model, nnx.Param))
    try:
        optim.update(model, grads)  # non-zero optimizer state (flax >= 0.11)
    except TypeError:
        optim.update(grads)  # flax 0.10.x
    for i in range(5):
        buf.add(np.full(3, i, np.float32), np.full(2, i, np.float32), float(i), np.full(3, i, np.float32), i % 2 == 0)

    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(d, 100, {"m": model, "o": optim}, {"b": buf}, {"logstd": -1.5, "key_data": np.zeros(2)})
        assert os.path.exists(os.path.join(d, "manifest.jsonl"))

        model2, optim2, buf2 = make()
        step, extra = load_checkpoint(find_latest(d), {"m": model2, "o": optim2}, {"b": buf2})

    assert step == 100 and extra["logstd"] == -1.5
    jax.tree.map(np.testing.assert_allclose, jax.device_get(nnx.state(model)), jax.device_get(nnx.state(model2)))
    jax.tree.map(np.testing.assert_allclose, jax.device_get(nnx.state(optim)), jax.device_get(nnx.state(optim2)))
    assert (buf2.ptr, buf2.is_full, len(buf2)) == (5, False, 5)
    np.testing.assert_allclose(buf.obs_buffer[:5], buf2.obs_buffer[:5])
    np.testing.assert_allclose(buf.reward_buffer[:5], buf2.reward_buffer[:5])
    assert (buf.done_buffer[:5] == buf2.done_buffer[:5]).all()
    print("[checkpoint] self-check ok")


if __name__ == "__main__":
    main()
