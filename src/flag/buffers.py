import jax
import jax.numpy as jnp
from jax import Array
from flax.struct import dataclass
import numpy as np


@dataclass
class ReplayBatch:
    obs: Array
    act: Array
    rwd: Array
    next_obs: Array
    done: Array


@dataclass
class GuidanceBatch:
    obs: Array
    act: Array
    noise: Array
    idx: Array


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        obs_dim: int,
        act_dim: int,
        batch_size: int,
    ) -> None:
        self.obs_buffer = np.empty((size, obs_dim), dtype=np.float32)
        self.action_buffer = np.empty((size, act_dim), dtype=np.float32)
        self.reward_buffer = np.empty((size,), dtype=np.float32)
        self.next_obs_buffer = np.empty((size, obs_dim), dtype=np.float32)
        self.done_buffer = np.empty((size,), dtype=bool)

        self.size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.is_full = False

    def __len__(self) -> int:
        return self.size if self.is_full else self.ptr

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:

        self.obs_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = done

        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.is_full = True

    def sample(self, batch_size: int | None = None) -> ReplayBatch:
        bs = batch_size if batch_size is not None else self.batch_size
        max_index = self.size if self.is_full else self.ptr

        indices = np.random.randint(0, max_index, size=bs)

        batch = ReplayBatch(
            obs=jnp.array(self.obs_buffer[indices]),
            act=jnp.array(self.action_buffer[indices]),
            rwd=jnp.array(self.reward_buffer[indices]),
            next_obs=jnp.array(self.next_obs_buffer[indices]),
            done=jnp.array(self.done_buffer[indices]),
        )
        return batch

    def can_sample(self) -> bool:
        return len(self) >= self.batch_size


class GuidanceBuffer:
    def __init__(
        self,
        size: int,
        obs_dim: int,
        act_dim: int,
        batch_size: int,
    ) -> None:
        self.obs_buffer = np.empty((size, obs_dim), dtype=np.float32)
        self.action_buffer = np.empty((size, act_dim), dtype=np.float32)
        self.noise_buffer = np.empty((size, act_dim), dtype=np.float32)
        self.size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.is_full = False

    def __len__(self) -> int:
        return self.size if self.is_full else self.ptr

    def add(self, obs: np.ndarray, action: np.ndarray, noise: np.ndarray) -> None:

        obs_arr = np.asarray(obs, dtype=np.float32)
        act_arr = np.asarray(action, dtype=np.float32)
        noise_arr = np.asarray(noise, dtype=np.float32)

        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
            act_arr = act_arr[None, :]
            noise_arr = noise_arr[None, :]

        if obs_arr.shape[0] != act_arr.shape[0] or obs_arr.shape[0] != noise_arr.shape[0]:
            raise ValueError("obs, action, and noise batch size must match")

        n = obs_arr.shape[0]
        if n == 0:
            return

        first = min(n, self.size - self.ptr)
        self.obs_buffer[self.ptr : self.ptr + first] = obs_arr[:first]
        self.action_buffer[self.ptr : self.ptr + first] = act_arr[:first]
        self.noise_buffer[self.ptr : self.ptr + first] = noise_arr[:first]

        self.ptr += first
        if self.ptr >= self.size:
            self.ptr = 0
            self.is_full = True

        remaining = n - first
        if remaining > 0:
            self.obs_buffer[self.ptr : self.ptr + remaining] = obs_arr[first:]
            self.action_buffer[self.ptr : self.ptr + remaining] = act_arr[first:]
            self.noise_buffer[self.ptr : self.ptr + remaining] = noise_arr[first:]
            self.ptr += remaining
            self.is_full = True

    def sample(self, batch_size: int | None = None) -> GuidanceBatch:
        bs = batch_size if batch_size is not None else self.batch_size
        max_index = self.size if self.is_full else self.ptr

        indices = np.random.randint(0, max_index, size=bs)

        batch = GuidanceBatch(
            obs=jnp.array(self.obs_buffer[indices]),
            act=jnp.array(self.action_buffer[indices]),
            noise=jnp.array(self.noise_buffer[indices]),
            idx=jnp.array(indices),
        )
        return batch

    def can_sample(self) -> bool:
        return len(self) >= self.batch_size
