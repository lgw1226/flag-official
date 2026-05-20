import jax
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx

from jax.lax import stop_gradient as sg


class Alpha(nnx.Module):
    def __init__(self, init_value: float, min_log_alpha: float = -40.0, max_log_alpha: float = 2.0) -> None:
        self._log_alpha = nnx.Param(jnp.log(jnp.array(init_value)))
        self.min_log_alpha = min_log_alpha
        self.max_log_alpha = max_log_alpha

    def __call__(self) -> Array:
        clipped_log_alpha = jnp.clip(self._log_alpha.value, self.min_log_alpha, self.max_log_alpha)
        return jnp.exp(clipped_log_alpha)


class ConstantAlpha(nnx.Module):
    def __init__(self, init_value: float) -> None:
        self._alpha = nnx.Variable(jnp.array(init_value))

    def __call__(self) -> Array:
        return self._alpha.value


class EntropyEMA(nnx.Module):

    def __init__(self, decay: float = 0.99, init_value: float = 0.0):
        self.decay = decay
        self.ema_value = nnx.Variable(jnp.array(init_value))
        self.initialized = nnx.Variable(jnp.array(False))

    def update(self, new_value: Array) -> Array:

        new_ema = jnp.where(
            self.initialized.value, self.decay * self.ema_value.value + (1 - self.decay) * new_value, new_value
        )
        self.ema_value.value = new_ema
        self.initialized.value = jnp.array(True)
        return new_ema

    def __call__(self) -> Array:
        return self.ema_value.value


def alpha_loss_fn(
    alpha: Alpha,
    logp: Array,
    target_entropy: float,
) -> Array:
    log_alpha = alpha._log_alpha.value
    alpha_loss = jnp.mean(log_alpha * sg(-logp - target_entropy))
    return alpha_loss
