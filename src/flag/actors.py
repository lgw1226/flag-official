from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.lax import stop_gradient as sg
import flax.nnx as nnx
from flax.nnx import Rngs

from flag.utils.flow import (
    sinusoidal_time_embedding,
    standard_normal_log_prob,
    solve,
    solve_with_logprob_backward,
)
from flag.utils.nn import MLPConfig, MLP


def _to_rngs(rngs):
    if isinstance(rngs, Rngs):
        return rngs
    if isinstance(rngs, dict):
        return Rngs(**rngs)
    return Rngs(rngs)


def calculate_logstd_bias_init(
    min_logstd: float,
    max_logstd: float,
    target_logstd: float,
) -> float:

    numerator = target_logstd - min_logstd
    denominator = 0.5 * (max_logstd - min_logstd)
    tanh_out = (numerator / denominator) - 1.0

    tanh_out = jnp.clip(tanh_out, -0.999, 0.999)
    bias = jnp.arctanh(tanh_out)

    return float(bias)


class BaseActor(nnx.Module):
    def get_action(self, observation: Array, evaluate: bool) -> Array:
        raise NotImplementedError

    def get_action_logprob(self, observation: Array) -> tuple[Array, ...]:
        raise NotImplementedError


class FlagActor(BaseActor):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        time_embed_dim: int,
        num_solver_steps: int,
        solver_name: Literal["euler", "midpoint", "dopri5"],
        num_train_action_samples: int,
        num_eval_action_samples: int,
        mean_network_config: MLPConfig,
        logstd_network_config: MLPConfig,
        rngs: Rngs | int | dict[str, int | Array],
        min_logstd: float = -20.0,
        max_logstd: float = -2.0,
        init_logstd: float = -2.0,
        logstd_mode: Literal["network", "fixed"] = "network",
    ) -> None:
        assert time_embed_dim % 2 == 0, "Time embedding dimension must be even."
        rngs = _to_rngs(rngs)

        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.logstd_mode = logstd_mode
        if logstd_mode == "fixed":
            fixed_value = jnp.asarray(init_logstd, dtype=jnp.float32)
            if fixed_value.ndim == 0:
                fixed_value = jnp.full((action_dim,), fixed_value)
            elif fixed_value.shape != (action_dim,):
                raise ValueError("fixed_logstd must be scalar or shape (action_dim,)")
            fixed_value = jnp.clip(fixed_value, min_logstd, max_logstd)
            self.fixed_logstd = nnx.Variable(fixed_value)
            self.logstd_network = None
        elif logstd_mode == "network":
            bias_init_value = calculate_logstd_bias_init(min_logstd, max_logstd, init_logstd)
            logstd_network_config.in_features = observation_dim + action_dim
            logstd_network_config.out_features = action_dim
            logstd_network_config.bias_init = bias_init_value
            self.logstd_network = MLP(config=logstd_network_config, rngs=rngs)
            self.fixed_logstd = None
        else:
            raise ValueError(f"Unknown logstd_mode: {logstd_mode}")

        mean_network_config.in_features = time_embed_dim + observation_dim + action_dim
        mean_network_config.out_features = action_dim
        self.mean_network = MLP(config=mean_network_config, rngs=rngs)

        self.action_dim = action_dim
        self.output_dim = action_dim
        self.time_embed_dim = time_embed_dim
        self.solver_name = solver_name
        self.num_train_action_samples = num_train_action_samples
        self.num_eval_action_samples = num_eval_action_samples
        self.dt = 1.0 / num_solver_steps

        self.rngs = rngs

    def __call__(self, t: Array, xt: Array, cond: Array) -> Array:
        emb = sinusoidal_time_embedding(t, self.time_embed_dim)
        x = jnp.concatenate([emb, xt, cond], axis=-1)
        return self.mean_network(x)

    def _get_logstd(self, obs, mu) -> Array:
        if self.logstd_mode == "fixed":
            fixed = self.fixed_logstd.value
            fixed = jnp.clip(fixed, self.min_logstd, self.max_logstd)
            return jnp.broadcast_to(fixed, (obs.shape[0], self.action_dim))
        net_out = self.logstd_network(jnp.concatenate([obs, sg(mu)], axis=-1))
        logstd = self.min_logstd + 0.5 * (self.max_logstd - self.min_logstd) * (jnp.tanh(net_out) + 1.0)
        return logstd

    def set_fixed_logstd(self, value: float | Array) -> None:
        if self.logstd_mode != "fixed":
            raise ValueError("logstd_mode must be 'fixed' to set fixed_logstd")
        value_arr = jnp.asarray(value, dtype=jnp.float32)
        if value_arr.ndim == 0:
            value_arr = jnp.full((self.action_dim,), value_arr)
        elif value_arr.shape != (self.action_dim,):
            raise ValueError("fixed_logstd must be scalar or shape (action_dim,)")
        self.fixed_logstd.value = jnp.clip(value_arr, self.min_logstd, self.max_logstd)

    def get_action_logprob(
        self,
        observation: Array,
        is_single: bool = False,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        B = observation.shape[0]
        S = self.num_train_action_samples
        A = self.action_dim

        noise = jax.random.normal(self.rngs.noise(), shape=(B, A))
        flow_sol = solve(
            noise,
            observation,
            self.__call__,
            self.solver_name,
            self.dt,
        )
        mu = flow_sol
        logstd = self._get_logstd(observation, mu)
        std = jnp.exp(logstd)
        if is_single:
            eps = jax.random.normal(self.rngs.eps(), shape=(B, A))
            logp_slice_preimage = standard_normal_log_prob(eps) - jnp.sum(logstd, axis=-1, keepdims=True)
            pretanh_action = sg(mu) + sg(std) * eps
            tangent = jax.random.rademacher(self.rngs.tangent(), shape=(B, A)).astype(jnp.float32)
            _, logp_flow_preimage = solve_with_logprob_backward(
                sg(pretanh_action),
                tangent,
                observation,
                self.__call__,
                self.solver_name,
                self.dt,
            )
        else:
            eps = jax.random.normal(self.rngs.eps(), shape=(B, S, A))
            eps = jnp.concatenate([eps, jnp.zeros_like(eps)], dtype=jnp.float32, axis=1)

            logp_slice_preimage = (
                standard_normal_log_prob(eps) - jnp.sum(logstd, axis=-1, keepdims=True)[:, jnp.newaxis]
            )
            pretanh_action = sg(mu)[:, jnp.newaxis] + sg(std[:, jnp.newaxis]) * eps
            tangent = jax.random.rademacher(self.rngs.tangent(), shape=(B, S, A)).astype(jnp.float32)
            tangent = jnp.concatenate([tangent, tangent], axis=1)
            _, logp_flow_preimage = solve_with_logprob_backward(
                sg(pretanh_action).reshape(B * 2 * S, A),
                tangent.reshape(B * 2 * S, A),
                observation[:, jnp.newaxis].repeat(2 * S, axis=1).reshape(B * 2 * S, -1),
                self.__call__,
                self.solver_name,
                self.dt,
            )
            logp_flow_preimage = logp_flow_preimage.reshape(B, 2 * S, 1)
        action = jnp.tanh(pretanh_action)

        clip_action = jnp.clip(action, -0.999, 0.999)
        correction = jnp.sum(
            jnp.log(1.0 - clip_action**2 + 1e-6),
            axis=-1,
            keepdims=True,
        )
        logp_slice = logp_slice_preimage - correction
        logp_flow = logp_flow_preimage - correction
        return action, logp_slice, logp_flow, pretanh_action, logstd, noise

    def get_action(self, observation: Array, evaluate: bool) -> Array:
        batch_size = observation.shape[0]
        if evaluate:
            repeated_obs = jnp.repeat(
                observation,
                repeats=self.num_eval_action_samples,
                axis=0,
            )
            noise = jax.random.normal(self.rngs.noise(), shape=(batch_size * self.num_eval_action_samples, self.output_dim))
            flow_sol = solve(
                noise,
                repeated_obs,
                self.__call__,
                self.solver_name,
                self.dt,
            )
            mu = sg(flow_sol)
            mu = mu.reshape(batch_size, self.num_eval_action_samples, -1)
            return jnp.tanh(mu)
        else:
            noise = jax.random.normal(self.rngs.noise(), shape=(batch_size, self.output_dim))
            flow_sol = solve(
                noise,
                observation,
                self.__call__,
                self.solver_name,
                self.dt,
            )
            mu = flow_sol
            logstd = self._get_logstd(observation, mu)
            std = jnp.exp(logstd)
            u = mu + std * jax.random.normal(self.rngs.eps(), shape=mu.shape)
            return sg(jnp.tanh(u))
