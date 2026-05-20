from typing import Callable, Literal, Optional
from copy import deepcopy

import jax
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
from flax.nnx import Rngs

from flag.utils.math import compute_ema
from flag.utils.nn import MLPConfig, MLP


def _to_rngs(rngs):
    if isinstance(rngs, Rngs):
        return rngs
    if isinstance(rngs, dict):
        return Rngs(**rngs)
    return Rngs(rngs)


class BaseCritic(nnx.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        num_networks: int,
        has_target: bool,
        use_crossq_trick: bool,
        is_distributional: bool,
        max_val: float,
        min_val: float,
        n_atoms: int,
        network_config: MLPConfig,
        rngs: Rngs | int | dict[str, int | Array],
    ):
        self.num_networks = num_networks
        self.has_target = has_target
        self.use_crossq_trick = use_crossq_trick
        self.is_distributional = is_distributional
        rngs = _to_rngs(rngs)
        self.rngs = rngs

        if use_crossq_trick:
            assert has_target is False

        if is_distributional:
            assert max_val > min_val
            assert n_atoms > 1
            self.z_atoms = jnp.linspace(min_val, max_val, n_atoms)

        input_dim = observation_dim + action_dim
        output_dim = 1 if not is_distributional else n_atoms
        network_config.in_features = input_dim
        network_config.out_features = output_dim

        self.networks = nnx.List()
        for _ in range(num_networks):
            network = MLP(config=network_config, rngs=rngs)
            self.networks.append(network)

        if has_target:
            self.target_networks = nnx.List()
            for network in self.networks:
                target_network = deepcopy(network)
                nnx.update(target_network, nnx.state(network))
                self.target_networks.append(target_network)

    def __call__(
        self,
        obs: Array,
        act: Array,
        use_target: bool = False,
    ) -> Array:
        if use_target and not self.has_target:
            raise ValueError("Target networks are not initialized.")

        x = jnp.concatenate([obs, act], axis=-1)
        networks = self.target_networks if use_target else self.networks

        outs = jnp.stack([network(x) for network in networks])
        return outs

    def update_target(self, tau: float) -> None:
        if not self.has_target:
            raise ValueError("Target networks are not initialized.")

        for network, target_network in zip(self.networks, self.target_networks):
            target_params = nnx.state(target_network, nnx.Param)
            online_params = nnx.state(network, nnx.Param)
            updated_params = jax.tree_util.tree_map(
                lambda target, online: compute_ema(tau, target, online),
                target_params,
                online_params,
            )
            nnx.update(target_network, updated_params)


class ScalarCritic(BaseCritic):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        has_target: bool,
        num_networks: int,
        use_crossq_trick: bool,
        network_config: MLPConfig,
        rngs: Rngs,
    ):
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            num_networks=num_networks,
            has_target=has_target,
            use_crossq_trick=use_crossq_trick,
            is_distributional=False,
            max_val=0.0,
            min_val=0.0,
            n_atoms=1,
            network_config=network_config,
            rngs=rngs,
        )

    def __call__(
        self,
        obs: Array,
        act: Array,
        use_target: bool = False,
    ) -> Array:
        q_values = super().__call__(
            obs,
            act,
            use_target=use_target,
        )
        return q_values


class DistributionalCritic(BaseCritic):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        num_networks: int,
        has_target: bool,
        use_crossq_trick: bool,
        max_val: float,
        min_val: float,
        n_atoms: int,
        network_config: MLPConfig,
        rngs: Rngs,
    ):
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            num_networks=num_networks,
            has_target=has_target,
            use_crossq_trick=use_crossq_trick,
            is_distributional=True,
            max_val=max_val,
            min_val=min_val,
            n_atoms=n_atoms,
            network_config=network_config,
            rngs=rngs,
        )

    def get_logits(
        self,
        obs: Array,
        act: Array,
        use_target: bool = False,
    ) -> Array:
        return super().__call__(obs, act, use_target)

    def __call__(
        self,
        obs: Array,
        act: Array,
        use_target: bool = False,
    ) -> Array:
        logits = self.get_logits(obs, act, use_target)
        probs = jax.nn.softmax(logits, axis=-1)
        q_values = jnp.sum(
            probs * self.z_atoms,
            axis=-1,
            keepdims=True,
        )
        return q_values


class ValueNetwork(nnx.Module):
    def __init__(
        self,
        observation_dim: int,
        network_config: MLPConfig,
        has_target: bool,
        rngs: Rngs | int | dict[str, int | Array],
    ):
        rngs = _to_rngs(rngs)
        self.rngs = rngs
        self.has_target = has_target

        v_config = deepcopy(network_config)
        v_config.in_features = observation_dim
        v_config.out_features = 1

        self.network = MLP(config=v_config, rngs=rngs)

        if self.has_target:
            self.target_network = deepcopy(self.network)
            nnx.update(self.target_network, nnx.state(self.network))

    def __call__(self, obs: Array, use_target: bool = False) -> Array:
        if use_target and not self.has_target:
            raise ValueError("Target network is not initialized.")
        network = self.target_network if use_target else self.network
        return network(obs)

    def update_target(self, tau: float) -> None:
        if not self.has_target:
            return

        target_params = nnx.state(self.target_network, nnx.Param)
        online_params = nnx.state(self.network, nnx.Param)
        updated_params = jax.tree_util.tree_map(
            lambda target, online: compute_ema(tau, target, online),
            target_params,
            online_params,
        )
        nnx.update(self.target_network, updated_params)
