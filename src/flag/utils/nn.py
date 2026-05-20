from typing import Callable, Optional
from dataclasses import dataclass

import flax.nnx as nnx
from flax.nnx import Rngs
from jax import Array

from flag.utils.batchrenorm import BatchRenorm


@dataclass
class MLPConfig:
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    hidden_features: tuple[int, ...] = (256, 256)
    activation: Callable[[Array], Array] = nnx.relu
    use_batch_renorm: bool = False
    batch_renorm_momentum: float = 0.99
    batch_renorm_warmup_steps: int = 100000
    bias_init: float | None = None
    use_layer_norm: bool = False
    use_dropout: bool = False
    dropout_rate: float = 0.1


class MLP(nnx.Module):
    def __init__(
        self,
        *,
        config: MLPConfig,
        rngs: Rngs,
    ) -> None:
        assert config.in_features is not None
        assert config.out_features is not None

        hidden = tuple(config.hidden_features)
        arch = (config.in_features,) + hidden + (config.out_features,)
        self.layers = nnx.List()

        for i in range(len(arch) - 2):
            if config.use_batch_renorm:
                self.layers.append(
                    BatchRenorm(
                        num_features=arch[i],
                        momentum=config.batch_renorm_momentum,
                        warmup_steps=config.batch_renorm_warmup_steps,
                        rngs=rngs,
                    )
                )
            self.layers.append(nnx.Linear(arch[i], arch[i + 1], rngs=rngs))
            if config.use_layer_norm:
                self.layers.append(nnx.LayerNorm(arch[i + 1], rngs=rngs))
            self.layers.append(config.activation)
            if config.use_dropout:
                self.layers.append(nnx.Dropout(config.dropout_rate, rngs=rngs))


        if config.use_batch_renorm:
            self.layers.append(
                BatchRenorm(
                    num_features=arch[-2],
                    momentum=config.batch_renorm_momentum,
                    warmup_steps=config.batch_renorm_warmup_steps,
                    rngs=rngs,
                )
            )

        if config.bias_init is not None:
            bias_initializer = nnx.initializers.constant(config.bias_init)
            self.layers.append(nnx.Linear(arch[-2], arch[-1], rngs=rngs, bias_init=bias_initializer))
        else:
            self.layers.append(nnx.Linear(arch[-2], arch[-1], rngs=rngs))

        self.activation = config.activation

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


def make_mlp_factory(config: MLPConfig):

    def factory(in_features: int, out_features: int, rngs: Rngs) -> nnx.Module:
        config.in_features = in_features
        config.out_features = out_features
        return MLP(config=config, rngs=rngs)

    return factory


if __name__ == "__main__":
    base_config = MLPConfig()
    mlp_factory = make_mlp_factory(base_config)
    mlp = mlp_factory(4, 2, nnx.Rngs(0))
    print(mlp)
