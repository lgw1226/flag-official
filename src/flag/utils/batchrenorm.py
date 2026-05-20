from typing import Optional, Any
import jax

import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.nnx import Rngs
from flax.typing import Array, Dtype, Initializer
from flax.nnx.module import first_from
from flax.nnx.nn.normalization import _canonicalize_axes, _compute_stats, _normalize


class BatchRenorm(nnx.Module):
    def __init__(
        self,
        num_features: int,
        *,
        use_running_average: Optional[bool] = None,
        axis: int = -1,
        momentum: float = 0.99,
        warmup_steps: int = 100000,
        epsilon: float = 1e-5,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = initializers.zeros_init(),
        scale_init: Initializer = initializers.ones_init(),
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: Rngs,
    ) -> None:
        super().__init__()
        feature_shape = (num_features,)
        self.running_mean = nnx.BatchStat(jnp.zeros(feature_shape, jnp.float32))
        self.running_var = nnx.BatchStat(jnp.ones(feature_shape, jnp.float32))


        self.r_max = nnx.BatchStat(jnp.array(3.0), dtype=jnp.float32)
        self.d_max = nnx.BatchStat(jnp.array(5.0), dtype=jnp.float32)
        self.steps = nnx.BatchStat(jnp.array(0), dtype=jnp.int32)

        key = rngs.params()
        self.scale: nnx.Param[jax.Array] | None
        if use_scale:
            key = rngs.params()
            self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
        else:
            self.scale = nnx.data(None)

        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            key = rngs.params()
            self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
        else:
            self.bias = nnx.data(None)

        self.num_features = num_features
        self.use_running_average = use_running_average
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups


        self.use_fast_variance = use_fast_variance
        self.warmup_steps = warmup_steps

    def __call__(
        self,
        x: Array,
        use_running_average: Optional[bool] = None,
        *,
        mask: Optional[jax.Array] = None,
    ):
        use_running_average = first_from(
            use_running_average,
            self.use_running_average,
            error_msg="""No `use_running_average` argument was provided to BatchReNorm
            as either a __call__ argument, class attribute, or nnx.flag.""",
        )





        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

        if use_running_average:
            custom_mean = self.running_mean.value
            custom_var = self.running_var.value
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
                mask=mask,
            )

            stop_gradient = jax.lax.stop_gradient


            custom_mean = mean
            custom_var = var

            std = jnp.sqrt(var + self.epsilon)
            ra_std = jnp.sqrt(self.running_var[...] + self.epsilon)

            r = jax.lax.stop_gradient(std / ra_std)
            r = jnp.clip(r, 1 / self.r_max.value, self.r_max.value)

            d = jax.lax.stop_gradient((mean - self.running_mean.value) / ra_std)
            d = jnp.clip(d, -self.d_max.value, self.d_max.value)

            tmp_var = var / (r**2)
            tmp_mean = mean - d * jnp.sqrt(custom_var) / r



            warmed_up = jnp.greater_equal(self.steps.value, self.warmup_steps).astype(jnp.float32)
            custom_var = warmed_up * tmp_var + (1 - warmed_up) * custom_var
            custom_mean = warmed_up * tmp_mean + (1 - warmed_up) * custom_mean

            self.running_mean[...] = stop_gradient(self.momentum * self.running_mean[...] + (1 - self.momentum) * mean)
            self.running_var[...] = stop_gradient(self.momentum * self.running_var[...] + (1 - self.momentum) * var)

            self.steps[...] += 1

        return _normalize(
            x,
            custom_mean,
            custom_var,
            self.scale.value if self.scale else None,
            self.bias.value if self.bias else None,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.epsilon,
        )
