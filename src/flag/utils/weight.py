from typing import Literal, Union
import jax
import jax.numpy as jnp
from jax import Array
from flag.alpha import Alpha, ConstantAlpha

def get_weights(
    q: Array,
    logp_gaussian: Array,
    logp_flow: Array,
    alpha: Union[Alpha, ConstantAlpha],
    eta: float,
):
    B = q.shape[0]
    logp_flow_dist, logp_flow_mean = jnp.split(logp_flow, 2, axis=1)
    logp_gaussian_dist, logp_gaussian_mean = jnp.split(logp_gaussian, 2, axis=1)

    f_dist = q[:, :-1] / alpha() - logp_flow_dist
    f_mean = q[:, -1:] / alpha() - logp_flow_mean
    adv = f_dist - f_mean
    adv = jnp.concatenate([adv, jnp.zeros((B, 1, 1))], axis=1)
    logits = eta * adv
    weights = jax.nn.softmax(logits, axis=1)



    return weights, alpha() * adv[:, :-1].mean()

def get_weights_flow(
    q: Array,
    logp_flow: Array,
    alpha: Union[Alpha, ConstantAlpha],
    eta: float,
):
    logits = eta * (q / alpha() - logp_flow)
    weights = jax.nn.softmax(logits, axis=1)


    return weights