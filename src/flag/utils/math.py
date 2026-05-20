import jax.numpy as jnp
from jax import Array
from jax.nn import one_hot

from flag.alpha import Alpha


def two_hot(x: Array, bins: Array) -> Array:
    
    num_bins = len(bins)
    diffs = x - bins
    abs_diffs = jnp.abs(diffs)


    lower_idxs = jnp.argmin(jnp.where(diffs <= 0, abs_diffs, jnp.inf), axis=-1)
    lower_idxs = jnp.clip(jnp.floor(lower_idxs).astype(jnp.int32), 0, num_bins - 1)
    upper_idxs = jnp.argmin(jnp.where(diffs >= 0, abs_diffs, jnp.inf), axis=-1)
    upper_idxs = jnp.clip(jnp.ceil(upper_idxs).astype(jnp.int32), 0, num_bins - 1)
    lower_bins = bins[lower_idxs][..., jnp.newaxis]
    upper_bins = bins[upper_idxs][..., jnp.newaxis]


    ratio = (x - lower_bins) / (upper_bins - lower_bins)
    upper_weights = jnp.where(upper_bins != lower_bins, ratio, 0.0)
    lower_weights = 1.0 - upper_weights


    lower_one_hot = one_hot(lower_idxs, num_bins)
    upper_one_hot = one_hot(upper_idxs, num_bins)


    lower_weights_expanded = lower_weights * lower_one_hot
    upper_weights_expanded = upper_weights * upper_one_hot
    two_hot_encoding = lower_weights_expanded + upper_weights_expanded
    return two_hot_encoding


def compute_ema(tau, target, online):
    return (1 - tau) * target + tau * online


from typing import Callable
import jax
from jax import Array
import jax.numpy as jnp

def compute_pretanh_q_hessian(q_func: Callable[[Array, Array], Array], obs: Array, pretanh_act: Array) -> Array:
    def q_fn_wrt_u(u):
        a = jnp.tanh(u)
        return jnp.mean(q_func(obs, a))
    return jax.hessian(q_fn_wrt_u)(pretanh_act)

def compute_q_hessian(q_func: Callable[[Array, Array], Array], obs: Array, act: Array) -> Array:
    def q_given_act(a: Array) -> Array:
        return jnp.mean(q_func(obs, a))

    return jax.hessian(q_given_act)(act)


def q_hessian_to_covariance(hessian: Array, epsilon=0.1) -> Array:
    


    eigvals, eigvecs = jnp.linalg.eigh(hessian)



    diagonal_stds = 1 / jnp.clip(-eigvals, a_min=epsilon)



    covariance = eigvecs @ jnp.diag(diagonal_stds) @ eigvecs.T


    covariance = covariance + 1e-6 * jnp.eye(hessian.shape[0])

    return covariance


def compute_q_diagonal_hessian(q_func: Callable[[Array, Array], Array], obs: Array, act: Array) -> Array:
    
    def q_given_act(a: Array) -> Array:
        return jnp.mean(q_func(obs, a))



    dim = act.shape[-1]

    def d_grad_i(i):
        e_i = jnp.zeros_like(act).at[i].set(1.0)


        _, hvp = jax.jvp(jax.grad(q_given_act), (act,), (e_i,))
        return hvp[i]

    return jax.vmap(d_grad_i)(jnp.arange(dim))


if __name__ == "__main__":

    from flax import nnx
    from flax.nnx import Rngs

    class DummyQ(nnx.Module):
        def __init__(self, rngs: Rngs):
            self.layer1 = nnx.Linear(3, out_features=128, rngs=rngs)
            self.layer2 = nnx.Linear(128, 1, rngs=rngs)

        def __call__(self, o: Array, a: Array) -> Array:
            x = jnp.concatenate([o, a], axis=-1)
            x = nnx.swish(self.layer1(x))
            x = self.layer2(x)
            return x

    seed = 42
    key = jax.random.key(seed)
    rngs = Rngs(key)
    model = DummyQ(rngs)

    batched_hessian = jax.vmap(compute_q_hessian, in_axes=(None, 0, 0))
    o_batch = jnp.array([[0.5], [-1.0]])
    a_batch = jnp.array([[0.1, 0.2], [-0.2, 0.3]])
    hessians = batched_hessian(model, o_batch, a_batch)
    print("Batched Hessian matrices:")
    print(hessians)

    batched_q_hessian_to_covariance = jax.vmap(q_hessian_to_covariance, in_axes=0)
    covariances = batched_q_hessian_to_covariance(hessians)
    print("Corresponding Covariance matrices:")
    print(covariances)



    batched_cholesky = jax.vmap(jnp.linalg.cholesky, in_axes=0)
    decompositions = batched_cholesky(covariances)
    key, subkey = jax.random.split(key)

    num_samples = 16
    mu = a_batch[:, jnp.newaxis].repeat(num_samples, axis=1)

    epsilon = jax.random.normal(subkey, shape=mu.shape)
    sampled_actions = mu + jnp.einsum("bij,bsj->bsi", decompositions, epsilon)
    print("Sampled action using covariance:")
    print(sampled_actions)
