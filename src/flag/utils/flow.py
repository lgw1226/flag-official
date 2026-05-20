from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array
from diffrax import (
    Euler,
    Midpoint,
    Dopri5,
    AbstractSolver,
    SaveAt,
    diffeqsolve,
    ODETerm,
)


def parse_ode_solver(name: str):
    assert name in ["euler", "midpoint", "dopri5"]
    if name == "euler":
        solver = Euler()
    elif name == "midpoint":
        solver = Midpoint()
    elif name == "dopri5":
        solver = Dopri5()
    else:
        raise ValueError("Unsupported solver type.")
    return solver


def sinusoidal_time_embedding(time: Array, dim: int) -> Array:
    
    assert dim % 2 == 0, "Time embedding dimension must be even."
    half_dim = dim // 2
    freq_idx = jnp.arange(half_dim, dtype=time.dtype)
    denom = max(half_dim - 1, 1)
    frequencies = jnp.exp(-jnp.log(10000.0) * freq_idx / denom)
    angles = time[..., jnp.newaxis] * frequencies
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def standard_normal_log_prob(x: Array) -> Array:
    
    d = x.shape[-1]
    log_norm = 0.5 * d * jnp.log(2 * jnp.pi)
    return -0.5 * jnp.sum(x**2, axis=-1, keepdims=True) - log_norm


def solve_with_logprob_single(
    noise: Array,
    tangent: Array,
    condition: Array,
    vector_field: Callable[[Array, Array, Array], Array],
    solver_name: str,
    dt: float,
):
    
    def augmented_vector_field(t, y, args):
        xt, logprob = y
        cond, tang = args


        xt_batch = xt[None, :]
        cond_batch = cond[None, :]
        t_batch = jnp.array([t])

        def conditioned_vector_field(x):
            return vector_field(t_batch, x, cond_batch)

        velocity, jvp = jax.jvp(conditioned_vector_field, (xt_batch,), (tang[None, :],))
        velocity = velocity[0]
        jvp = jvp[0]
        divergence = jnp.sum(tang * jvp, axis=-1, keepdims=True)
        return velocity, -divergence

    sol = diffeqsolve(
        ODETerm(augmented_vector_field),
        parse_ode_solver(solver_name),
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=(noise, standard_normal_log_prob(noise)),
        args=(condition, tangent),
    )
    return sol.ys[0][0], sol.ys[1][0]


def solve_with_logprob(
    noise: Array,
    tangent: Array,
    condition: Array,
    vector_field: Callable[[Array, Array, Array], Array],
    solver_name: str,
    dt: float,
):
    
    flow_sol, logp = jax.vmap(
        lambda n, t, c: solve_with_logprob_single(n, t, c, vector_field, solver_name, dt)
    )(noise, tangent, condition)
    return flow_sol, logp


def solve_single(
    noise: Array,
    condition: Array,
    vector_field: Callable[[Array, Array, Array], Array],
    solver_name: str,
    dt: float,
):
    
    def conditioned_vector_field(t, y, args):
        xt = y
        cond = args

        xt_batch = xt[None, :]
        cond_batch = cond[None, :]
        t_batch = jnp.array([t])

        vf = vector_field(t_batch, xt_batch, cond_batch)
        return vf[0]

    sol = diffeqsolve(
        ODETerm(conditioned_vector_field),
        parse_ode_solver(solver_name),
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=noise,
        args=condition,
        saveat=SaveAt(t1=True),
    )
    return sol.ys[0]


def solve(
    noise: Array,
    condition: Array,
    vector_field: Callable[[Array, Array, Array], Array],
    solver_name: str,
    dt: float,
):
    
    return jax.vmap(
        lambda n, c: solve_single(n, c, vector_field, solver_name, dt)
    )(noise, condition)

def solve_sde(
    noise: Array,
    condition: Array,
    eps: Array,
    vector_field: Callable[[Array, Array, Array], Array],
    beta: float,
    t0: float = 0.0,
    t1: float = 1.0,
    min_std: float = 1e-6,
):
    
    B, A = noise.shape
    Tm1 = eps.shape[1]
    N = eps.shape[2]
    O = condition.shape[-1]

    t_span = jnp.linspace(t0, t1, Tm1 + 1, dtype=jnp.float32)
    dt_span = t_span[1:] - t_span[:-1]

    eps_steps = jnp.swapaxes(eps, 0, 1)


    x = jnp.broadcast_to(noise[:, jnp.newaxis], (B, N, A))
    cond_bn = jnp.broadcast_to(condition[:, jnp.newaxis], (B, N, O)).reshape(B * N, O)
    beta_sq = jnp.asarray(beta, dtype=noise.dtype) ** 2

    def step(x_curr: Array, inp):
        t, dt, eps_step = inp

        x_bn = x_curr.reshape(B*N, A)
        t_bn = jnp.full((B*N,), t, dtype=jnp.float32)

        v_bn = vector_field(t_bn, x_bn, cond_bn)
        v = v_bn.reshape(B, N, A)

        denom = jnp.maximum(1.0 - t, min_std)
        t_safe = jnp.maximum(t, min_std)
        sigma2 = beta_sq * (t_safe / denom)
        sigma = jnp.sqrt(sigma2)

        drift = v + (sigma2 / (2.0 * t_safe)) * (x_curr + (1.0 - t) * v)
        mean = x_curr + drift * dt
        std = jnp.maximum(sigma * jnp.sqrt(dt), min_std)

        x_next = mean + std * eps_step
        return x_next, None

    x_final, _ = jax.lax.scan(step, x, (t_span[:-1], dt_span, eps_steps))
    return x_final


def solve_with_logprob_backward_single(
    preimage_action: Array,
    tangent: Array,
    condition: Array,
    vector_field: Callable[[Array, Array, Array], Array],
    solver_name: str,
    dt: float,
):
    
    def augmented_vector_field_backward(t, y, args):
        xt, logprob = y
        cond, tang = args
        t_forward = 1.0 - t


        xt_batch = xt[None, :]
        cond_batch = cond[None, :]
        t_batch = jnp.array([t_forward])

        def conditioned_vector_field(x):
            return vector_field(t_batch, x, cond_batch)

        velocity, jvp = jax.jvp(conditioned_vector_field, (xt_batch,), (tang[None, :],))
        velocity = velocity[0]
        jvp = jvp[0]
        divergence = jnp.sum(tang * jvp, axis=-1, keepdims=True)
        return -velocity, divergence

    sol = diffeqsolve(
        ODETerm(augmented_vector_field_backward),
        parse_ode_solver(solver_name),
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=(preimage_action, jnp.array([0.0])),
        args=(condition, tangent),
    )



    x0, logp_integral = sol.ys[0][0], sol.ys[1][0]
    logp_x0 = standard_normal_log_prob(x0)
    logp_x1 = logp_x0 - logp_integral
    return x0, logp_x1


def solve_with_logprob_backward(
    preimage_action: Array,
    tangent: Array,
    condition: Array,
    vector_field: Callable[[Array, Array, Array], Array],
    solver_name: str,
    dt: float,
):
    
    flow_sol, logp = jax.vmap(
        lambda n, t, c: solve_with_logprob_backward_single(n, t, c, vector_field, solver_name, dt)
    )(preimage_action, tangent, condition)
    return flow_sol, logp

if __name__ == "__main__":
    t = jnp.array([0.0, 0.5, 1.0])
    d = 6
    emb = sinusoidal_time_embedding(t, d)
    print("Time Embedding:\n", emb)

    z = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    logp = standard_normal_log_prob(z)
    print("Standard Normal Log Probabilities:\n", logp)
