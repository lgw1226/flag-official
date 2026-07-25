"""Phase-0 training diagnostics (plan.md Part II-V).

All logic lives here; train.py only calls the guarded hooks. Isolation rules:
- own jax key chain (seed+999) and private np Generator -> never touches
  actor.rngs, global np.random, or buffer.sample().
- all Q evaluations with critic in eval mode; nothing here mutates the live
  actor/critic/alpha (asserted bitwise in main()).
- every jitted function sees static shapes only (batch sizes from config).
Residuals/targets live in PRETANH (u) space, matching flag_actor_loss_fn.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient as sg
import numpy as np
import flax.nnx as nnx

from flag.utils.flow import solve, solve_with_logprob_backward, standard_normal_log_prob
from flag.utils.weight import get_weights


def _msn(x):
    """Mean over batch of squared L2 norms (plan's 1/B sum ||.||^2)."""
    return jnp.mean(jnp.sum(x**2, axis=-1))


def _tanh_correction(pretanh):
    # copied from actors.py get_action_logprob
    clip_action = jnp.clip(jnp.tanh(pretanh), -0.999, 0.999)
    return jnp.sum(jnp.log(1.0 - clip_action**2 + 1e-6), axis=-1, keepdims=True)


@nnx.jit
def _residuals(actor, obs, noise, target):
    """(plain, Mahalanobis-scaled) endpoint residual vs a pretanh target."""
    e = solve(noise, obs, actor.__call__, actor.solver_name, actor.dt)
    logstd = actor._get_logstd(obs, e)
    r = _msn(e - target)
    r_kl = jnp.mean(jnp.sum((e - target) ** 2 / (2.0 * jnp.exp(2.0 * logstd)), axis=-1))
    return r, r_kl


@nnx.jit
def _sample_probe(actor, obs, key):
    """Probe pretanh actions from the current policy + Rademacher tangent (CRN)."""
    kz, ke, kt = jax.random.split(key, 3)
    B, A = obs.shape[0], actor.action_dim
    z = jax.random.normal(kz, (B, A))
    mu = solve(z, obs, actor.__call__, actor.solver_name, actor.dt)
    logstd = actor._get_logstd(obs, mu)
    pretanh = mu + jnp.exp(logstd) * jax.random.normal(ke, (B, A))
    tangent = jax.random.rademacher(kt, (B, A)).astype(jnp.float32)
    return pretanh, tangent


@nnx.jit
def _flow_logprob(actor, obs, pretanh, tangent):
    """log pi_flow(pretanh|obs) with the tanh log-det correction from actors.py."""
    _, logp_pre = solve_with_logprob_backward(
        pretanh, tangent, obs, actor.__call__, actor.solver_name, actor.dt
    )
    return logp_pre - _tanh_correction(pretanh)


@partial(nnx.jit, static_argnames=("n_ref", "eta"))
def _reference_target(actor, critic, alpha, obs, z, key, n_ref, eta):
    """High-sample u* (plan sec.8): flag_actor_loss_fn's target with S := n_ref.

    Reproduces the exact (B, 2S) concat layout of actors.get_action_logprob's
    batch branch so get_weights is reused unchanged.
    """
    B, O = obs.shape
    A = actor.action_dim
    S = n_ref
    critic.eval()
    ke, kt = jax.random.split(key)

    mu = solve(z, obs, actor.__call__, actor.solver_name, actor.dt)
    logstd = actor._get_logstd(obs, mu)
    std = jnp.exp(logstd)

    eps = jax.random.normal(ke, (B, S, A))
    eps = jnp.concatenate([eps, jnp.zeros_like(eps)], axis=1)
    logp_slice_pre = standard_normal_log_prob(eps) - jnp.sum(logstd, axis=-1, keepdims=True)[:, jnp.newaxis]
    pretanh = mu[:, jnp.newaxis] + std[:, jnp.newaxis] * eps

    tangent = jax.random.rademacher(kt, (B, S, A)).astype(jnp.float32)
    tangent = jnp.concatenate([tangent, tangent], axis=1)
    _, logp_flow_pre = solve_with_logprob_backward(
        sg(pretanh).reshape(B * 2 * S, A),
        tangent.reshape(B * 2 * S, A),
        obs[:, jnp.newaxis].repeat(2 * S, axis=1).reshape(B * 2 * S, -1),
        actor.__call__,
        actor.solver_name,
        actor.dt,
    )
    logp_flow_pre = logp_flow_pre.reshape(B, 2 * S, 1)

    correction = _tanh_correction(pretanh)
    logp_slice = logp_slice_pre - correction
    logp_flow = logp_flow_pre - correction

    # mirror flag_actor_loss_fn: q on act[:, :S+1], ensemble mean, use_target=False
    action = jnp.tanh(pretanh)
    obs_flat = obs[:, jnp.newaxis].repeat(S + 1, axis=1).reshape(-1, O)
    act_flat = action[:, : S + 1].reshape(-1, A)
    q = jnp.mean(critic(obs_flat, act_flat, use_target=False), axis=0).reshape(B, S + 1, 1)

    weights, _ = get_weights(sg(q), logp_slice, logp_flow, alpha, eta)
    pretanh_clip = jnp.clip(pretanh, -3.8, 3.8)
    u_ref = (weights * pretanh_clip[:, : S + 1]).sum(axis=1)
    ess = 1.0 / jnp.sum(weights[:, : S + 1] ** 2, axis=1)
    return u_ref, jnp.mean(ess)


@nnx.jit
def _staleness_errors(actor, obs, z, u_star, u_ref):
    e = solve(z, obs, actor.__call__, actor.solver_name, actor.dt)
    return _msn(e - u_star), _msn(u_star - u_ref), _msn(e - u_ref)


# --- sparse KL-CFM alignment shadow (phase1_plan.md sec.2.5). No separate KL Adam
# optimizer: g_KL is compared against the pure CFM gradient and against the actual
# AdamW step (cloned live optimizer incl. momentum) taken with the CFM loss.


def _shadow_kl_loss(model, obs, z, u_ref):
    e = solve(z, obs, model.__call__, model.solver_name, model.dt)
    logstd = model._get_logstd(obs, e)
    return jnp.mean(jnp.sum((e - u_ref) ** 2 / (2.0 * jnp.exp(2.0 * logstd)), axis=-1))


def _shadow_cfm_loss(model, obs, z, u_ref, t):
    # the actual FLAG CFM loss form (jnp.mean of squares), toward u_ref
    xt = (1.0 - t) * z + t * u_ref
    vt = model(t.ravel(), xt, obs)
    return jnp.mean((vt - (u_ref - z)) ** 2)


def _tree_dot(a, b):
    return sum(jax.tree.leaves(jax.tree.map(lambda x, y: jnp.vdot(x, y), a, b)))


@nnx.jit
def _shadow_alignment(actor, optim_cfm, model_kln, obs, z, u_ref, t):
    """Gradient directions + one live-Adam CFM step + norm-matched KL comparator.

    Mutates only the clones (optim_cfm, model_kln); `actor` is read-only.
    """
    eps = 1e-12
    g_kl = nnx.grad(_shadow_kl_loss)(actor, obs, z, u_ref)
    g_cfm = nnx.grad(_shadow_cfm_loss)(actor, obs, z, u_ref, t)
    p0 = jax.tree.map(jnp.copy, nnx.state(optim_cfm.model, nnx.Param))
    optim_cfm.update(g_cfm)  # actual practical update, momentum included
    dtheta = jax.tree.map(lambda a_, b_: a_ - b_, nnx.state(optim_cfm.model, nnx.Param), p0)

    n_gkl = jnp.sqrt(_tree_dot(g_kl, g_kl))
    n_gcfm = jnp.sqrt(_tree_dot(g_cfm, g_cfm))
    n_dt = jnp.sqrt(_tree_dot(dtheta, dtheta))
    dot_negkl_dt = -_tree_dot(g_kl, dtheta)

    # norm-matched KL comparator step (sec.2.5) on the plain actor clone
    scale = n_dt / (n_gkl + eps)
    params = nnx.state(model_kln, nnx.Param)
    nnx.update(model_kln, jax.tree.map(lambda p, g: p - scale * g, params, g_kl))

    # post-step fresh-target residuals (Mahalanobis, frozen sigma_k)
    out = {}
    for name, model in (("adam", optim_cfm.model), ("klnorm", model_kln)):
        e = solve(z, obs, model.__call__, model.solver_name, model.dt)
        logstd = model._get_logstd(obs, e)
        out[name] = jnp.mean(jnp.sum((e - u_ref) ** 2 / (2.0 * jnp.exp(2.0 * logstd)), axis=-1))

    return {
        "diag/shadow_cos_KL_CFM": _tree_dot(g_kl, g_cfm) / (n_gkl * n_gcfm + eps),
        "diag/shadow_cos_negKL_dAdam": dot_negkl_dt / (n_gkl * n_dt + eps),
        "diag/shadow_dot_negKL_dAdam": dot_negkl_dt,
        "diag/shadow_norm_gKL": n_gkl,
        "diag/shadow_norm_gCFM": n_gcfm,
        "diag/shadow_norm_dAdam": n_dt,
        "diag/shadow_R_adam": out["adam"],
        "diag/shadow_R_klnorm": out["klnorm"],
    }


_AGE_BINS = ((0, 1_000), (1_000, 4_000), (4_000, 16_000), (16_000, 64_000), (64_000, np.inf))


class Phase0Diagnostics:
    def __init__(self, diag_cfg, seed, act_dim, eta):
        self.cfg = diag_cfg
        self.act_dim = int(act_dim)
        self.eta = float(eta)
        self.n_ref = int(diag_cfg.n_ref)
        self._key = jax.random.key(int(seed) + 999)
        self._rng = np.random.default_rng(int(seed) + 999)
        self._fixed_obs = None  # fixed probe set (plan sec.9.1), captured at first heavy_probe
        self._fixed_z = None

    def _next_key(self):
        self._key, k = jax.random.split(self._key)
        return k

    def _replay_obs(self, buffer, n):
        idx = self._rng.integers(0, len(buffer), size=n)
        return jnp.asarray(buffer.obs_buffer[idx])

    # --- light per-update proxy (plan sec.5-7) -----------------------------

    def light_pre(self, actor, alpha, guidance_batch, buffer):
        assert actor.logstd_mode == "fixed", "Phase 0 diagnostics require fixed logstd mode"
        pre = {}
        if guidance_batch is not None:
            pre["R_before"], _ = _residuals(actor, guidance_batch.obs, guidance_batch.noise, guidance_batch.act)
        probe_obs = self._replay_obs(buffer, int(self.cfg.drift_probe_batch))
        pretanh, tangent = _sample_probe(actor, probe_obs, self._next_key())
        pre["probe_obs"] = probe_obs
        pre["pretanh"] = pretanh
        pre["tangent"] = tangent
        pre["ell"] = _flow_logprob(actor, probe_obs, pretanh, tangent)
        pre["alpha"] = alpha()
        pre["params"] = jax.tree.map(jnp.copy, nnx.state(actor, nnx.Param))
        return pre

    def light_post(self, actor, alpha, guidance_batch, guidance_buffer, step, pre):
        out = {}
        if guidance_batch is not None:
            r_after, r_kl = _residuals(actor, guidance_batch.obs, guidance_batch.noise, guidance_batch.act)
            r_before = float(pre["R_before"])
            out["diag/R_before"] = r_before
            out["diag/R_after"] = float(r_after)
            out["diag/realization"] = 1.0 - float(r_after) / (r_before + 1e-12)
            out["diag/R_KL"] = float(r_kl)
            # buffer metadata (outside jit); -1 = unknown provenance, excluded
            idx = np.asarray(guidance_batch.idx)
            created = guidance_buffer.created_buffer[idx]
            valid = created >= 0
            if valid.any():
                ages = (step - created[valid]).astype(np.float64)
                out["diag/buffer_age_mean"] = float(ages.mean())
                out["diag/buffer_age_median"] = float(np.median(ages))
                out["diag/buffer_age_p95"] = float(np.percentile(ages, 95))
                out["diag/buffer_frac_age_le_1k"] = float(np.mean(ages <= 1000))
            # sample() already counted this draw -> subtract 1 for "previously reused"
            out["diag/buffer_reuse_mean"] = float(np.mean(guidance_buffer.reuse_buffer[idx] - 1))
        # reward drift, CRN: same pretanh actions + same tangent as light_pre
        ell1 = _flow_logprob(actor, pre["probe_obs"], pre["pretanh"], pre["tangent"])
        a0, a1 = pre["alpha"], alpha()
        d = jnp.abs(ell1 - pre["ell"])
        # raw D_logpi (phase1_plan.md sec.2.1) alongside the alpha-scaled D_policy
        out["diag/drift_raw_mean"] = float(jnp.mean(d))
        out["diag/drift_raw_median"] = float(jnp.median(d))
        out["diag/drift_raw_p95"] = float(jnp.percentile(d, 95))
        out["diag/drift_mean"] = float(a0 * jnp.mean(d))
        out["diag/drift_median"] = float(a0 * jnp.median(d))
        out["diag/drift_p95"] = float(a0 * jnp.percentile(d, 95))
        out["diag/drift_max"] = float(a0 * jnp.max(d))
        out["diag/drift_full"] = float(jnp.mean(jnp.abs(a1 * ell1 - a0 * pre["ell"])))
        # parameter displacement (plan sec.6.4)
        params_now = nnx.state(actor, nnx.Param)
        sq = jax.tree.map(lambda a, b: jnp.sum((a - b) ** 2), params_now, pre["params"])
        dtheta = jnp.sqrt(sum(jax.tree.leaves(sq)))
        norm0 = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda p: jnp.sum(p**2), pre["params"]))))
        out["diag/dtheta"] = float(dtheta)
        out["diag/dtheta_rel"] = float(dtheta / (norm0 + 1e-12))
        return out

    # --- periodic probes (plan sec.8-11) -----------------------------------

    def heavy_probe(self, step, actor, critic, alpha, buffer):
        assert actor.logstd_mode == "fixed"
        B = int(self.cfg.probe_batch)
        if self._fixed_obs is None:
            self._fixed_obs = self._replay_obs(buffer, B)
            self._fixed_z = jax.random.normal(self._next_key(), (B, self.act_dim))
        obs_m = self._replay_obs(buffer, B)
        z_m = jax.random.normal(self._next_key(), (B, self.act_dim))

        out = {}
        u_ref_m = None
        for name, obs, z in (("fixed", self._fixed_obs, self._fixed_z), ("moving", obs_m, z_m)):
            u_ref, ess = _reference_target(actor, critic, alpha, obs, z, self._next_key(), self.n_ref, self.eta)
            e, e_kl = _residuals(actor, obs, z, u_ref)
            out[f"diag/E_current_{name}"] = float(e)
            out[f"diag/E_current_KL_{name}"] = float(e_kl)
            out[f"diag/ess_{name}"] = float(ess)
            if name == "moving":
                u_ref_m = u_ref
        # SNIS noise floor (plan sec.11): independent second estimate, moving probe
        u_ref_m2, _ = _reference_target(actor, critic, alpha, obs_m, z_m, self._next_key(), self.n_ref, self.eta)
        out["diag/E_SNIS"] = float(0.5 * _msn(u_ref_m - u_ref_m2))
        return out

    def staleness(self, step, actor, critic, alpha, guidance_buffer):
        assert actor.logstd_mode == "fixed"
        n = len(guidance_buffer)
        created = guidance_buffer.created_buffer[:n]
        valid = created >= 0
        ages = step - created
        out = {}
        for i, (lo, hi) in enumerate(_AGE_BINS):
            pool = np.nonzero(valid & (ages >= lo) & (ages < hi))[0]
            out[f"diag/stale_count_bin{i}"] = float(pool.size)
            if pool.size == 0:
                continue
            # exactly staleness_per_bin with replacement -> static shape, one jit compile
            idx = self._rng.choice(pool, size=int(self.cfg.staleness_per_bin), replace=True)
            obs = jnp.asarray(guidance_buffer.obs_buffer[idx])
            u_star = jnp.asarray(guidance_buffer.action_buffer[idx])
            z = jnp.asarray(guidance_buffer.noise_buffer[idx])
            u_ref, _ = _reference_target(actor, critic, alpha, obs, z, self._next_key(), self.n_ref, self.eta)
            e_fit, e_stale, e_cur = _staleness_errors(actor, obs, z, u_star, u_ref)
            out[f"diag/stale_E_fit_bin{i}"] = float(e_fit)
            out[f"diag/stale_E_stale_bin{i}"] = float(e_stale)
            out[f"diag/stale_E_current_bin{i}"] = float(e_cur)
        return out

    # --- sparse KL-CFM alignment shadow (phase1_plan.md sec.2.5) -----------

    def kl_shadow(self, step, actor, critic, alpha, actor_optim, buffer):
        assert actor.logstd_mode == "fixed"
        B = int(self.cfg.shadow_batch)
        obs = self._replay_obs(buffer, B)
        z = jax.random.normal(self._next_key(), (B, self.act_dim))
        u_ref, _ = _reference_target(actor, critic, alpha, obs, z, self._next_key(), self.n_ref, self.eta)
        t = jax.random.uniform(self._next_key(), (B, 1))
        optim_cfm = nnx.clone(actor_optim)  # carries actor params + Adam state
        model_kln = nnx.clone(actor)
        out = _shadow_alignment(actor, optim_cfm, model_kln, obs, z, u_ref, t)
        return {k: float(v) for k, v in out.items()}


def main():
    import optax
    from types import SimpleNamespace

    from flag.actors import FlagActor
    from flag.critics import ScalarCritic
    from flag.alpha import ConstantAlpha
    from flag.buffers import ReplayBuffer, GuidanceBuffer, GuidanceBatch
    from flag.utils.nn import MLPConfig

    O, A, B = 3, 2, 8
    actor = FlagActor(
        observation_dim=O,
        action_dim=A,
        time_embed_dim=4,
        num_solver_steps=2,
        solver_name="euler",
        num_train_action_samples=2,
        num_eval_action_samples=2,
        mean_network_config=MLPConfig(hidden_features=(16,)),
        logstd_network_config=MLPConfig(hidden_features=(16,)),
        rngs=0,
        init_logstd=-1.0,
        logstd_mode="fixed",
    )
    critic = ScalarCritic(
        observation_dim=O,
        action_dim=A,
        has_target=False,
        num_networks=2,
        use_crossq_trick=False,
        network_config=MLPConfig(hidden_features=(16,)),
        rngs=1,
    )
    critic.eval()  # steady state between updates
    alpha = ConstantAlpha(0.2)
    optim = getattr(nnx, "ModelAndOptimizer", nnx.Optimizer)(actor, optax.adam(3e-4), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(size=64, obs_dim=O, act_dim=A, batch_size=8)
    for _ in range(64):
        buffer.add(
            rng.normal(size=O).astype(np.float32),
            rng.normal(size=A).astype(np.float32),
            0.0,
            rng.normal(size=O).astype(np.float32),
            False,
        )
    gbuf = GuidanceBuffer(size=32, obs_dim=O, act_dim=A, batch_size=8)
    gbuf.add(rng.normal(size=(16, O)), np.tanh(rng.normal(size=(16, A))), rng.normal(size=(16, A)))  # step=-1
    gbuf.add(rng.normal(size=(16, O)), np.tanh(rng.normal(size=(16, A))), rng.normal(size=(16, A)), step=50)
    assert (gbuf.created_buffer[:16] == -1).all() and (gbuf.created_buffer[16:] == 50).all()

    idx = rng.integers(0, len(gbuf), size=B)
    gbuf.reuse_buffer[idx] += 1  # mimic sample() bookkeeping
    gb = GuidanceBatch(
        obs=jnp.asarray(gbuf.obs_buffer[idx]),
        act=jnp.asarray(gbuf.action_buffer[idx]),
        noise=jnp.asarray(gbuf.noise_buffer[idx]),
        idx=jnp.asarray(idx),
    )

    cfg = SimpleNamespace(
        enabled=True,
        light_every=1,
        drift_probe_batch=B,
        probe_batch=B,
        n_ref=4,
        probe_interval=1,
        staleness_interval=1,
        staleness_per_bin=4,
        shadow_interval=1,
        shadow_batch=B,
        shadow_seeds=[0, 1, 2],
    )
    diag = Phase0Diagnostics(cfg, seed=0, act_dim=A, eta=0.1)

    def finite(d):
        return all(np.isfinite(v) for v in d.values())

    # light pre -> fake one SGD step -> light post
    pre = diag.light_pre(actor, alpha, gb, buffer)
    params = nnx.state(actor, nnx.Param)
    nnx.update(actor, jax.tree.map(lambda p: p - 1e-3, params))
    out = diag.light_post(actor, alpha, gb, gbuf, 100, pre)
    assert finite(out), out
    assert out["diag/realization"] <= 1.0 + 1e-6
    assert out["diag/dtheta"] > 0.0
    assert out["diag/buffer_age_mean"] == 50.0  # -1 entries excluded

    # reference target
    u_ref, ess = _reference_target(actor, critic, alpha, gb.obs, gb.noise, jax.random.key(7), 4, 0.1)
    assert np.isfinite(np.asarray(u_ref)).all() and u_ref.shape == (B, A)
    assert 1.0 - 1e-6 <= float(ess) <= 4 + 1 + 1e-6

    def snap(*modules):
        return [jax.device_get(nnx.state(m)) for m in modules]

    def _np(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            x = jax.random.key_data(x)  # rng counters compare via raw key data
        return np.asarray(x)

    def assert_same(before, after):
        for b, a in zip(before, after):
            lb, la = jax.tree.leaves(b), jax.tree.leaves(a)
            assert len(lb) == len(la)
            for x, y in zip(lb, la):
                assert np.array_equal(_np(x), _np(y))

    # heavy_probe / staleness / kl_shadow leave the FULL live state bitwise unchanged
    before = snap(actor, critic, alpha)
    hp = diag.heavy_probe(100, actor, critic, alpha, buffer)
    assert finite(hp), hp
    assert 1.0 - 1e-6 <= hp["diag/ess_moving"] <= cfg.n_ref + 1 + 1e-6
    assert_same(before, snap(actor, critic, alpha))

    st = diag.staleness(100, actor, critic, alpha, gbuf)
    assert finite(st), st
    assert st["diag/stale_count_bin0"] > 0 and "diag/stale_E_fit_bin0" in st
    assert_same(before, snap(actor, critic, alpha))

    before_opt = snap(optim)
    ks = diag.kl_shadow(100, actor, critic, alpha, optim, buffer)
    assert finite(ks), ks
    assert abs(ks["diag/shadow_cos_KL_CFM"]) <= 1.0 + 1e-5, ks
    assert abs(ks["diag/shadow_cos_negKL_dAdam"]) <= 1.0 + 1e-5, ks
    assert ks["diag/shadow_norm_gKL"] > 0 and ks["diag/shadow_norm_dAdam"] > 0, ks
    assert ks["diag/shadow_R_adam"] >= 0 and ks["diag/shadow_R_klnorm"] >= 0, ks
    assert_same(before, snap(actor, critic, alpha))
    assert_same(before_opt, snap(optim))

    # supervision-off path: reward drift + dtheta still logged
    pre2 = diag.light_pre(actor, alpha, None, buffer)
    out2 = diag.light_post(actor, alpha, None, gbuf, 101, pre2)
    assert "diag/R_before" not in out2 and finite(out2)
    assert "diag/drift_raw_mean" in out2

    # fresh-target/IS diag inside the actual actor loss (phase1_plan.md sec.2.2/2.4)
    from flag.losses.flag import flag_actor_loss_fn
    from flag.buffers import ReplayBatch

    rb = ReplayBatch(
        obs=jnp.asarray(buffer.obs_buffer[:B]),
        act=jnp.asarray(buffer.action_buffer[:B]),
        rwd=jnp.zeros((B,)),
        next_obs=jnp.asarray(buffer.obs_buffer[:B]),
        done=jnp.zeros((B,)),
    )
    _, aux = flag_actor_loss_fn(actor, critic, alpha, 0.1, rb, diag=True)
    dm = aux[-1]
    assert dm is not None and finite(dm), dm
    assert 0.0 <= dm["diag/P_hit"] <= 1.0 and 0.0 <= dm["diag/p_better"] <= 1.0, dm
    assert dm["diag/R_fresh_before"] >= 0 and dm["diag/D_local"] >= 0, dm
    _, aux_off = flag_actor_loss_fn(actor, critic, alpha, 0.1, rb, diag=False)
    assert aux_off[-1] is None

    print("[diagnostics] self-check ok")


if __name__ == "__main__":
    main()
