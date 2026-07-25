from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.lax import stop_gradient as sg
import flax.nnx as nnx

from flag.alpha import Alpha, alpha_loss_fn, ConstantAlpha
from flag.actors import FlagActor
from flag.critics import ScalarCritic, DistributionalCritic
from flag.buffers import ReplayBatch, GuidanceBatch
from flag.utils.weight import get_weights
from flag.utils.math import two_hot
from flag.utils.flow import solve, solve_with_logprob_backward


def guidance_supervision_loss_fn(
    actor: FlagActor,
    batch: GuidanceBatch,
) -> Array:

    B = batch.obs.shape[0]
    A = batch.act.shape[1]

    x1 = sg(batch.act)

    x0 = sg(batch.noise)
    t = jax.random.uniform(actor.rngs['t'](), shape=(B, 1))

    xt = (1.0 - t) * x0 + t * x1
    vt = actor(t.ravel(), xt, batch.obs)
    return jnp.mean((vt - (x1 - x0)) ** 2)


def flag_actor_loss_fn(
    actor: FlagActor,
    critic: ScalarCritic | DistributionalCritic,
    alpha: Alpha,
    eta: float,
    batch: ReplayBatch,
    diag: bool = False,
):
    B = batch.obs.shape[0]
    S = actor.num_train_action_samples
    O = batch.obs.shape[1]
    A = batch.act.shape[1]
    critic.eval()

    outs = actor.get_action_logprob(batch.obs, return_tangent=diag)

    def selective_sg(x, apply_sg):
        return sg(x) if apply_sg else x

    sg_mask = (True, True, True, True, False, True) + ((True,) if diag else ())
    outs = jax.tree.map(selective_sg, outs, sg_mask)
    if diag:
        act, logp_gaussian, logp_flow, pretanh_a, logstd, flow_noise, flow_tangent = outs
    else:
        act, logp_gaussian, logp_flow, pretanh_a, logstd, flow_noise = outs

    obs_repeated = batch.obs[:, jnp.newaxis].repeat(S + 1, axis=1)
    obs_flat = obs_repeated.reshape(-1, O)
    act_flat = act[:, : S + 1].reshape(-1, A)

    q = jnp.mean(critic(obs_flat, act_flat, use_target=False), axis=0)

    q = q.reshape(B, S + 1, 1)
    q_sg = sg(q)

    weights, adv = get_weights(
        q_sg,
        logp_gaussian,
        logp_flow,
        alpha,
        eta,
    )

    mu_anchor = pretanh_a[:, -1]  # flow endpoint T_theta(s,z): anchor entry (eps=0), pre-clip
    pretanh_a = jnp.clip(pretanh_a, -3.8, 3.8)
    u_star = sg(weights * pretanh_a[:, : S + 1]).sum(axis=1)

    target_means = u_star

    t = jax.random.uniform(actor.rngs['t'](), shape=(B, 1))
    xt = (1 - t) * flow_noise + t * target_means
    vt = actor(t.ravel(), xt, batch.obs)
    flow_loss = jnp.mean(((vt - (target_means - flow_noise)) ** 2))
    adv_log = sg(adv)

    actor_loss = flow_loss

    variance = sg(jnp.mean(jnp.var(act[:, :-1], axis=1)))

    diag_metrics = None
    if diag:
        # phase1_plan.md sec.2.2/2.4 — all sg'd, no gradient contribution, no extra RNG.
        # Per-candidate scaled advantage a_i = A_i/alpha. Each candidate is paired
        # with the duplicated anchor evaluated using the same Hutchinson tangent,
        # exactly as in get_weights' f_dist - f_mean.
        a_i = sg(
            (q_sg[:, :-1] - q_sg[:, -1:]) / alpha()
            - (logp_flow[:, :S] - logp_flow[:, S:])
        )[..., 0]
        std = sg(jnp.exp(logstd))
        # target augmented value: one extra critic eval on the target action only.
        q_tgt = sg(jnp.mean(critic(batch.obs, jnp.tanh(u_star), use_target=False), axis=0))
        d_u = (u_star - sg(mu_anchor)) / std  # logstd/std are (B, A)

        # Evaluate target and anchor under the same flow-policy density estimator.
        # Reuse candidate 0's existing Rademacher tangent, so diagnostics consume no
        # additional actor RNG. The matching anchor logp is already at index S.
        target_tangent = flow_tangent[:, 0]
        _, target_logp_pre = solve_with_logprob_backward(
            u_star,
            target_tangent,
            batch.obs,
            actor.__call__,
            actor.solver_name,
            actor.dt,
        )
        target_action = jnp.clip(jnp.tanh(u_star), -0.999, 0.999)
        target_correction = jnp.sum(
            jnp.log(1.0 - target_action**2 + 1e-6),
            axis=-1,
            keepdims=True,
        )
        target_logp_flow = sg(target_logp_pre - target_correction)
        anchor_logp_flow = sg(logp_flow[:, S])
        dF = (q_tgt - q_sg[:, -1]) - alpha() * (target_logp_flow - anchor_logp_flow)
        diag_metrics = {
            "diag/R_fresh_before": jnp.mean(jnp.sum((sg(mu_anchor) - u_star) ** 2, axis=-1)),
            "diag/P_hit": jnp.mean(jnp.max(a_i, axis=1) > 0),
            "diag/p_better": jnp.mean(a_i > 0),
            "diag/dF_target": jnp.mean(dF),
            "diag/dQ_target": jnp.mean(q_tgt - q_sg[:, -1]),
            "diag/D_local": jnp.mean(jnp.linalg.norm(d_u, axis=-1)) / jnp.sqrt(A),
        }

    return actor_loss, (logp_gaussian[:, :-1], logp_flow[:, :-1], variance, adv_log, u_star, flow_noise, diag_metrics)


def scalar_critic_loss_fn(
    actor: FlagActor,
    critic: ScalarCritic,
    alpha: Alpha,
    gamma: float,
    batch: ReplayBatch,
) -> Array:
    obs = batch.obs
    act = batch.act
    rwd = batch.rwd[..., jnp.newaxis]
    not_done = (1.0 - batch.done)[..., jnp.newaxis]
    next_obs = batch.next_obs
    use_target = critic.has_target

    next_act, next_logp_slice, next_logp_flow, _, _, _ = actor.get_action_logprob(next_obs, is_single=True)

    if not critic.use_crossq_trick:
        critic.eval()
        next_qs = critic(next_obs, next_act, use_target=use_target)
        critic.train()
        current_qs = critic(obs, act, use_target=False)
        critic.eval()
    else:
        critic.train()
        catted_obs = jnp.concatenate([obs, next_obs], axis=0)
        catted_act = jnp.concatenate([act, next_act], axis=0)
        catted_qs = critic(catted_obs, catted_act, use_target=use_target)
        current_qs, next_qs = jnp.split(catted_qs, 2, axis=1)
        critic.eval()

    next_q = jnp.mean(next_qs, axis=0)

    entropy = alpha() * next_logp_flow
    target = sg(rwd + not_done * gamma * (next_q - entropy))

    critic_loss = jnp.mean((current_qs - target) ** 2)
    return critic_loss


def distributional_critic_loss_fn(
    actor: FlagActor,
    critic: DistributionalCritic,
    alpha: Alpha,
    gamma: float,
    batch: ReplayBatch,
    ent_coeff: float = 0.0,
) -> tuple[Array, tuple[Array, Array, Array, Array, Array, Array]]:

    use_target = critic.has_target

    obs = batch.obs
    act = batch.act
    rwd = batch.rwd[..., jnp.newaxis]
    not_done = (1.0 - batch.done)[..., jnp.newaxis]
    next_obs = batch.next_obs

    next_act, next_logp_slice, next_logp_flow, _, _, _ = actor.get_action_logprob(next_obs, is_single=True)

    if not critic.use_crossq_trick:
        critic.eval()
        next_q_logits = critic.get_logits(next_obs, next_act, use_target=use_target)
        critic.train()
        current_q_logits = critic.get_logits(obs, act, use_target=False)
        critic.eval()
    else:
        critic.train()
        catted_obs = jnp.concatenate([obs, next_obs], axis=0)
        catted_act = jnp.concatenate([act, next_act], axis=0)
        catted_q_logits = critic.get_logits(catted_obs, catted_act, use_target=use_target)
        current_q_logits, next_q_logits = jnp.split(catted_q_logits, 2, axis=1)
        critic.eval()

    target_dist = []
    entropy = alpha() * next_logp_flow

    target = rwd + not_done * gamma * (critic.z_atoms.reshape(1, -1) - entropy)
    target = jnp.clip(target, critic.z_atoms[0], critic.z_atoms[-1])
    weights = two_hot(target[..., jnp.newaxis], critic.z_atoms)

    def _apply_projection(logits: Array) -> Array:
        dist = jax.nn.softmax(logits, axis=-1)
        projection = jnp.sum(dist[..., jnp.newaxis] * weights, axis=1)
        return projection

    projected_dists = jax.vmap(_apply_projection)(next_q_logits)
    target_dist = sg(jnp.mean(projected_dists, axis=0))
    next_q_values = jnp.mean(jnp.sum(target_dist * critic.z_atoms.reshape(1, -1), axis=-1))

    def _logit_to_q_values(logits: Array) -> Array:
        return jnp.sum(jax.nn.softmax(logits, axis=-1) * critic.z_atoms.reshape(1, -1), axis=-1)

    current_q_values = jax.vmap(_logit_to_q_values)(current_q_logits)
    min_qf_pi = jnp.min(sg(current_q_values), axis=0)

    def compute_single_loss(logits):
        current_probs = jax.nn.softmax(logits, axis=-1)
        current_logprobs = jax.nn.log_softmax(logits, axis=-1)

        ce_loss = -jnp.mean(jnp.sum(target_dist * current_logprobs, axis=-1))

        ent_reg = ent_coeff * jnp.mean(jnp.sum(current_probs * current_logprobs, axis=-1))
        return ce_loss + ent_reg

    critic_loss = jnp.sum(jax.vmap(compute_single_loss)(current_q_logits))

    return critic_loss, (min_qf_pi, next_q_values, -sg(entropy.mean()))


@nnx.jit(
    static_argnames=[
        "target_ent_alpha",
        "gamma",
        "tau",
        "eta",
        "gradient_steps",
        "update_actor_mask",
        "enable_supervision",
        "ent_coeff",
        "diag_metrics",
    ]
)
def gaussian_flow_step(
    batch: ReplayBatch,
    guidance_batch: GuidanceBatch,
    actor: FlagActor,
    critic: ScalarCritic | DistributionalCritic,
    alpha: Union[Alpha, ConstantAlpha],
    actor_optim: nnx.Optimizer,
    critic_optim: nnx.Optimizer,
    alpha_optim: nnx.Optimizer | None,
    target_ent_alpha: float,
    gamma: float,
    tau: float,
    eta: float,
    gradient_steps: int,
    update_actor_mask: tuple[bool, ...],
    enable_supervision: bool,
    supervision_coef: Array,
    ent_coeff: float = 0.0,
    diag_metrics: bool = False,
) -> tuple[dict[str, Array], Array | None, Array | None, Array | None]:

    def reshape_batch(x):
        return x.reshape((gradient_steps, -1) + x.shape[1:])

    mini_batches = jax.tree.map(reshape_batch, batch)
    mini_guidance_batches = jax.tree.map(reshape_batch, guidance_batch)

    critic_metrics_list = []
    actor_metrics_list = []
    refined_actions_list = []
    refined_obs_list = []
    refined_noise_list = []
    supervision_loss_list = []

    for i in range(gradient_steps):
        b = jax.tree.map(lambda x: x[i], mini_batches)
        fb = jax.tree.map(lambda x: x[i], mini_guidance_batches)

        current_q_values = jnp.nan
        next_q_values = jnp.nan
        debug_entropy = jnp.nan

        if isinstance(critic, DistributionalCritic):
            critic_loss_fn = distributional_critic_loss_fn
            critic_value_grad_fn = nnx.value_and_grad(critic_loss_fn, argnums=1, has_aux=True)
            (critic_loss, (current_q_values, next_q_values, debug_entropy)), critic_grads = critic_value_grad_fn(
                actor, critic, alpha, gamma, b, ent_coeff
            )

        elif isinstance(critic, ScalarCritic):
            critic_loss_fn = scalar_critic_loss_fn
            critic_value_grad_fn = nnx.value_and_grad(critic_loss_fn, argnums=1)
            critic_loss, critic_grads = critic_value_grad_fn(actor, critic, alpha, gamma, b)

        critic_optim.update(critic_grads)
        if critic.has_target:
            critic.update_target(tau)

        critic_metrics_list.append(
            {
                "critic_loss": critic_loss,
                "current_q_values": jnp.nan_to_num(current_q_values, nan=0.0),
                "next_q_values": jnp.nan_to_num(next_q_values, nan=0.0),
                "debug_entropy": jnp.nan_to_num(debug_entropy, nan=0.0),
            }
        )

        if update_actor_mask[i]:

            def _combined_actor_loss_fn(
                actor: FlagActor,
                critic: ScalarCritic | DistributionalCritic,
                alpha: Alpha,
                eta: float,
                b: ReplayBatch,
                fb: GuidanceBatch,
            ):
                if enable_supervision:
                    supervision_loss_raw = guidance_supervision_loss_fn(actor, fb)
                    supervision_loss = supervision_coef * supervision_loss_raw
                else:
                    supervision_loss_raw = jnp.array(0.0, dtype=jnp.float32)
                    supervision_loss = jnp.array(0.0, dtype=jnp.float32)
                actor_loss, aux = flag_actor_loss_fn(actor, critic, alpha, eta, b, diag=diag_metrics)
                total_loss = actor_loss + supervision_loss
                return total_loss, (actor_loss, supervision_loss_raw, aux)

            actor_value_grad_fn = nnx.value_and_grad(_combined_actor_loss_fn, has_aux=True)
            (total_loss, (actor_loss, supervision_loss_raw, aux)), actor_grads = actor_value_grad_fn(
                actor, critic, alpha, eta, b, fb
            )
            logp_gaussian, logp_flow, variance, advantage, refined_actions, refined_noise, diag_aux = aux
            actor_optim.update(actor_grads)
            if diag_metrics:
                # phase1_plan.md sec.2.2: fresh-target residual immediately after this
                # optimizer mini-step, before the next one. Same (s, z, u_star) as the loss.
                e_after = solve(refined_noise, b.obs, actor.__call__, actor.solver_name, actor.dt)
                logstd_after = actor._get_logstd(b.obs, e_after)
                r_fresh_after = jnp.mean(jnp.sum((e_after - refined_actions) ** 2, axis=-1))
                diag_aux = dict(diag_aux)
                diag_aux["diag/R_fresh_after"] = r_fresh_after
                diag_aux["diag/realization_fresh"] = 1.0 - r_fresh_after / (diag_aux["diag/R_fresh_before"] + 1e-12)
                diag_aux["diag/R_fresh_KL_after"] = jnp.mean(
                    jnp.sum((e_after - refined_actions) ** 2 / (2.0 * jnp.exp(2.0 * logstd_after)), axis=-1)
                )
            refined_actions_list.append(refined_actions)
            refined_obs_list.append(b.obs)
            refined_noise_list.append(refined_noise)
            supervision_loss_list.append(supervision_loss_raw)

            if isinstance(alpha, Alpha):
                alpha_value_grad_fn = nnx.value_and_grad(alpha_loss_fn)
                alpha_loss, alpha_grads = alpha_value_grad_fn(alpha, logp_flow, target_ent_alpha)
                alpha_optim.update(alpha_grads)
            else:
                alpha_loss = jnp.array(0.0)

            actor_metrics_list.append(
                {
                    "actor_loss": actor_loss,
                    "alpha_loss": alpha_loss,
                    "alpha": alpha(),
                    "conditional_entropy": -jnp.mean(logp_gaussian),
                    "mean_marginal_entropy": -jnp.mean(logp_flow),
                    "variance": variance,
                    "advantage": advantage,
                    **(diag_aux or {}),
                }
            )
        else:

            if enable_supervision:

                def _scaled_supervision_loss_fn(actor: FlagActor, fb: GuidanceBatch):
                    raw = guidance_supervision_loss_fn(actor, fb)
                    return supervision_coef * raw, raw

                supervision_value_grad_fn = nnx.value_and_grad(_scaled_supervision_loss_fn, has_aux=True)
                (scaled_loss, raw_loss), supervision_grads = supervision_value_grad_fn(actor, fb)
                actor_optim.update(supervision_grads)
                supervision_loss_list.append(raw_loss)
            else:
                supervision_loss_list.append(jnp.array(0.0, dtype=jnp.float32))

    final_metrics = {}
    for key in critic_metrics_list[0].keys():
        final_metrics[key] = jnp.mean(jnp.stack([m[key] for m in critic_metrics_list]))

    raw_supervision_loss = jnp.mean(jnp.stack(supervision_loss_list))
    final_metrics["supervision_loss"] = supervision_coef * raw_supervision_loss
    final_metrics["supervision_coef"] = supervision_coef

    if actor_metrics_list:
        for key in actor_metrics_list[0].keys():
            final_metrics[key] = jnp.mean(jnp.stack([m[key] for m in actor_metrics_list]))
    else:

        actor_keys = [
            "actor_loss",
            "alpha_loss",
            "alpha",
            "conditional_entropy",
            "mean_marginal_entropy",
            "variance",
            "advantage",
        ]
        for key in actor_keys:
            final_metrics[key] = jnp.nan

    if refined_actions_list:
        refined_actions = jnp.concatenate(refined_actions_list, axis=0)
        refined_obs = jnp.concatenate(refined_obs_list, axis=0)
        refined_noise = jnp.concatenate(refined_noise_list, axis=0)
    else:
        refined_actions = None
        refined_obs = None
        refined_noise = None

    return final_metrics, refined_obs, refined_actions, refined_noise
