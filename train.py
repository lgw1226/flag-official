import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax")

from time import time
from typing import Any, Callable, Union
from collections import deque
from functools import partial
import random
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# flax >= 0.12 makes Modules pytrees, which rejects plain lists of submodules (MLP.layers, BaseCritic.networks)
os.environ.setdefault("FLAX_PYTREE_MODULE", "false")

import jax
import jax.numpy as jnp
from jax import Array
from jax.lax import stop_gradient as sg
import flax.nnx as nnx

# flax >= 0.11 moved the model out of nnx.Optimizer; ModelAndOptimizer keeps the old model-holding behaviour
NNXOptimizer = getattr(nnx, "ModelAndOptimizer", nnx.Optimizer)

import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from flag.actors import FlagActor
from flag.critics import ScalarCritic, DistributionalCritic
from flag.alpha import Alpha
from flag.losses.flag import gaussian_flow_step
from flag.utils.wrappers.normalize_env import (
    Normalizer,
    NormalizeJAXEnvWrapper,
    NormalizeJAXVectorEnvWrapper,
)
from flag.buffers import ReplayBuffer, GuidanceBuffer, GuidanceBatch
from flag.utils.std_scheduler import LogstdScheduler
from flag.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest
from flag.utils.diagnostics import Phase0Diagnostics

from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


def evaluate(
    env: Any,
    policy: Callable[[Array], Array],
    num_episodes: int,
    key: Array,
) -> dict[str, float]:
    total_steps = 0
    total_reward = 0.0
    for _ in range(num_episodes):
        key, eval_key = jax.random.split(key)
        obs, info = env.reset(eval_key)
        done = False
        while not done:
            action = policy(observation=obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            total_reward += reward
            done = terminated or truncated
    avg_reward = total_reward / num_episodes
    avg_steps = total_steps / num_episodes
    return {
        "eval/episode_return": avg_reward,
        "eval/episode_length": avg_steps,
    }


def evaluate_vectorized_env(
    vec_env: Any,
    policy: Callable[[Array], Array],
    key: Array,
) -> dict[str, float]:
    obs, info = vec_env.reset(key)

    n_envs = vec_env.num_envs
    episode_rewards = np.zeros(n_envs)
    episode_lenghts = np.zeros(n_envs)
    episode_success = np.zeros(n_envs, dtype=bool)

    finished_mask = np.zeros(n_envs, dtype=bool)
    final_rewards = np.zeros(n_envs)
    final_lengths = np.zeros(n_envs)
    final_success = np.zeros(n_envs, dtype=bool)

    while not np.all(finished_mask):

        action = policy(observation=obs)

        next_obs, rewards, terminateds, truncateds, infos = vec_env.step(action)

        not_done = ~finished_mask
        episode_rewards[not_done] += rewards[not_done]
        episode_lenghts[not_done] += 1

        if "solved" in infos:
            episode_success[not_done] = episode_success[not_done] | infos["solved"][not_done]

        dones = terminateds | truncateds

        newly_done = dones & not_done
        if np.any(newly_done):
            final_rewards[newly_done] = episode_rewards[newly_done]
            final_lengths[newly_done] = episode_lenghts[newly_done]
            final_success[newly_done] = episode_success[newly_done]
            finished_mask[newly_done] = True

        obs = next_obs

    results = {
        "eval/episode_return": float(np.mean(final_rewards)),
        "eval/episode_length": float(np.mean(final_lengths)),
    }

    if "solved" in infos:
        results["eval/success_rate"] = float(np.mean(final_success))

    return results


@nnx.jit
def train_policy(actor: FlagActor, observation: Array) -> Array:
    if observation.ndim == 1:
        observation = observation[jnp.newaxis, :]
    return actor.get_action(observation, evaluate=False)[0]


@nnx.jit
def evaluate_policy(actor: FlagActor, critic: Union[ScalarCritic, DistributionalCritic], observation: Array) -> Array:
    is_single = observation.ndim == 1
    if is_single:
        observation = observation[jnp.newaxis, :]

    critic.eval()

    if is_single:
        actions = actor.get_action(observation, evaluate=True)[0]
        obs_repeated = jnp.repeat(observation, actions.shape[0], axis=0)
        if critic.has_target:
            q = sg(critic(obs_repeated, actions, use_target=True)).squeeze(-1)
        else:
            q = sg(critic(obs_repeated, actions, use_target=False)).squeeze(-1)
        q = jnp.mean(q, axis=0)
        best_sample_idx = jnp.argmax(q)
        return actions[best_sample_idx]
    else:
        batch_size = observation.shape[0]
        all_actions = actor.get_action(observation, evaluate=True)
        num_samples_per_obs = all_actions.shape[1]
        best_actions = []
        for i in range(batch_size):
            obs_i = observation[i : i + 1]
            actions_i = all_actions[i]
            obs_repeated = jnp.repeat(obs_i, num_samples_per_obs, axis=0)

            if critic.has_target:
                q = sg(critic(obs_repeated, actions_i, use_target=True)).squeeze(-1)
            else:
                q = sg(critic(obs_repeated, actions_i, use_target=False)).squeeze(-1)
            q = jnp.mean(q, axis=0)
            best_sample_idx = jnp.argmax(q)
            best_actions.append(actions_i[best_sample_idx])

        return jnp.stack(best_actions)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    assert cfg.eval_interval % cfg.log_interval == 0

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.env_id.startswith("dm_control/"):
        import flag.utils.wrappers.dmcontrol as dmcontrol_wrappers

        env, eval_env = dmcontrol_wrappers.make_env(cfg)
        cfg.critic.max_val = 200.0
        cfg.critic.min_val = -200.0
    elif cfg.env_id.startswith("myo"):
        import flag.utils.wrappers.myosuite as myosuite_wrappers

        env, eval_env = myosuite_wrappers.make_env(cfg)
        cfg.critic.max_val = 3600.0
        cfg.critic.min_val = -3600.0
    else:
        import flag.utils.wrappers.mujoco as mujoco_wrappers

        env, eval_env = mujoco_wrappers.make_env(cfg)
        cfg.critic.max_val = 1600.0
        cfg.critic.min_val = -1600.0

    if cfg.normalize_env:
        normalizer = Normalizer(
            obs_shape=(env.obs_dim,),
            gamma=float(cfg.normalize_gamma),
            epsilon=float(cfg.normalize_eps),
        )

        env = NormalizeJAXEnvWrapper(
            env,
            normalizer=normalizer,
            training=True,
            norm_obs=bool(cfg.normalize_obs),
            norm_reward=bool(cfg.normalize_reward),
            clip_obs=float(cfg.normalize_clip_obs),
            clip_reward=float(cfg.normalize_clip_reward),
        )

        eval_env = NormalizeJAXVectorEnvWrapper(
            eval_env,
            normalizer=normalizer,
            training=False,
            norm_obs=bool(cfg.normalize_obs),
            norm_reward=bool(cfg.normalize_reward_eval),
            clip_obs=float(cfg.normalize_clip_obs),
            clip_reward=float(cfg.normalize_clip_reward),
        )
    seed = cfg.seed
    key = jax.random.key(seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    actor_seed = int(jax.random.randint(actor_key, (), 0, 2**31 - 1))
    critic_seed = int(jax.random.randint(critic_key, (), 0, 2**31 - 1))

    actor = instantiate(
        cfg.actor, observation_dim=env.obs_dim, action_dim=env.act_dim, rngs=actor_seed, _convert_="all"
    )
    actor_optim_tx = instantiate(cfg.actor_optim.tx)
    actor_optim = NNXOptimizer(
        actor,
        actor_optim_tx,
        wrt=nnx.Param,
    )

    critic = instantiate(
        cfg.critic, observation_dim=env.obs_dim, action_dim=env.act_dim, rngs=critic_seed, _convert_="all"
    )
    critic.train()
    critic_optim_tx = instantiate(cfg.critic_optim.tx)
    critic_optim = NNXOptimizer(
        critic,
        critic_optim_tx,
        wrt=nnx.Param,
    )

    alpha = instantiate(cfg.alpha)
    if isinstance(alpha, Alpha):
        alpha_optim_tx = instantiate(cfg.alpha_optim.tx)
        alpha_optim = NNXOptimizer(
            alpha,
            alpha_optim_tx,
            wrt=nnx.Param,
        )
    else:
        alpha_optim = None

    step_fn = partial(
        gaussian_flow_step,
        target_ent_alpha=cfg.alpha_target_entropy_coeff * env.act_dim,
        gamma=cfg.discount,
        tau=cfg.critic_ema_rate,
        eta=cfg.eta,
        gradient_steps=cfg.utd,
        ent_coeff=cfg.ent_coeff,
        diag_metrics=bool(cfg.diag.enabled),
    )

    critic_metrics = nnx.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        current_q_values=nnx.metrics.Average("current_q_values"),
        next_q_values=nnx.metrics.Average("next_q_values"),
        debug_entropy=nnx.metrics.Average("debug_entropy"),
        supervision_loss=nnx.metrics.Average("supervision_loss"),
        supervision_coef=nnx.metrics.Average("supervision_coef"),
        train_time=nnx.metrics.Average("train_time"),
    )

    actor_metrics = nnx.MultiMetric(
        actor_loss=nnx.metrics.Average("actor_loss"),
        alpha_loss=nnx.metrics.Average("alpha_loss"),
        alpha=nnx.metrics.Average("alpha"),
        conditional_entropy=nnx.metrics.Average("conditional_entropy"),
        mean_marginal_entropy=nnx.metrics.Average("mean_marginal_entropy"),
        variance=nnx.metrics.Average("variance"),
        advantage=nnx.metrics.Average("advantage"),
    )

    logstd_scheduler = None
    if actor.logstd_mode == "fixed":
        logstd_scheduler = LogstdScheduler(
            init_logstd=float(cfg.actor.init_logstd),
            final_logstd=float(cfg.logstd_min),
            warmup_steps=int(cfg.logstd_warmup_steps),
            decay_steps=int(cfg.logstd_decay_steps),
            schedule_type=str(cfg.logstd_scheduler),
        )

    buffer: ReplayBuffer = instantiate(cfg.buffer, obs_dim=env.obs_dim, act_dim=env.act_dim)
    guidance_buffer: GuidanceBuffer = instantiate(cfg.guidance_buffer, obs_dim=env.obs_dim, act_dim=env.act_dim)

    diag = Phase0Diagnostics(cfg.diag, seed=cfg.seed, act_dim=env.act_dim, eta=cfg.eta) if cfg.diag.enabled else None

    episode_reward = 0.0
    episode_return = 0.0
    episode_length = 0
    episode_cnt = 0
    WINDOW_SIZE = 100
    episode_return_window = deque(maxlen=WINDOW_SIZE)
    episode_length_window = deque(maxlen=WINDOW_SIZE)

    return_sum = 0.0
    length_sum = 0.0
    total_updates = 0

    ckpt_dir = os.path.abspath(cfg.ckpt_dir)
    ckpt_modules = {
        "actor": actor,
        "critic": critic,
        "alpha": alpha,
        "actor_optim": actor_optim,
        "critic_optim": critic_optim,
        "alpha_optim": alpha_optim,
    }
    ckpt_buffers = {"buffer": buffer, "guidance_buffer": guidance_buffer}

    start_step = 0
    wandb_id = None
    resume_path = find_latest(ckpt_dir) if cfg.resume == "latest" else cfg.resume
    if resume_path:
        last_step, extra = load_checkpoint(resume_path, ckpt_modules, ckpt_buffers)
        # phase1_plan §3: a checkpoint must only resume a run with the same N
        ckpt_n = extra.get("num_train_action_samples")
        if ckpt_n is not None and int(ckpt_n) != int(actor.num_train_action_samples):
            raise ValueError(f"checkpoint N={ckpt_n} != configured N={actor.num_train_action_samples}")
        start_step = last_step + 1
        key = jax.random.wrap_key_data(extra["key_data"])
        episode_cnt = extra["episode_cnt"]
        total_updates = extra["total_updates"]
        episode_return_window = extra["episode_return_window"]
        episode_length_window = extra["episode_length_window"]
        return_sum = float(sum(episode_return_window))
        length_sum = float(sum(episode_length_window))
        wandb_id = extra["wandb_id"]
        if cfg.normalize_env:
            normalizer.__dict__.update(extra["normalizer"].__dict__)  # wrappers hold a reference: restore in place
        print(f"[ckpt] resumed {resume_path} @ step {last_step} (logstd={extra['logstd']})")

    wandb.init(
        **cfg.wandb,
        id=wandb_id,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    key, reset_key = jax.random.split(key)
    obs, info = env.reset(reset_key)
    pbar = tqdm(range(start_step, cfg.total_steps), desc="Training", ncols=100)

    actor_updates_since_log = 0
    for step in pbar:
        if step <= cfg.start_steps:
            key, exploration_key = jax.random.split(key)
            act = env.sample_random_action(exploration_key)
        else:
            act = train_policy(actor, obs)

        act_np = np.array(act)
        next_obs, reward, terminated, truncated, info = env.step(act_np)
        raw_reward = info.get("raw_reward", reward)
        buffer.add(obs, act_np, reward, next_obs, terminated)
        episode_reward += raw_reward
        episode_length += 1
        obs = next_obs

        if terminated or truncated:
            if episode_cnt % 100 == 0:
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"🎯 Episode {episode_cnt} finished at step {step}")
                tqdm.write(f"   📈 Return: {float(episode_reward):>8.2f}")
                tqdm.write(f"   📏 Length: {episode_length:>8d}")
                tqdm.write(f"{'='*60}")

            episode_cnt += 1
            episode_return = episode_reward
            current_episode_length = episode_length

            if len(episode_return_window) == WINDOW_SIZE:

                return_sum -= episode_return_window[0]
                length_sum -= episode_length_window[0]

            episode_return_window.append(episode_return)
            episode_length_window.append(current_episode_length)
            return_sum += episode_return
            length_sum += current_episode_length

            window_size = len(episode_return_window)
            wandb.log(
                {
                    "rollout/episode_return": return_sum / window_size,
                    "rollout/episode_length": length_sum / window_size,
                },
                step=step,
            )

            episode_reward = 0.0
            episode_length = 0

            key, reset_key = jax.random.split(key)
            obs, info = env.reset(reset_key)

        if step > cfg.start_steps and buffer.can_sample():
            start = time()

            batch = buffer.sample(batch_size=cfg.buffer.batch_size * cfg.utd)

            t = max(0, int(step - cfg.start_steps))

            enable_supervision = (t >= int(cfg.supervision_warmup_steps)) and bool(guidance_buffer.is_full)

            if (t < int(cfg.supervision_warmup_steps)) or (not enable_supervision):
                supervision_coef = 0.0
            else:
                ramp_t = t - int(cfg.supervision_warmup_steps)
                ramp_steps = max(1, int(cfg.supervision_ramp_steps))
                progress = min(1.0, ramp_t / ramp_steps)
                supervision_coef = float(cfg.supervision_coef_max) * progress

            if logstd_scheduler is not None:
                current_logstd = logstd_scheduler.get_logstd(t)
                actor.set_fixed_logstd(current_logstd)

            bs_flow = int(cfg.guidance_buffer.batch_size * cfg.utd)
            if enable_supervision:
                guidance_batch = guidance_buffer.sample(batch_size=bs_flow)
            else:
                guidance_batch = GuidanceBatch(
                    obs=jnp.zeros((bs_flow, env.obs_dim), dtype=jnp.float32),
                    act=jnp.zeros((bs_flow, env.act_dim), dtype=jnp.float32),
                    noise=jnp.zeros((bs_flow, env.act_dim), dtype=jnp.float32),
                    idx=jnp.zeros((bs_flow,), dtype=jnp.int32),
                )
            update_actor_mask = tuple(((total_updates + i + 1) % max(1, cfg.policy_delay) == 0) for i in range(cfg.utd))
            # phase-0 diagnostics (plan.md II-V). One "update" = one gaussian_flow_step call
            # (utd gradient steps; with supervision the actor moves on every one), so light
            # pre/post brackets a call-level aggregate — the per-update proxy of plan 1.2.
            do_light = diag is not None and step % cfg.diag.light_every == 0
            if do_light:
                pre = diag.light_pre(actor, alpha, guidance_batch if enable_supervision else None, buffer)
            metrics, refined_obs, refined_actions, refined_noise = step_fn(
                batch=batch,
                guidance_batch=guidance_batch,
                actor=actor,
                critic=critic,
                alpha=alpha,
                actor_optim=actor_optim,
                critic_optim=critic_optim,
                alpha_optim=alpha_optim,
                update_actor_mask=update_actor_mask,
                enable_supervision=enable_supervision,
                supervision_coef=jnp.asarray(supervision_coef, dtype=jnp.float32),
            )

            if do_light:
                light = diag.light_post(
                    actor, alpha, guidance_batch if enable_supervision else None, guidance_buffer, step, pre
                )
                # per-mini-step fresh-target/IS metrics from inside gaussian_flow_step
                # (phase1_plan §2.2/2.4); absent when this call had no actor update
                light.update(
                    {k: float(v) for k, v in metrics.items() if k.startswith("diag/") and np.isfinite(float(v))}
                )
                wandb.log(light, step=step)
            if diag is not None:
                if cfg.diag.probe_interval and step % cfg.diag.probe_interval == 0:
                    wandb.log(diag.heavy_probe(step, actor, critic, alpha, buffer), step=step)
                if (
                    cfg.diag.shadow_interval
                    and cfg.seed in list(cfg.diag.shadow_seeds)
                    and step % cfg.diag.shadow_interval == 0
                ):
                    wandb.log(diag.kl_shadow(step, actor, critic, alpha, actor_optim, buffer), step=step)
                # stored-target metrics are the only ones that require guidance targets
                if enable_supervision and cfg.diag.staleness_interval and step % cfg.diag.staleness_interval == 0:
                    wandb.log(diag.staleness(step, actor, critic, alpha, guidance_buffer), step=step)

            critic_metrics.update(
                critic_loss=metrics["critic_loss"],
                current_q_values=metrics["current_q_values"],
                next_q_values=metrics["next_q_values"],
                debug_entropy=metrics["debug_entropy"],
                supervision_loss=metrics["supervision_loss"],
                supervision_coef=metrics["supervision_coef"],
                train_time=time() - start,
            )
            actor_update_cnt = int(sum(update_actor_mask))

            if actor_update_cnt > 0:
                actor_updates_since_log += actor_update_cnt
                actor_metrics.update(
                    actor_loss=metrics["actor_loss"],
                    alpha_loss=metrics["alpha_loss"],
                    alpha=metrics["alpha"],
                    conditional_entropy=metrics["conditional_entropy"],
                    mean_marginal_entropy=metrics["mean_marginal_entropy"],
                    variance=metrics["variance"],
                    advantage=metrics["advantage"],
                )
                guidance_buffer.add(refined_obs, refined_actions, refined_noise, step=step)
            total_updates += cfg.utd

        if step % cfg.log_interval == 0:
            log = {}

            if step > cfg.start_steps:
                for k, v in critic_metrics.compute().items():
                    log[f"train/{k}"] = float(v)

                if actor_updates_since_log > 0:
                    for k, v in actor_metrics.compute().items():
                        log[f"train/{k}"] = float(v)

                if actor.logstd_mode == "fixed":
                    log["train/logstd"] = float(actor.fixed_logstd.value.mean())

                critic_metrics.reset()
                actor_metrics.reset()
                actor_updates_since_log = 0

            if step % cfg.eval_interval == 0:
                start = time()
                key, eval_key = jax.random.split(key)
                eval_results = evaluate_vectorized_env(
                    vec_env=eval_env,
                    policy=partial(
                        evaluate_policy,
                        actor=actor,
                        critic=critic,
                    ),
                    key=eval_key,
                )
                log.update(eval_results)
                log.update({"eval/eval_time": time() - start})
                tqdm.write(f"\n🔍 Evaluation at step {step}:")
                tqdm.write(f"   Eval Return: {log['eval/episode_return']:.2f}")
                tqdm.write(f"   Eval Length: {log['eval/episode_length']:.2f}")
                if "eval/success_rate" in log:
                    tqdm.write(f"   Success Rate: {log['eval/success_rate']:.2%}")
                tqdm.write("")
            if log:
                wandb.log(log, step=step)

        if cfg.ckpt_interval and step > start_step and step % cfg.ckpt_interval == 0:
            logstd = float(actor.fixed_logstd.value.mean()) if actor.logstd_mode == "fixed" else None
            path = save_checkpoint(
                ckpt_dir,
                step,
                ckpt_modules,
                ckpt_buffers,
                extra={
                    "key_data": np.asarray(jax.random.key_data(key)),
                    "logstd": logstd,
                    "num_train_action_samples": int(actor.num_train_action_samples),
                    "episode_cnt": episode_cnt,
                    "total_updates": total_updates,
                    "episode_return_window": episode_return_window,
                    "episode_length_window": episode_length_window,
                    "normalizer": normalizer if cfg.normalize_env else None,
                    "wandb_id": wandb.run.id,
                },
            )
            tqdm.write(f"💾 checkpoint @ step {step} (logstd={logstd}) → {path}")


if __name__ == "__main__":
    main()
