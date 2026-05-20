import numpy as np
from jax import Array
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from flag.utils.wrappers.mujoco import JAXEnvWrapper, JAXVectorEnvWrapper
    from flag.utils.wrappers.dmcontrol import DMControlJAXEnvWrapper, DMControlJAXVectorEnvWrapper

class RunningMeanStd:
    def __init__(self, shape, epsilon:float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = x.reshape(1)

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class Normalizer:
    def __init__(
        self,
        obs_shape,
        gamma: float=0.99,
        epsilon: float=1e-8
    ):
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)


class NormalizeJAXEnvWrapper:
    
    def __init__(
        self,
        env: Union["JAXEnvWrapper", "DMControlJAXEnvWrapper"],
        normalizer: Normalizer,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
    ):
        self.env = env
        self.normalizer = normalizer
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.max_episode_steps = getattr(env, "max_episode_steps", None)

        self._returns = 0.0

    def sample_random_action(self, key: Array | None = None) -> Array:
        return self.env.sample_random_action(key)

    def reset(self, key: Array):
        obs, info = self.env.reset(key)
        self._returns = 0.0

        if self.norm_obs and self.training:
            self.normalizer.obs_rms.update(np.asarray(obs)[None])
        if self.norm_obs:
            obs = self._normalize_obs(obs)

        return obs, info

    def step(self, action: Array):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(reward)

        if self.norm_reward:
            self._returns = self._returns * self.normalizer.gamma + raw_reward
            raw_reward = float(reward)

            if self.training:
                self.normalizer.ret_rms.update(np.asarray(self._returns, dtype=np.float64))

            denom = np.sqrt(np.maximum(self.normalizer.ret_rms.var, 0.0) + self.normalizer.epsilon)
            reward = float(np.clip(raw_reward / denom, -self.clip_reward, self.clip_reward))

            if terminated or truncated:
                self._returns = 0.0

        if self.norm_obs and self.training:
            self.normalizer.obs_rms.update(np.asarray(next_obs)[None])
        if self.norm_obs:
            next_obs = self._normalize_obs(next_obs)

        info = dict(info)
        info["raw_reward"] = raw_reward
        return next_obs, reward, terminated, truncated, info

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        mean = self.normalizer.obs_rms.mean.astype(np.float32)
        var = self.normalizer.obs_rms.var.astype(np.float32)
        obs = (obs - mean) / np.sqrt(np.maximum(var, 0.0) + self.normalizer.epsilon)
        return np.clip(obs, -self.clip_obs, self.clip_obs)


class NormalizeJAXVectorEnvWrapper:
    
    def __init__(
        self,
        env: Union["JAXVectorEnvWrapper", "DMControlJAXVectorEnvWrapper"],
        normalizer: Normalizer,
        training: bool = False,
        norm_obs: bool = True,
        norm_reward: bool = False,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
    ):
        self.env = env
        self.normalizer = normalizer
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = float(clip_obs)
        self.clip_reward = float(clip_reward)

        self.num_envs = env.num_envs
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self._returns = np.zeros(self.num_envs, dtype=np.float64)

    def reset(self, key: Array | None = None):
        obs, infos = self.env.reset(key)
        self._returns[...] = 0.0

        if self.norm_obs and self.training:
            self.normalizer.obs_rms.update(np.asarray(obs))
        if self.norm_obs:
            obs = self._normalize_obs(obs)

        return obs, infos

    def step(self, actions: np.ndarray):
        next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        raw_rewards = np.asarray(rewards, dtype=np.float32)

        if self.norm_reward:
            self._returns = self._returns * self.normalizer.gamma + raw_rewards
            if self.training:
                self.normalizer.ret_rms.update(self._returns)

            denom = np.sqrt(np.maximum(self.normalizer.ret_rms.var, 0.0) + self.normalizer.epsilon)
            rewards = np.clip(raw_rewards / denom, -self.clip_reward, self.clip_reward)

            dones = np.asarray(terminateds) | np.asarray(truncateds)
            self._returns[dones] = 0.0
        else:
            rewards = raw_rewards

        if self.norm_obs and self.training:
            self.normalizer.obs_rms.update(np.asarray(next_obs))
        if self.norm_obs:
            next_obs = self._normalize_obs(next_obs)

        infos = dict(infos)
        infos["raw_reward"] = raw_rewards
        return next_obs, rewards, terminateds, truncateds, infos

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        mean = self.normalizer.obs_rms.mean.astype(np.float32)
        var = self.normalizer.obs_rms.var.astype(np.float32)
        obs = (obs - mean) / np.sqrt(np.maximum(var, 0.0) + self.normalizer.epsilon)
        return np.clip(obs, -self.clip_obs, self.clip_obs)

    def close(self):
        return self.env.close()