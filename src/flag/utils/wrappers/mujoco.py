

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax import Array

class JAXEnvWrapper:
    

    def __init__(self, env: gym.Env, seed: int):

        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box)
        self.env = env


        assert env.spec is not None and env.spec.max_episode_steps is not None,\
            "MuJoCo/Gymnasium environments must have max_episode_steps in spec"
        self.max_episode_steps = env.spec.max_episode_steps

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        low = env.action_space.low
        high = env.action_space.high
        self.scale = (high - low) / 2.0
        self.shift = (high + low) / 2.0

    def sample_random_action(self, key: Array | None = None) -> Array:
        if key is not None:
            action = jax.random.uniform(
                key,
                shape=(self.act_dim,),
                minval=-1.0,
                maxval=1.0
            )
        else:
            action = (self.env.action_space.sample() - self.shift) / self.scale
        return action

    def reset(self, key: Array) -> tuple[Array, dict]:
        seed_int = int(jax.random.randint(key, (), 0, 2**31 - 1))
        obs, info = self.env.reset(seed=seed_int)
        return obs, info

    def step(self, action: Array) -> tuple[Array, float, bool, bool, dict]:

        action_np = np.asarray(action)
        action_denorm = self.scale * action_np + self.shift
        next_obs, reward, terminated, truncated, info = self.env.step(action_denorm)
        return next_obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    @staticmethod
    def convert_info_to_jax(info: dict[str, NDArray]) -> dict[str, Array]:
        return {key: jnp.array(value) for key, value in info.items()}

class JAXVectorEnvWrapper:
    
    def __init__(self, vector_env: gym.vector.VectorEnv):
        self.env = vector_env
        self.num_envs = vector_env.num_envs
        self.single_observation_space = vector_env.single_observation_space
        self.single_action_space = vector_env.single_action_space

        assert isinstance(self.single_observation_space, Box)
        assert isinstance(self.single_action_space, Box)

        self.obs_dim = self.single_observation_space.shape[0]
        self.act_dim = self.single_action_space.shape[0]

        low = self.single_action_space.low
        high = self.single_action_space.high
        self.scale = (high - low) / 2.0
        self.shift = (high + low) / 2.0

    def reset(self, key: Array | None = None) -> tuple[np.ndarray, dict]:
        
        if key is not None:

            seed = int(jax.random.randint(key, (), 0, 2**30))
            obs, infos = self.env.reset(seed=seed)
        else:
            obs, infos = self.env.reset()

        return obs, infos

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:

        if hasattr(actions, '__array__') or isinstance(actions, Array):
            actions = np.asarray(actions)

        actions_denorm = self.scale * actions + self.shift

        return self.env.step(actions_denorm)

    def close(self):
        self.env.close()


def make_env(cfg):
    

    if cfg.env_id.startswith("dm_control/"):
        raise ValueError(
            f"Environment {cfg.env_id} is a dm_control environment. "
            "Use flag.utils.wrappers.dmcontrol.make_env instead."
        )


    base_env = gym.make(cfg.env_id)
    train_env = JAXEnvWrapper(base_env, cfg.seed)


    vector_eval_env = gym.make_vec(
        cfg.env_id,
        num_envs=cfg.eval_episodes,
        vectorization_mode="sync"
    )
    eval_env = JAXVectorEnvWrapper(vector_eval_env)

    return train_env, eval_env


if __name__ == "__main__":
    seed = 42
    env_name = "HalfCheetah-v5"
    env = JAXEnvWrapper(gym.make(env_name), seed)
    obs, info = env.reset()
    print("obs:", obs)
    print("info:", info)

    action = env.sample_random_action()
    print("action:", action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    print("next_obs:", next_obs)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("info:", info)
