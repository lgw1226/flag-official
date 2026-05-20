

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from numpy.typing import NDArray
from collections import OrderedDict
import jax
import jax.numpy as jnp
from jax import Array
from dm_control import suite


class DMControlJAXEnvWrapper:
    

    def __init__(self, env, seed: int):
        
        self.env = env
        self._seed = seed


        obs_spec = env.observation_spec()
        act_spec = env.action_spec()


        self.obs_dim = self._calculate_obs_dim(obs_spec)
        self.act_dim = act_spec.shape[0]


        self.scale = (act_spec.maximum - act_spec.minimum) / 2.0
        self.shift = (act_spec.maximum + act_spec.minimum) / 2.0
        self.max_episode_steps = 1000

    def _calculate_obs_dim(self, obs_spec) -> int:
        
        total_dim = 0
        for key, spec in obs_spec.items():
            if hasattr(spec, 'shape'):
                total_dim += int(np.prod(spec.shape))
            else:
                total_dim += 1
        return total_dim

    def _flatten_observation(self, obs: OrderedDict) -> np.ndarray:
        
        flat_obs = []
        for key, value in obs.items():
            flat_obs.append(np.asarray(value).flatten())
        return np.concatenate(flat_obs, dtype=np.float32)

    def reset(self, key: Array) -> tuple[np.ndarray, dict]:
        

        seed_int = int(jax.random.randint(key, (), 0, 2**31 - 1))
        np.random.seed(seed_int)


        timestep = self.env.reset()
        obs = self._flatten_observation(timestep.observation)

        info = {}
        return obs, info

    def step(self, action: Array) -> tuple[np.ndarray, float, bool, bool, dict]:
        
        action_np = np.asarray(action)
        action_denorm = self.scale * action_np + self.shift

        timestep = self.env.step(action_denorm)

        obs = self._flatten_observation(timestep.observation)



        terminated = bool(timestep.last() and timestep.discount == 0.0)
        truncated = bool(timestep.last() and timestep.discount > 0.0)

        info = {
            'discount': timestep.discount,
        }

        return obs, float(timestep.reward), terminated, truncated, info

    def sample_random_action(self, key: Array | None = None) -> np.ndarray:
        
        if key is not None:
            action = jax.random.uniform(
                key,
                shape=(self.act_dim,),
                minval=-1.0,
                maxval=1.0
            )
            return np.asarray(action)
        else:
            act_spec = self.env.action_spec()
            raw_action = np.random.uniform(
                act_spec.minimum,
                act_spec.maximum,
                size=act_spec.shape
            )

            normalized_action = (raw_action - self.shift) / self.scale
            return normalized_action

    def render(self, width=384, height=384, camera_id=0):
        
        return self.env.physics.render(height, width, camera_id)


class DMControlJAXVectorEnvWrapper:
    

    def __init__(self, domain: str, task: str, num_envs: int, seed: int):
        

        self.envs = [suite.load(domain, task) for _ in range(num_envs)]
        self.num_envs = num_envs


        obs_spec = self.envs[0].observation_spec()
        act_spec = self.envs[0].action_spec()

        self.obs_dim = self._calculate_obs_dim(obs_spec)
        self.act_dim = act_spec.shape[0]


        self.scale = (act_spec.maximum - act_spec.minimum) / 2.0
        self.shift = (act_spec.maximum + act_spec.minimum) / 2.0

        self._base_seed = seed

    def _calculate_obs_dim(self, obs_spec) -> int:
        
        total_dim = 0
        for key, spec in obs_spec.items():
            if hasattr(spec, 'shape'):
                total_dim += int(np.prod(spec.shape))
            else:
                total_dim += 1
        return total_dim

    def _flatten_observation(self, obs: OrderedDict) -> np.ndarray:
        
        flat_obs = []
        for key, value in obs.items():
            flat_obs.append(np.asarray(value).flatten())
        return np.concatenate(flat_obs, dtype=np.float32)

    def reset(self, key: Array | None = None) -> tuple[np.ndarray, dict]:
        
        if key is not None:
            base_seed = int(jax.random.randint(key, (), 0, 2**30))
        else:
            base_seed = self._base_seed

        obs_list = []
        for i, env in enumerate(self.envs):
            np.random.seed(base_seed + i)
            timestep = env.reset()
            obs_list.append(self._flatten_observation(timestep.observation))

        obs_array = np.stack(obs_list, axis=0)
        infos = {}

        return obs_array, infos

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        
        if hasattr(actions, '__array__') or isinstance(actions, Array):
            actions = np.asarray(actions)

        actions_denorm = self.scale * actions + self.shift

        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        discount_list = []

        for i, env in enumerate(self.envs):
            action = actions_denorm[i]
            timestep = env.step(action)

            is_terminated = bool(timestep.last() and timestep.discount == 0.0)
            is_truncated = bool(timestep.last() and timestep.discount > 0.0)

            obs_list.append(self._flatten_observation(timestep.observation))
            reward_list.append(timestep.reward)
            terminated_list.append(is_terminated)
            truncated_list.append(is_truncated)
            discount_list.append(timestep.discount)

        obs_array = np.stack(obs_list, axis=0)
        rewards = np.array(reward_list, dtype=np.float32)
        terminateds = np.array(terminated_list, dtype=bool)
        truncateds = np.array(truncated_list, dtype=bool)

        infos = {'discount': np.array(discount_list, dtype=np.float32)}

        return obs_array, rewards, terminateds, truncateds, infos

    def close(self):
        
        for env in self.envs:
            env.close()


def parse_dm_control_env_id(env_id: str) -> tuple[str, str]:
    
    if not env_id.startswith("dm_control/"):
        raise ValueError(f"Invalid dm_control env_id: {env_id}")


    env_name = env_id.replace("dm_control/", "").replace("-v0", "")




    parts = env_name.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse domain and task from: {env_id}")

    domain, task = parts

    domain = domain.replace("-", "_")

    return domain, task


def make_env(cfg):
    
    domain, task = parse_dm_control_env_id(cfg.env_id)


    train_env_base = suite.load(domain, task)
    train_env = DMControlJAXEnvWrapper(train_env_base, cfg.seed)


    eval_env = DMControlJAXVectorEnvWrapper(
        domain=domain,
        task=task,
        num_envs=cfg.eval_episodes,
        seed=cfg.seed
    )

    return train_env, eval_env
