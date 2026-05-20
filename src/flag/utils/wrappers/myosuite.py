import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax import Array

try:
    from myosuite.utils import gym as myo_gym
    import myosuite
except ImportError:
    myo_gym = None
    myosuite = None

MYOSUITE_TASKS = {
    "myo-reach": "myoHandReachFixed-v0",
    "myo-reach-hard": "myoHandReachRandom-v0",
    "myo-pose": "myoHandPoseFixed-v0",
    "myo-pose-hard": "myoHandPoseRandom-v0",
    "myo-obj-hold": "myoHandObjHoldFixed-v0",
    "myo-obj-hold-hard": "myoHandObjHoldRandom-v0",
    "myo-key-turn": "myoHandKeyTurnFixed-v0",
    "myo-key-turn-hard": "myoHandKeyTurnRandom-v0",
    "myo-pen-twirl": "myoHandPenTwirlFixed-v0",
    "myo-pen-twirl-hard": "myoHandPenTwirlRandom-v0",
}


class MyoSuiteJAXEnvWrapper:

    def __init__(self, env, seed: int, max_episode_steps: int | None = None):
        if myo_gym is None or myosuite is None:
            raise ImportError("MyoSuite not installed. Please install it to use this wrapper.")

        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box)
        self.env = env
        self._seed = seed

        task_specific_max_steps = getattr(env, "_max_episode_steps", None)
        if task_specific_max_steps is None and hasattr(env, "spec") and env.spec is not None:
            task_specific_max_steps = getattr(env.spec, "max_episode_steps", 100)
        if task_specific_max_steps is None:
            task_specific_max_steps = 100

        if max_episode_steps is not None:
            self.max_episode_steps = min(max_episode_steps, task_specific_max_steps)
        else:
            self.max_episode_steps = task_specific_max_steps

        self._current_step = 0

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        low = env.action_space.low
        high = env.action_space.high
        self.scale = (high - low) / 2.0
        self.shift = (high + low) / 2.0

    def sample_random_action(self, key: Array | None = None) -> Array:
        if key is not None:
            action = jax.random.uniform(key, shape=(self.act_dim,), minval=-1.0, maxval=1.0)
        else:
            action = (self.env.action_space.sample() - self.shift) / self.scale
        return action

    def reset(self, key: Array) -> tuple[Array, dict]:
        seed_int = int(jax.random.randint(key, (), 0, 2**31 - 1))
        obs, info = self.env.reset(seed=seed_int)
        self._current_step = 0
        return obs, info

    def step(self, action: Array) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_np = np.asarray(action)
        action_denorm = self.scale * action_np + self.shift
        next_obs, reward, terminated, truncated, info = self.env.step(action_denorm)

        self._current_step += 1

        if self._current_step >= self.max_episode_steps:
            truncated = True

        return next_obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, width=384, height=384, camera_id="hand_side_inter"):
        return self.env.unwrapped.sim.renderer.render_offscreen(width=width, height=height, camera_id=camera_id).copy()

    def close(self):
        self.env.close()


class MyoSuiteJAXVectorEnvWrapper:

    def __init__(self, env_id: str, num_envs: int, seed: int, max_episode_steps: int | None = None):
        if myo_gym is None:
            raise ImportError("MyoSuite not installed")

        self.envs = [MyoSuiteJAXEnvWrapper(myo_gym.make(env_id), seed + i, max_episode_steps) for i in range(num_envs)]
        self.num_envs = num_envs

        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim
        self.scale = self.envs[0].scale
        self.shift = self.envs[0].shift
        self.max_episode_steps = self.envs[0].max_episode_steps

    def reset(self, key: Array | None = None) -> tuple[np.ndarray, dict]:
        if key is not None:
            base_seed = int(jax.random.randint(key, (), 0, 2**30))
        else:
            base_seed = 0

        obs_list = []
        for i, env in enumerate(self.envs):
            k = jax.random.PRNGKey(base_seed + i)
            o, _ = env.reset(k)
            obs_list.append(o)

        return np.stack(obs_list), {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        if hasattr(actions, "__array__") or isinstance(actions, Array):
            actions = np.asarray(actions)

        obs_list = []
        rew_list = []
        term_list = []
        trunc_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            o, r, t, tr, info = env.step(actions[i])
            obs_list.append(o)
            rew_list.append(r)
            term_list.append(t)
            trunc_list.append(tr)
            info_list.append(info)

        solved_flags = [info.get("solved", False) for info in info_list]

        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            {"solved": np.array(solved_flags, dtype=bool)},
        )

    def close(self):
        for env in self.envs:
            env.close()


def is_myosuite_env(env_id: str) -> bool:

    return env_id in MYOSUITE_TASKS


def make_env(cfg):

    if myo_gym is None:
        raise ImportError("MyoSuite not installed")

    if cfg.env_id not in MYOSUITE_TASKS:
        raise ValueError(f"Unknown task: {cfg.env_id}")

    myo_env_id = MYOSUITE_TASKS[cfg.env_id]

    max_episode_steps = getattr(cfg, "max_episode_steps", 100)

    train_env = MyoSuiteJAXEnvWrapper(myo_gym.make(myo_env_id), cfg.seed, max_episode_steps=max_episode_steps)

    eval_env = MyoSuiteJAXVectorEnvWrapper(
        env_id=myo_env_id, num_envs=cfg.eval_episodes, seed=cfg.seed, max_episode_steps=max_episode_steps
    )

    return train_env, eval_env
