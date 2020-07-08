import gym
import torch
from procgen.env import ENV_NAMES as VALID_ENV_NAMES
from rltorch2_ppo.env.wrappers import VecExtractDictObs, TransposeImage, VecPyTorch, \
    VecPyTorchFrameStack


default_config = {
    # Set to 0 to use unlimited levels.
    "num_levels": 0,
    "env_name": "coinrun",
    "start_level": 0,
    "paint_vel_info": False,
    "use_generated_assets": False,
    "center_agent": True,
    "use_sequential_levels": False,
    "distribution_mode": "easy",
    'nstacks': 4,
    'num_envs': 8
}


class ProcgenEnvWrapper(gym.Env):

    def __init__(self):
        # Config.
        self.config = default_config
        nstacks = self.config.pop('nstacks')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Env name.
        self.env_name = self.config.pop("env_name")
        assert self.env_name in VALID_ENV_NAMES

        env = gym.vector.make(
            f"procgen:procgen-{self.env_name}-v0", **self.config)

        # env = VecExtractDictObs(env, "rgb")
        obs_shape = env.observation_space.shape

        env = TransposeImage(env, op=[0, 3, 1, 2])

        env = VecPyTorch(env, device)
        env = VecPyTorchFrameStack(env, nstacks, device)

        self.env = env

        # Enable video recording features.
        self.metadata = self.env.metadata

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._done = True

    def reset(self):
        assert self._done, "procgen envs cannot be early-restarted."
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._done = done
        return obs, rew, done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __repr__(self):
        return self.env.__repr()

    @property
    def spec(self):
        return self.env.spec
