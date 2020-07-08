import gym
import torch
from procgen import ProcgenEnv

from rltorch2_ppo.env.procgen_env_wrapper import default_config
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
    'nstacks': 1,
    'num_envs': 8
}

class MyEnv(gym.Env):

    def __init__(self, config={}):
        self.config = default_config
        self.config.update(config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nstacks = self.config.pop('nstacks')

        env = ProcgenEnv(**self.config)
        env = VecExtractDictObs(env, "rgb")
        obs_shape = env.observation_space.shape

        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        env = VecPyTorch(env, device)
        env = VecPyTorchFrameStack(env, nstacks, device)
        self.venv = env

    def reset(self):
        return self.venv.reset()

    def step(self, actions):
        obs, rew, done, info = self.venv.step(actions)
        return obs, rew, done, info

    @property
    def num_envs(self):
        return self.venv.num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space