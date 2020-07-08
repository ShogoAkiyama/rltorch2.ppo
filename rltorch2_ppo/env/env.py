import numpy as np
import gym
import torch


def make_env(env_id, num_env, device):
    env = gym.vector.make(env_id, num_env)

    env = VecPyTorch(env, device)
    env = VecPyTorchFrameStack(env, 4, device)

    return env


class VecPyTorch(gym.Wrapper):
    def __init__(self, env, device):
        gym.Wrapper.__init__(self, env)
        self.device = device

    def reset(self):
        obs = self.env.reset()
        obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions.cpu().numpy().reshape(-1))
        obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        done = torch.from_numpy(done).to(self.device).unsqueeze(dim=1)
        return obs, reward, done, info


class VecPyTorchFrameStack(gym.Wrapper):
    def __init__(self, env, nstack, device=None):
        gym.Wrapper.__init__(self, env)
        self.nstack = nstack

        self.shape_dim0 = 3

        obs_shape = self.env.observation_space.shape
        obs_shape = [obs_shape[3]*nstack,
            obs_shape[1], obs_shape[2]]
        low = np.zeros((
            obs_shape[0], obs_shape[1], obs_shape[2]), dtype=np.int8)
        high = np.ones((
            obs_shape[0], obs_shape[1], obs_shape[2]), dtype=np.int8) * 255
        self.observation_space = gym.spaces.Box(
            low, high, obs_shape)

        self.stacked_obs = torch.zeros(
            (env.num_envs, ) + (3*nstack, 64, 64)).to(device)

    def step(self, actions):
        obs, rews, news, infos = self.env.step(actions)
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.env.reset()
        self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

