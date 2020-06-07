from collections import deque
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import PPOModel
from agent.storage import RolloutStorage
from agent.utils import update_linear_schedule


class PPO:

    def __init__(self, envs, device, log_dir, max_num_steps, dataset_size,
                 num_processes, recurrent, clip_param, ppo_epoch,
                 num_mini_batch, value_loss_coef, entropy_coef, lr=None,
                 gamma=None, lambd=None, eps=None, max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.device = device
        self.envs = envs

        self.online_network = PPOModel(
            envs.observation_space.shape,
            envs.action_space).to(device)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            self.online_network.parameters(), lr=lr, eps=eps)

        self.max_num_steps = max_num_steps
        self.dataset_size = dataset_size
        self.num_processes = num_processes
        self.num_updates = \
            int(self.max_num_steps) // self.dataset_size // self.num_processes

        self.rollouts = RolloutStorage(
            self.dataset_size, self.num_processes,
            self.num_mini_batch, self.envs.observation_space.shape)

        self.episode_r = [[] for _ in range(num_processes)]
        self.episode_rewards = deque(maxlen=10)

        self.recurrent = recurrent
        self.lr = lr
        self.gamma = gamma
        self.lambd = lambd

        # Prepare for logging.
        self._log_dir = log_dir
        self._model_dir = os.path.join(log_dir, 'model')
        self._summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        if not os.path.exists(self._summary_dir):
            os.makedirs(self._summary_dir)

        self._writer = SummaryWriter(log_dir=self._summary_dir)

    def run(self):
        states = self.envs.reset()
        # self.rollouts.states[0].copy_(states)
        self.rollouts.to(self.device)

        for step in range(self.num_updates):
            # decrease learning rate linearly
            update_linear_schedule(
                self.optimizer, step, self.num_updates, self.lr)

            # experiments
            for _ in range(self.dataset_size):
                with torch.no_grad():
                    value, action, action_log_prob = \
                        self.online_network.act(states)

                next_states, reward, done, infos = self.envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        self.episode_rewards.append(info['episode']['r'])

                self.rollouts.insert(
                    next_states, action, action_log_prob, value, reward, done)

                states = next_states

            # V(s_T)(129)
            with torch.no_grad():
                next_value = \
                    self.online_network.get_value(next_states).detach()

            self.rollouts.compute_returns(
                next_value, self.gamma, self.lambd)

            self.update(self.rollouts)

            self.rollouts.after_update()

            if len(self.episode_rewards) > 1:
                total_num_steps = \
                    (step + 1) * self.num_processes * self.dataset_size
                r_mean = np.mean(self.episode_rewards)
                r_median = np.median(self.episode_rewards)
                r_min = np.min(self.episode_rewards)
                r_max = np.max(self.episode_rewards)
                print(f"Updates {step}, total_steps {total_num_steps}   "
                      "mean/median/min/max reward   "
                      f"{r_mean:.1f}/{r_median:.1f}/{r_min:.1f}/{r_max:.1f}")
                self._writer.add_scalar(
                    'return/train', np.mean(self.episode_rewards),
                    total_num_steps)

    def update(self, rollouts):
        all_advs = rollouts.vs_targets[:-1] - rollouts.value_preds[:-1]
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-5)

        for epoch in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(all_advs)

            for sample in data_generator:
                states, actions, value_preds, \
                    vs_targets, log_probs_old, advs = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = \
                    self.online_network.evaluate_actions(states, actions)

                ratio = torch.exp(action_log_probs - log_probs_old)

                loss_policy = -torch.min(ratio * advs, torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advs
                ).mean()

                value_pred_clipped = value_preds + (
                    values - value_preds
                ).clamp(-self.clip_param, self.clip_param)

                loss_value = 0.5 * torch.max(
                    (values - vs_targets).pow(2),
                    (value_pred_clipped - vs_targets).pow(2)
                ).mean()

                self.optimizer.zero_grad()
                loss = (
                    loss_policy
                    + self.value_loss_coef * loss_value
                    - self.entropy_coef * dist_entropy
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.online_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
