from collections import deque
import numpy as np
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from rltorch2_ppo.model import PPOModel
from rltorch2_ppo.storage import RolloutStorage


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PPO:

    def __init__(self, envs, device, log_dir, num_processes,
                 num_steps=10**6, batch_size=256, unroll_length=128, lr=2.5e-4,
                 adam_eps=1e-5, gamma=0.99, clip_param=0.1,
                 num_gradient_steps=4, value_loss_coef=0.5, entropy_coef=0.01,
                 lambd=0.95, max_grad_norm=0.5):

        self.envs = envs
        self.device = device

        # PPO network.
        self.network = PPOModel(
            envs.observation_space.shape,
            envs.action_space).to(device)

        # Optimizer.
        self.optimizer = Adam(self.network.parameters(), lr=lr, eps=adam_eps)

        # Storage.
        self.storage = RolloutStorage(
            unroll_length, num_processes, batch_size,
            envs.observation_space.shape, gamma, lambd, device)

        # Batch size.
        self.batch_size = batch_size
        # Unroll length.
        self.unroll_length = unroll_length
        # Number of processes.
        self.num_processes = num_processes
        # Number of staps to update.
        self.num_updates = num_steps // (unroll_length * num_processes)
        # Number of gradient staps per update.
        self.num_gradient_steps = num_gradient_steps

        # Hyperparameters.
        self.lr = lr
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # For logging.
        self.model_dir = os.path.join(log_dir, 'model')
        summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        self.writer = SummaryWriter(log_dir=summary_dir)
        self.episode_r = [[] for _ in range(num_processes)]
        self.episode_rewards = deque(maxlen=10)

    def run(self):
        states = self.envs.reset()
        self.storage.init_states(states)
        episodes = np.zeros(self.num_processes)
        for step in range(self.num_updates):
            # Decrease learning rate linearly.
            # update_linear_schedule(
            #     self.optimizer, step, self.num_updates, self.lr)

            for _ in range(self.unroll_length):
                with torch.no_grad():
                    values, actions, action_log_probs = self.network(states)
                next_states, rewards, dones, infos = self.envs.step(actions)

                episodes += rewards.cpu().numpy().reshape(-1)

                for i, done in enumerate(dones.cpu().detach().numpy()):
                    if done:
                        self.episode_rewards.append(episodes[i])
                        episodes[i] = 0

                self.storage.insert(
                    next_states, actions, rewards, dones, action_log_probs,
                    values)

                states = next_states

            with torch.no_grad():
                next_values = self.network.calculate_value(next_states)

            self.storage.end_rollout(next_values)
            self.update()
            if len(self.episode_rewards) > 1:
                total_steps = \
                    (step + 1) * self.num_processes * self.unroll_length
                print(f"\rSteps: {total_steps}   "
                      f"Updates: {step}   "
                      f"Mean Return: {np.mean(self.episode_rewards)}", end='')
                self.writer.add_scalar(
                    'return/train', np.mean(self.episode_rewards),
                    total_steps)

    def update(self):
        for sample in self.storage.iterate(self.num_gradient_steps):
            states, actions, pred_values, \
                target_values, log_probs_old, advs = sample

            # Reshape to do in a single forward pass for all steps.
            values, action_log_probs, dist_entropy = \
                self.network.evaluate_actions(states, actions)

            ratio = torch.exp(action_log_probs - log_probs_old)

            loss_policy = -torch.min(ratio * advs, torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advs
            ).mean()

            value_pred_clipped = pred_values + (
                values - pred_values
            ).clamp(-self.clip_param, self.clip_param)

            loss_value = 0.5 * torch.max(
                (values - target_values).pow(2),
                (value_pred_clipped - target_values).pow(2)
            ).mean()

            self.optimizer.zero_grad()
            loss = (
                loss_policy
                + self.value_loss_coef * loss_value
                - self.entropy_coef * dist_entropy
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

    def save_models(self, filename):
        torch.save(
            self.network.state_dict(), os.path.join(self.model_dir, filename))

    def load_models(self, filename):
        self.network.load_state_dict(
            torch.load(os.path.join(self.model_dir, filename)))
