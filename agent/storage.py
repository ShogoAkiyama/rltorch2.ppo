import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, num_mini_batch, img_shape):
        self.states = torch.zeros(num_steps + 1, num_processes, *img_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.vs_targets = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1)
        self.actions = self.actions.long()
        self.dones = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

        self.img_shape = img_shape
        self.batch_size = num_processes * num_steps
        self.mini_batch_size = self.batch_size // num_mini_batch

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.vs_targets = self.vs_targets.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.dones = self.dones.to(device)

    def insert(self, states, actions, action_log_probs,
               value_preds, rewards, dones):
        self.states[self.step+1].copy_(states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step+1].copy_(dones)
        self.step = (self.step+1) % self.num_steps

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.dones[0].copy_(self.dones[-1])

    def compute_returns(self, next_value, gamma, lambd):
        self.value_preds[-1] = next_value
        adv = 0
        for step in reversed(range(self.num_steps)):
            td_error = self.rewards[step] + \
                       gamma * self.value_preds[step+1] * (1 - self.dones[step+1]) -\
                       self.value_preds[step]
            adv = td_error + gamma * lambd * (1 - self.dones[step+1]) * adv
            self.vs_targets[step] = adv + self.value_preds[step]

    def feed_forward_generator(self, advantages):
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, drop_last=True)
        for indices in sampler:
            states = self.states[:-1].view(-1, *self.img_shape)[indices]

            actions = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds = self.value_preds[:-1].view(-1, 1)[indices]
            vs_targets = self.vs_targets[:-1].view(-1, 1)[indices]
            log_probs_old = self.action_log_probs.view(-1, 1)[indices]
            advs = advantages.view(-1, 1)[indices]

            yield states, actions, value_preds, vs_targets, log_probs_old, advs
