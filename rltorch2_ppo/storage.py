import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, unroll_length, num_processes, batch_size, img_shape,
                 gamma, lambd, device):

        # Transitions.
        self.states = torch.zeros(
            unroll_length + 1, num_processes, *img_shape, device=device)
        self.rewards = torch.zeros(
            unroll_length, num_processes, 1, device=device)
        self.actions = torch.zeros(
            unroll_length, num_processes, 1, device=device, dtype=torch.long)
        self.dones = torch.ones(
            unroll_length + 1, num_processes, 1, device=device)

        # Log of action probabilities based on the current policy.
        self.action_log_probs = torch.zeros(
            unroll_length, num_processes, 1, device=device)
        # Predictions of V(s_t) based on the current value function.
        self.pred_values = torch.zeros(
            unroll_length + 1, num_processes, 1, device=device)
        # Target estimate of V(s_t) based on rollouts.
        self.target_values = torch.zeros(
            unroll_length + 1, num_processes, 1, device=device)

        self.step = 0
        self.unroll_length = unroll_length
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.gamma = gamma
        self.lambd = lambd

        self._is_ready = False

    def init_states(self, states):
        self.states[0].copy_(states)

    def insert(self, next_states, actions, rewards, dones, action_log_probs,
               pred_values):
        self.states[self.step + 1].copy_(next_states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step + 1].copy_(dones)

        self.action_log_probs[self.step].copy_(action_log_probs)
        self.pred_values[self.step].copy_(pred_values)
        self.step = (self.step + 1) % self.unroll_length

    def end_rollout(self, next_value):
        assert not self._is_ready
        self._is_ready = True

        self.pred_values[-1].copy_(next_value)
        adv = 0
        for step in reversed(range(self.unroll_length)):
            td_error = self.rewards[step] + \
                self.gamma * self.pred_values[step+1] * (1 - self.dones[step])\
                - self.pred_values[step]
            adv = td_error + \
                self.gamma * self.lambd * (1 - self.dones[step]) * adv
            self.target_values[step] = adv + self.pred_values[step]

    def iterate(self, num_gradient_steps):
        assert self._is_ready

        # Calculate advantages.
        all_advs = self.target_values[:-1] - self.pred_values[:-1]
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-5)

        for _ in range(num_gradient_steps):
            # Sampler for indices.
            sampler = BatchSampler(
                SubsetRandomSampler(
                    range(self.num_processes * self.unroll_length)),
                self.batch_size, drop_last=True)

            for indices in sampler:
                states = self.states[:-1].view(-1, *self.img_shape)[indices]
                actions = self.actions.view(-1, self.actions.size(-1))[indices]
                pred_values = self.pred_values[:-1].view(-1, 1)[indices]
                target_values = self.target_values.view(-1, 1)[indices]
                action_log_probs = self.action_log_probs.view(-1, 1)[indices]
                advs = all_advs.view(-1, 1)[indices]

                yield states, actions, pred_values, \
                    target_values, action_log_probs, advs

        self.states[0].copy_(self.states[-1])
        self.dones[0].copy_(self.dones[-1])
        self._is_ready = False
