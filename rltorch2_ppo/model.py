from functools import partial
import torch
from torch import nn

from rltorch2_ppo.network import ImpalaCNNBody

def init_fn(m, gain=1):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PPOModel(nn.Module):
    def __init__(self, img_shape, action_space):
        super(PPOModel, self).__init__()
        # self.base = CNNBase(img_shape[0])
        self.base = ImpalaCNNBody(
            img_shape[0])
        self.dist = Categorical(self.base.output_size, 15)

    def forward(self, states, deterministic=False):
        value, actor_features = self.base(states)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def calculate_value(self, states):
        value, _ = self.base(states)
        return value

    def evaluate_actions(self, states, action):
        value, actor_features = self.base(states)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase, self).__init__()

        self._hidden_size = hidden_size

        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 4 * 4, hidden_size),
            nn.ReLU()
        ).apply(partial(init_fn, gain=nn.init.calculate_gain('relu')))

        # self.critic_linear = nn.Linear(hidden_size, 1).apply(init_fn)

        Nb = 50
        self.critic_linear = nn.Linear(hidden_size, Nb).apply(init_fn)
        self.cluster_linear = nn.Linear(Nb, Nb).apply(init_fn)
        self.softmax = nn.Softmax(dim=1)
        self.mu_linear = nn.Linear(Nb, Nb).apply(init_fn)

    @property
    def output_size(self):
        return self._hidden_size

    # def forward(self, inputs):
    #     x = self.main(inputs / 255.0)
    #     return self.critic_linear(x), x

    def forward(self, inputs):
        x = self.main(inputs / 255.0)
        y = self.critic_linear(x)
        a = self.cluster_linear(y)
        a = self.softmax(a)
        mu = self.mu_linear(y)

        value = (a * mu).sum(axis=1).unsqueeze(1)

        return value, x


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        self.linear = nn.Linear(num_inputs, num_outputs).apply(
            partial(init_fn, gain=0.01))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
