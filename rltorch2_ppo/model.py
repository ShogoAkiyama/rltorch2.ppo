from functools import partial
import torch
from torch import nn
import torch.nn.functional as F


def init_fn(m, gain=1):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


def weights_init_head(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.orthogonal(m.weight)
        m.weight.data.mul_(nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PPOModel(nn.Module):
    def __init__(self, img_shape, action_space):
        super(PPOModel, self).__init__()
        hidden_size = 512
        self.base = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, hidden_size),
            nn.ReLU()
        ).apply(weights_init_head)

        self.critic_linear = nn.Linear(hidden_size, 1).apply(init_fn)
        self.dist = Categorical(512, action_space.n)

        self.icm = ICM(img_shape[0], action_space.n, 512)
        self.eta = 0.01

    def forward(self, states):
        features = self.base(states / 255)
        value = self.critic_linear(features)
        actions = self.dist.sample(features)
        action_log_probs, dist_entropy, action_probs = \
            self.dist.logprobs_and_entropy(features, actions)

        return value, actions, action_log_probs, action_probs

    def calculate_value(self, states):
        features = self.base(states / 255)
        value = self.critic_linear(features)
        return value

    def evaluate_actions(self, states, actions):
        features = self.base(states / 255)
        value = self.critic_linear(features)
        action_log_probs, dist_entropy, action_probs = \
            self.dist.logprobs_and_entropy(features, actions)

        return value, action_log_probs, dist_entropy, action_probs

    def get_bonus(self, states, next_states, action_probs):
        action_pred, phi2_pred, phi1, phi2 = self.icm(
            states, next_states, action_probs)
        forward_loss = 0.5 * F.mse_loss(phi2_pred, phi2, reduce=False).sum(-1).unsqueeze(-1)
        return self.eta * forward_loss

    def get_icm_loss(self, states, next_states, actions, action_probs):
        action_pred, phi2_pred, phi1, phi2 = self.icm(
            states, next_states, action_probs)
        inverse_loss = F.cross_entropy(action_pred, actions.view(-1))
        forward_loss = 0.5 * F.mse_loss(phi2_pred, phi2.detach(), reduce=False).sum(-1).mean()
        return inverse_loss, forward_loss


class ICM(torch.nn.Module):
    def __init__(self, num_inputs, action_space, state_size):
        super(ICM, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
        ).apply(weights_init_head)

        self.forward_model = nn.Sequential(
            nn.Linear(state_size + action_space, 256),
            nn.ReLU(),
            nn.Linear(256, state_size))
        self.inverse_model = nn.Sequential(
            nn.Linear(state_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
            nn.ReLU())

    def forward(self, state, next_state, action):
        phi1 = self.head(state / 255)
        phi2 = self.head(next_state / 255)

        phi2_pred = self.forward_model(torch.cat([action, phi1], 1))
        action_pred = F.softmax(self.inverse_model(torch.cat([phi1, phi2], 1)), -1)
        return action_pred, phi2_pred, phi1, phi2


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x):
        x = self(x)
        probs = F.softmax(x, dim=1)
        action = probs.multinomial(num_samples=1)
        return action

    def mode(self, x):
        x = self(x)
        probs = F.softmax(x, dim=1)
        action = probs.max(1, keepdim=True)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x, dim=1)
        probs = F.softmax(x, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy, probs
