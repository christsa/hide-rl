import math
import torch
import numpy as np
from utils import layer
from torch.nn import functional as F
from torch import nn
from radam import RAdam

class HERPolicy(nn.Module):
    def __init__(self, env, hidden_dim=256, norm_clip=5, obs_clip=200):
        super(HERPolicy, self).__init__()
        self.goal_dim = env.subgoal_dim
        self.state_dim = env.state_dim
        num_inputs = self.state_dim + self.goal_dim
        self.features_fn = lambda states, goals: torch.cat([states, goals], dim=-1)
        self.action_space_bounds = env.action_bounds
        self.action_offset = env.action_offset
        self.obs_clip = obs_clip

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, env.action_dim)

        # Normalizer fields
        self.norm_clip = norm_clip
        self.obs_mean = torch.nn.Parameter(torch.zeros(self.state_dim))
        self.obs_std = torch.nn.Parameter(torch.zeros(self.state_dim))
        self.goal_mean = torch.nn.Parameter(torch.zeros(self.goal_dim))
        self.goal_std = torch.nn.Parameter(torch.zeros(self.goal_dim))

    def normalize_obs(self, value):
        assert len(value.shape) == 2
        assert value.shape[1] == self.obs_mean.shape[0], (value.shape, self.obs_mean.shape)
        exmean = self.obs_mean.expand(value.shape[0], -1)
        exstd = self.obs_std.expand(value.shape[0], -1)
        return torch.clamp((value - exmean) / exstd, min=-self.norm_clip, max=self.norm_clip)

    def normalize_goal(self, value):
        assert len(value.shape) == 2
        assert value.shape[1] == self.goal_mean.shape[0], (value.shape, self.goal_mean.shape)
        exmean = self.goal_mean.expand(value.shape[0], -1)
        exstd = self.goal_std.expand(value.shape[0], -1)
        return torch.clamp((value - exmean) / exstd, min=-self.norm_clip, max=self.norm_clip)

    def forward(self, state, goal):
        state = torch.clamp(state, -self.obs_clip, self.obs_clip)
        goal = torch.clamp(goal, -self.obs_clip, self.obs_clip)
        state = self.normalize_obs(state)
        goal = self.normalize_goal(goal)
        x = self.features_fn(state, goal)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return torch.tanh(self.linear4(x)) * torch.from_numpy(self.action_space_bounds).to(device=x.device, dtype=torch.float32) + torch.from_numpy(self.action_offset).to(device=x.device, dtype=torch.float32)

class HERActor():

    def __init__(self, device, env, FLAGS):
        super().__init__()
        self.device = device
        self.actor_name = 'actor_0'

        self.infer_net = HERPolicy(env).to(device=self.device)

    def get_action(self, state, goal, image, noise=False, symbolic=False):
        with torch.no_grad():
            return self.infer_net(state, goal)

    def state_dict(self):
        return {
            'infer_net': self.infer_net.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.infer_net.load_state_dict(state_dict['infer_net'])

    def update(self, state, goal, action_derivs, next_batch_size, metrics, goal_derivs=None):
        raise ValueError("Not implemented.")
