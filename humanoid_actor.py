import math
import torch
import numpy as np
from utils import layer
from torch.nn import functional as F
from torch import nn
from radam import RAdam

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        self.goal_dim = env.subgoal_dim
        self.state_dim = env.state_dim
        num_inputs = self.state_dim + 1
        self.features_fn = lambda states, goals: torch.cat([self.get_angle_from_goal_coords(goals, states), states], dim=-1)
        self.action_space_bounds = env.action_bounds
        self.action_offset = env.action_offset

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, env.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, env.action_dim)

        self.apply(weights_init_)
        # self.pi = torch.FloatTensor(math.pi)

        # action rescaling
        self.action_scale = torch.FloatTensor(
            self.action_space_bounds)
        self.action_bias = torch.FloatTensor(
            self.action_offset)

    def degrees(self, rad):
        import math
        return 180. * rad / math.pi

    def quaternion_to_euler(self, w, x, y, z):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = self.degrees(torch.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = self.degrees(torch.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = torch.atan2(t3, t4)

        return X, Y, Z

    def get_angle_from_goal_coords(self, goal, state):
        _, _, yaw = self.quaternion_to_euler(state[..., 3],state[..., 4],state[..., 5],state[..., 6])
        walk_target_theta = torch.atan2( goal[..., 1], goal[..., 0])
        # walk_target_dist  = torch.norm( [goal[..., 1], goal[..., 0]] )
        angle_to_target = walk_target_theta - yaw
        return angle_to_target.view(state.shape[0], 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, goal):
        inputs = self.features_fn(state, goal)
        mean, log_std = self.forward(inputs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        # self.pi = self.pi.to(device)
        return super(GaussianPolicy, self).to(device)

class HumanoidActor():

    def __init__(self, device, env, FLAGS):
        super().__init__()
        self.device = device
        self.actor_name = 'actor_0'

        self.infer_net = GaussianPolicy(env).to(device=self.device)

    def get_action(self, state, goal, image, noise=False, symbolic=False):
        with torch.no_grad():
            return self.infer_net.sample(state, goal)[2]

    def state_dict(self):
        return {
            'infer_net': self.infer_net.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.infer_net.load_state_dict(state_dict['infer_net'])

    def update(self, state, goal, action_derivs, next_batch_size, metrics, goal_derivs=None):
        raise ValueError("Not implemented.")
