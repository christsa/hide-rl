import numpy as np
import torch
import torch.nn.functional as F
import gym
import time
import scipy.signal
import bandit.core as core
from collections import defaultdict
from itertools import chain

class ReplayBuffer:
    """
    A buffer for storing trajectories experienced by a Bandit agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, device=None):
        self.obs_buf = torch.zeros(size, 2, *obs_dim, dtype=torch.float32, device=device)
        self.act_buf = torch.zeros(size, act_dim, dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get(self, batch_size):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        idxs = torch.randint(0, self.size, size=(batch_size,), device=self.obs_buf.device)
        return [
            self.obs_buf[idxs], self.act_buf[idxs], self.rew_buf[idxs],
        ]

class Bandit(object):
    
    def __init__(self, 
        env,
        FLAGS,
        device,
        attempt_range=(6, 9),
        actor_critic=core.ActorCritic,
        replay_size=int(1e4),
        tau=0.05,
        lr=1e-3,
        batch_size=1024):
        # Main model
        self.covariance = FLAGS.covariance
        if self.covariance:
            self.action_max = torch.tensor([30, 30, 1], dtype=torch.float32, device=device)
            self.action_min = torch.tensor([0.01, 0.01, -1], dtype=torch.float32, device=device)
        else:
            self.action_max = torch.tensor([30], dtype=torch.float32, device=device)
            self.action_min = torch.tensor([0.01], dtype=torch.float32, device=device)
        self.attempt_range = attempt_range
        action_space = gym.spaces.Box(low=self.action_min.cpu().numpy(), high=self.action_max.cpu().numpy())
        self.main = actor_critic(img_shape=env.image_size, action_space=action_space).to(device)
        self.batch_size = batch_size
        self.act_noise = 0.1

        self.agg_metrics = defaultdict(list)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(env.image_size, action_space.shape[0], env.max_actions, device)
        self.last_inputs, self.last_action, self.last_logp_t, self.last_v_t = None, None, None, None

        # Optimizers
        self.train_pi = torch.optim.Adam(chain(self.main.policy.parameters(), self.main.base_net.parameters()), lr=lr)
        self.train_Q = torch.optim.Adam(chain(self.main.base_net.parameters(), 
            self.main.q.parameters()), lr=lr)

    def get_action(self, o, noise_scale):
        with torch.no_grad():
            pi = self.main.policy_forward(o)
            a = pi + noise_scale * torch.randn_like(pi)
            # Clamp to the limits.
            return torch.max(torch.min(a, self.action_max), self.action_min)

    def update(self, j, metrics):
        self.main.train()
        obs1, acts, rews = self.replay_buffer.get(self.batch_size)
        q1 = self.main.q_function_forward(obs1, acts)

        q_loss = F.mse_loss(q1, rews)

        self.train_Q.zero_grad()
        q_loss.backward()
        self.train_Q.step()

        metrics['bandit/q_loss'] = q_loss.item()
        metrics['bandit/q_val'] = q1.mean().item()

        if j % 2 == 0:
            embedding = self.main.base_net(obs1)
            q1_pi = self.main.q(torch.cat([embedding, self.main._pi(embedding)], dim=-1))

            # TD3 policy loss
            pi_loss = -q1_pi.mean()

            # Delayed policy update
            self.train_pi.zero_grad()
            pi_loss.backward()
            self.train_pi.step()

            metrics['bandit/pi_loss'] = pi_loss.item()

    def get_range(self, wall_probs_image, pos_image, train=False):
        self.main.eval()
        assert len(pos_image.shape) == 3, pos_image.shape
        assert len(wall_probs_image.shape) == 3, wall_probs_image.shape
        inputs = torch.stack([wall_probs_image, pos_image], dim=1)
        a = self.get_action(inputs, self.act_noise * float(train))
        self.last_inputs, self.last_action = inputs, a
        self.agg_metrics['bandit/sigmas'].append(torch.norm(a).item())
        clipped_a = torch.max(torch.min(a, self.action_max), self.action_min)
        if self.covariance:
            sigma1 = clipped_a[:, 0]*clipped_a[:, 0]
            sigma2 = clipped_a[:, 1]*clipped_a[:, 1]
            rho = clipped_a[:, 0]*clipped_a[:, 1]*clipped_a[:, 2]
            return torch.stack([sigma1, rho, rho, sigma2], dim=-1).view(pos_image.shape[0], 2, 2)
        return clipped_a

    def compute_reward(self, num_attempts):
        if (num_attempts <= self.attempt_range[1] and num_attempts >= self.attempt_range[0]):
            return 0
        else:
            return -1
        # return reward_utils.tolerance(num_attempts, bounds=self.attempt_range, margin=20, sigmoid='linear', value_at_margin=0)

    def store_reward(self, num_attempts):
        if self.last_action is None or self.last_action.shape[0] > 1:
            return
        # save and log
        reward = self.compute_reward(num_attempts)
        self.replay_buffer.store(self.last_inputs, self.last_action, reward)
        self.agg_metrics['bandit/rewards'].append(reward)
        self.last_inputs, self.last_action = None, None

    def state_dict(self):
        return {
            'pi_optimizer': self.train_pi.state_dict(),
            'Q_optimizer': self.train_Q.state_dict(),
            'actor_critic': self.main.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.train_pi.load_state_dict(state_dict['pi_optimizer'])
        self.train_Q.load_state_dict(state_dict['Q_optimizer'])
        self.main.load_state_dict(state_dict['actor_critic'])