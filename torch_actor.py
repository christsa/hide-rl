import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from radam import RAdam

class ActorModel(nn.Module):

    def __init__(self, layer_number, env, FLAGS):
        super().__init__()
        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = env.action_bounds
            self.action_offset = env.action_offset
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1 or (layer_number == FLAGS.layers -2 and FLAGS.oracle):
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.state_dim = env.state_dim

        if FLAGS.mask_global_info and layer_number == 1 and FLAGS.layers == 3:
            features_dim = self.goal_dim
            self.features_fn = lambda states, goals: goals
        elif FLAGS.mask_global_info and layer_number == 0:
            features_dim = self.state_dim - 2 + self.goal_dim
            self.features_fn = lambda states, goals: torch.cat([states[:,2:], goals], dim=-1)
        else:
            features_dim = self.state_dim + self.goal_dim
            self.features_fn = lambda states, goals: torch.cat([states, goals], dim=-1)

        self.fc1 = self.linear(features_dim, 64)
        self.fc2 = self.linear(64, 64)
        self.fc3 = self.linear(64, 64)
        self.fc4 = self.linear(64, self.action_space_size, output=True)

    def linear(self, fin, fout, output=False):
        linear_layer = nn.Linear(fin, fout)
        if output:
            nn.init.uniform_(linear_layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(linear_layer.bias, -3e-3, 3e-3)
        else:
            fan_in_init = 1 / fin ** 0.5
            nn.init.uniform_(linear_layer.weight, -fan_in_init, fan_in_init)
            nn.init.uniform_(linear_layer.bias, -fan_in_init, fan_in_init)
        return linear_layer

    def forward(self, state, goal):
        features = self.features_fn(state, goal)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x)) * torch.from_numpy(self.action_space_bounds).to(device=x.device, dtype=torch.float32) + torch.from_numpy(self.action_offset).to(device=x.device, dtype=torch.float32)

class Actor():

    def __init__(self,
            device,
            env,
            batch_size,
            layer_number,
            FLAGS,
            vpn=None,
            learning_rate=0.001,
            tau=0.05):
        super().__init__()
        self.device = device
        
        self.time_scale = FLAGS.time_scale
        self.actor_name = 'actor_' + str(layer_number)
        self.learning_rate = learning_rate
        # self.exploration_policies = exploration_policies
        self.tau = tau

        self.infer_net = ActorModel(layer_number, env, FLAGS).to(device=self.device)

        # Create target actor network
        if FLAGS.no_target_net:
            self.target_net = self.infer_net
        else:
            self.target_net = ActorModel(layer_number, env, FLAGS).to(device=self.device)
        
        opt_class = RAdam if FLAGS.radam else torch.optim.Adam
        self.optimizer = opt_class(self.infer_net.parameters(), lr=learning_rate)

    def t(self, numpy_arr):
        return numpy_arr if isinstance(numpy_arr, torch.Tensor) else torch.from_numpy(np.asarray(numpy_arr)).to(device=self.device, dtype=torch.float32)
            
    def get_action(self, state, goal, image, noise=False, symbolic=False):
        if not symbolic:
            with torch.no_grad():
                actions = self.infer_net(state, goal)
        else:
            actions = self.infer_net(state, goal)
        return actions

    def get_target_action(self, state, goal, image, symbolic=False):
        if not symbolic:
            with torch.no_grad():
                actions = self.target_net(state, goal)
        else:
            actions = self.target_net(state, goal)
        return actions, None

    def update_target_weights(self):
        for source, target in zip(self.infer_net.parameters(), self.target_net.parameters()):
            target.data.copy_(self.tau * source + (1.0 - self.tau) * target)


    def state_dict(self):
        return {
            'target_net': self.target_net.state_dict(),
            'infer_net': self.infer_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.target_net.load_state_dict(state_dict['target_net'])
        self.infer_net.load_state_dict(state_dict['infer_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def update(self, state, goal, action_derivs, next_batch_size, metrics):
        self.optimizer.zero_grad()
        loss = -action_derivs.mean()
        loss.backward()
        self.optimizer.step()
        metrics[self.actor_name+'/loss'] = loss.item()