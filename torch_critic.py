from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
from utils import layer
from radam import RAdam

class CriticModel(nn.Module):
    def __init__(self, env, layer_number, FLAGS):
        super().__init__()
        self.negative_distance = FLAGS.negative_distance
        if not self.negative_distance:
            self.q_limit = -FLAGS.time_scale
            # Set parameters to give critic optimistic initialization near q_init
            self.q_init = -0.067
            self.q_offset = -np.log(self.q_limit/self.q_init - 1)
        self.no_target_net = FLAGS.no_target_net

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1 or (layer_number == FLAGS.layers -2 and FLAGS.oracle):
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        mask_global_info = FLAGS.mask_global_info

        if mask_global_info and layer_number == 1 and FLAGS.layers == 3:
            if FLAGS.relative_subgoals:
                self.features_dim = self.goal_dim + action_dim
                self.features_fn = lambda states, goals, actions: torch.cat([goals, actions], dim=-1)
            else:
                self.features_dim = self.goal_dim + action_dim + 2
                self.features_fn = lambda states, goals, actions: torch.cat([states[:,:2], goals, actions], dim=-1)
        elif mask_global_info and layer_number == 0:
            if FLAGS.relative_subgoals:
                self.features_dim = self.state_dim + self.goal_dim + action_dim - 2
                self.features_fn = lambda states, goals, actions: torch.cat([states[:, 2:], goals, actions], dim=-1)
            else:
                self.features_dim = self.state_dim + self.goal_dim + action_dim
                self.features_fn = lambda states, goals, actions: torch.cat([states, goals, actions], dim=-1)
        else:
            self.features_dim = self.state_dim + self.goal_dim + action_dim
            self.features_fn = lambda states, goals, actions: torch.cat([states, goals, actions], dim=-1)

        self.fc1 = self.linear(self.features_dim, 64)
        self.fc2 = self.linear(64, 64)
        self.fc3 = self.linear(64, 64)
        self.fc4 = self.linear(64, 1, output=True)

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

    def forward(self, states, goals, actions):
        features = self.features_fn(states, goals, actions)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if self.negative_distance:
            return x
        else:
            # A q_offset is used to give the critic function an optimistic initialization near 0
            return torch.sigmoid(x + self.q_offset) * self.q_limit

class Critic():

    def __init__(self, device, env, layer_number, FLAGS, vpn=None, learning_rate=0.001, gamma=0.98, tau=0.05):
        self.device = device # Session in its TF equivalent
        self.critic_name = 'critic_' + str(layer_number)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.sac = FLAGS.sac
        self.td3 = FLAGS.td3
        self.time_scale = FLAGS.time_scale
        self.negative_distance = FLAGS.negative_distance
        if not self.negative_distance:
            self.q_limit = -FLAGS.time_scale
            # Set parameters to give critic optimistic initialization near q_init
            self.q_init = -0.067
            self.q_offset = -np.log(self.q_limit/self.q_init - 1)
        self.no_target_net = FLAGS.no_target_net
        # Create critic network graph
        self.infer_nets = [CriticModel(env, layer_number, FLAGS).to(device=self.device) for i in range(FLAGS.num_Qs)]
        opt_class = RAdam if FLAGS.radam else torch.optim.Adam
        self.optimizers = [opt_class(infer_net.parameters(), learning_rate) for infer_net in self.infer_nets]

        if FLAGS.no_target_net:
            self.target_nets = self.infer_nets
        else:
            self.target_nets = [CriticModel(env, layer_number, FLAGS).to(device=self.device) for i in range(FLAGS.num_Qs)]

    def get_Q_value(self,state, goal, action, image):
        with torch.no_grad():
            Q_s = torch.stack([net(state, goal, action) for net in self.infer_nets], dim=0)
            return torch.min(Q_s, dim=0)[0]

    def get_target_Q_value(self,state, goal, action, image):
        assert not self.no_target_net
        with torch.no_grad():
            Q_s = torch.stack([net(state, goal, action) for net in self.target_nets], dim=0)
            return torch.min(Q_s, dim=0)[0]

    def update_target_weights(self):
        for infer_net, target_net in zip(self.infer_nets, self.target_nets):
            for source, target in zip(infer_net.parameters(), target_net.parameters()):
                target.data.copy_(self.tau * source + (1.0 - self.tau) * target)

    def update(self, old_states, old_actions, rewards, new_states, old_goals, new_goals, new_actions, is_terminals, is_weights, next_entropy, images, metrics, total_steps_taken=None):
        with torch.no_grad():
            wanted_qs = torch.stack([net(new_states, new_goals, new_actions) for net in self.target_nets], dim=0)
            wanted_qs = torch.min(wanted_qs, dim=0)[0].detach().squeeze()
       

        # for i in range(len(wanted_qs)):
        #     if is_terminals[i]:
        #         wanted_qs[i] = rewards[i]
        #     else:
        #         wanted_qs[i] = rewards[i] + self.gamma * wanted_qs[i]
        #     if next_entropy is not None:
        #         wanted_qs[i] = wanted_qs[i] - next_entropy[i]

        #     # Ensure Q target is within bounds [-self.time_limit,0]
        #     if not (self.negative_distance or self.sac):
        #         wanted_qs[i] = max(min(wanted_qs[i],0), self.q_limit)
        #         assert wanted_qs[i] <= 0 and wanted_qs[i] >= self.q_limit, "Q-Value target not within proper bounds"
        wanted_qs = rewards + (1 - is_terminals) * (self.gamma * wanted_qs)
        if next_entropy is not None:
            wanted_qs -= next_entropy
        if not (self.negative_distance or self.sac):
            wanted_qs = torch.clamp(wanted_qs, max=0, min=self.q_limit)
        wanted_qs = wanted_qs.unsqueeze(-1)

        infered_Qs = [net(old_states, old_goals, old_actions) for net in self.infer_nets]
        infered_Qs_min = torch.min(torch.stack(infered_Qs, dim=0), dim=0)[0]
        if is_weights is None:
            is_weights = torch.ones_like(wanted_qs)
        abs_errors = torch.abs(wanted_qs - (infered_Qs_min if (self.sac or self.td3) else infered_Qs[0])).detach()
        self.loss_val = 0
        for i, infered_q in enumerate(infered_Qs):
            self.optimizers[i].zero_grad()
            difference = (wanted_qs - infered_q)
            loss = torch.mean(is_weights * torch.mul(difference, difference), dim=0)
            self.loss_val += loss
            loss.backward()
            self.optimizers[i].step()
        
        metrics[self.critic_name + '/Q_loss'] = self.loss_val.item() / len(self.infer_nets)
        metrics[self.critic_name + '/Q_val'] = torch.mean(wanted_qs).item()
        return abs_errors

    def get_gradients_for_actions(self, state, goal, actor, images):
        action = actor.get_action(state, goal, images, symbolic=True)
        Q_s = torch.stack([net(state, goal, action) for net in self.infer_nets], dim=0)
        Q = torch.min(Q_s, dim=0)[0] if self.sac else Q_s[0]
        # We are not returning gradients, but the actor expects it.
        return Q

    def get_gradients_for_goals(self, state, goal, action):
        Q_s = torch.stack([net(state, goal, action) for net in self.infer_nets], dim=0)
        Q = torch.min(Q_s, dim=0)[0] if self.sac else Q_s[0]
        # We are not returning gradient, but the actor expects it.
        return Q

    def state_dict(self):
        result = {}
        for i, (infer_net, target_net, optimizer) in enumerate(zip(self.infer_nets, self.target_nets, self.optimizers)):
            result['target_net_%d' % i] = self.target_nets[i].state_dict()
            result['infer_net_%d' % i] = self.infer_nets[i].state_dict()
            result['optimizer_%d' % i] = self.optimizers[i].state_dict()
        return result

    def load_state_dict(self, state_dict):
        for i, (infer_net, target_net) in enumerate(zip(self.infer_nets, self.target_nets)):
            self.target_nets[i].load_state_dict(state_dict['target_net_%d' % i])
            self.infer_nets[i].load_state_dict(state_dict['infer_net_%d' % i])
            self.optimizers[i].load_state_dict(state_dict['optimizer_%d' % i])
