from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
from utils import layer
from radam import RAdam
from vpn import MVProp
import utils
from torch_critic import Critic as ClassicCritic

class CriticModel(nn.Module):
    def __init__(self, env, layer_number, FLAGS):
        super().__init__()
        self.q_limit = -FLAGS.time_scale
        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_limit/self.q_init - 1)
        self.no_target_net = FLAGS.no_target_net
        self.time_scale = FLAGS.time_scale

        self.offset = 2

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

    def forward(self, v_image, actor_pixel_selection):
        # v_image shape [batch_size, height, width]
        x_coords = actor_pixel_selection[:, 0]
        y_coords = actor_pixel_selection[:, 1]
        
        assert (x_coords >= 0).all()
        assert (x_coords < v_image.shape[-1]).all(), (torch.min(x_coords), torch.max(x_coords), v_image.shape)
        assert (y_coords >= 0).all(), y_coords.min()
        assert (y_coords < v_image.shape[-2]).all(), (y_coords.max(), v_image.shape[-2])
        x_slice = x_coords.long().unsqueeze(1).unsqueeze(2).expand(-1, v_image.shape[1], -1)
        value = v_image.gather(2, x_slice)
        y_slice = y_coords.long().unsqueeze(1).unsqueeze(2)
        values = value.gather(1, y_slice)
        return values * self.time_scale

class Critic():

    def __init__(self, device, env, layer_number, FLAGS, learning_rate=0.001, gamma=0.98, tau=0.05):
        self.device = device # Session in its TF equivalent
        self.critic_name = 'vpn_critic_' + str(layer_number)
        self.learning_rate = learning_rate
        self.q_limit = -FLAGS.time_scale
        self.gamma = gamma
        self.tau = tau
        self.sac = FLAGS.sac
        self.td3 = FLAGS.td3
        self.vpn = MVProp(self.gamma, FLAGS, env).to(self.device)
        self.no_target_net = FLAGS.no_target_net
        # Create critic network graph
        self.infer_net = CriticModel(env, layer_number, FLAGS).to(device=self.device)
        self.no_weights = FLAGS.no_vpn_weights
        self.vpn_masking = FLAGS.vpn_masking

        self.classic_critic = None
        if FLAGS.boost_vpn:
            self.classic_critic =  ClassicCritic(device, env, layer_number, FLAGS, learning_rate, gamma, tau)

        if not self.no_weights:
            opt_class = RAdam if FLAGS.radam else torch.optim.Adam
            self.optimizer = opt_class(self.vpn.parameters(), learning_rate)

        if FLAGS.no_target_net:
            self.target_net = self.infer_net
            self.vpn_target = self.vpn
        else:
            self.target_net = self.infer_net
            self.vpn_target = MVProp(self.gamma, FLAGS, env).to(self.device)
            self.vpn_target.load_state_dict(self.vpn.state_dict())

        self.get_pos_image = lambda states, images: env.pos_image(states[..., :2], images[:, 0])
        self.get_image_pos = lambda states, images: torch.stack(env.get_image_position(states[..., :2], images), dim=-1)

    def get_Q_value(self,state, goal, action, image):
        with torch.no_grad():
            q = self.infer_net(self.vpn.critic(image), self.get_image_pos(action, image))
            return q

    def get_target_Q_value(self,state, goal, action, image):
        assert not self.no_target_net
        with torch.no_grad():
            q = self.infer_net(self.target_net.critic(image), self.get_image_pos(action, image))
            return q

    def update_target_weights(self):
        for source, target in zip(self.vpn.parameters(), self.vpn_target.parameters()):
            target.data.copy_(self.tau * source + (1.0 - self.tau) * target)

    def _value(self, net, vpn_net, images, states, actions, get_extra_loss=False):
        pos_images = self.get_pos_image(states, images)
        action_image_position = self.get_image_pos(actions, images)
        agent_image_position = self.get_image_pos(states, images)
        vpn_values, vpn_probs = vpn_net.actor(images, pos_images)
        extra_loss = 0
        if self.vpn_masking:
            vpn_values, extra_loss = vpn_net.mask_image(vpn_values, vpn_probs, pos_images, agent_image_position)
        if get_extra_loss:
            return net(vpn_values, action_image_position).squeeze(), extra_loss
        return net(vpn_values, action_image_position).squeeze()

    def update(self, old_states, old_actions, rewards, new_states, old_goals, new_goals, new_actions, is_terminals, is_weights, next_entropy, images, metrics, total_steps_taken=None):
        if self.no_weights:
            return torch.ones_like(rewards)
        if self.classic_critic is not None:
            self.classic_critic.update(old_states, old_actions, rewards, new_states, old_goals, new_actions, is_terminals, is_weights, next_entropy, None, metrics)

        with torch.no_grad():
            wanted_qs = self._value(self.target_net, self.vpn_target, images, new_states, new_actions)
            if self.classic_critic is not None:
                alpha = 1 - (min(total_steps_taken, 1e-6) / 1e-6)
                wanted_qs_classic = torch.stack([net(new_states, new_goals, new_actions) for net in self.classic_critic.target_nets], dim=0)
                wanted_qs_classic = torch.min(wanted_qs_classic, dim=0)[0].detach().squeeze()
                alpha*(wanted_qs_classic) + (1-alpha)*wanted_qs
       
        wanted_qs = rewards + (1 - is_terminals) * (self.gamma * wanted_qs)
        if next_entropy is not None:
            wanted_qs -= next_entropy
        wanted_qs = torch.clamp(wanted_qs, max=0, min=self.q_limit)

        infered_Qs, extra_loss = self._value(self.infer_net, self.vpn, images, old_states, old_actions, get_extra_loss=True)
        if is_weights is None:
            is_weights = torch.ones_like(wanted_qs)
        abs_errors = torch.abs(wanted_qs - infered_Qs).detach()
        self.optimizer.zero_grad()
        difference = (wanted_qs - infered_Qs)
        loss = torch.mean(is_weights * torch.mul(difference, difference), dim=0) + extra_loss
        loss.backward()
        self.optimizer.step()
        
        metrics[self.critic_name + '/Q_loss'] = loss.item()
        metrics[self.critic_name + '/Q_val'] = torch.mean(wanted_qs).item()
        return abs_errors

    def get_gradients_for_actions(self, state, goal, actor, images):
        return None

    def state_dict(self):
        result = {}
        if self.no_weights: return result
        result['target_net'] = self.target_net.state_dict()
        result['infer_net'] = self.infer_net.state_dict()
        result['optimizer'] = self.optimizer.state_dict()
        result['vpn'] = self.vpn.state_dict()
        result['vpn_target'] = self.vpn_target.state_dict()
        return result

    def load_state_dict(self, state_dict):
        if self.no_weights: return
        self.target_net.load_state_dict(state_dict['target_net'])
        self.infer_net.load_state_dict(state_dict['infer_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.vpn.load_state_dict(state_dict['vpn'])
        self.vpn_target.load_state_dict(state_dict['vpn_target'])
