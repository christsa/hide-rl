import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from radam import RAdam
import utils

class ActorModel(nn.Module):

    def __init__(self, layer_number, env, FLAGS):
        super().__init__()

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        # Determine symmetric range of subgoal space and offset
        self.action_space_bounds = env.subgoal_bounds_symmetric
        self.action_offset = env.subgoal_bounds_offset
        self.gaussian_attention = FLAGS.gaussian_attention
        self.covariance = FLAGS.covariance
        self.no_attention = FLAGS.no_attention

        self.actor_name = 'actor_' + str(layer_number)
        self.offset = 2
        self.vpn_dqn = FLAGS.vpn_dqn

    def forward(self, v_image, pos_coords, sigma=None, pixel_probs=False):
        if self.gaussian_attention:
            if self.covariance:
                cropped_v = utils.multivariate_gaussian_attention(v_image, pos_coords, cov=sigma)[0]
            else:
                cropped_v = utils.gaussian_attention(v_image, pos_coords, sigma=sigma)[0]
            x_coords = torch.zeros(1,1, device=v_image.device, dtype=torch.int64).expand(v_image.shape[0], -1)
            y_coords = torch.zeros(1,1, device=v_image.device, dtype=torch.int64).expand(v_image.shape[0], -1)
        elif self.no_attention:
            cropped_v = v_image
            x_coords = torch.zeros(1,1, device=v_image.device, dtype=torch.int64).expand(v_image.shape[0], -1)
            y_coords = torch.zeros(1,1, device=v_image.device, dtype=torch.int64).expand(v_image.shape[0], -1)
        else:
            cropped_v, x_coords, y_coords = utils.attention(v_image, pos_coords, offset=self.offset)
        pixel_pos = utils.argmax(cropped_v, x_coords, y_coords)
        # output = F.softmax(cropped_v, dim=-1).view(v_image.shape[0], cropped_v.shape[-2], cropped_v.shape[-1])
        # pixel_pos2 = utils.softargmax(output, x_coords, y_coords)

        height, width = v_image.shape[-2:]
        assert (pixel_pos[:,1] >= 0).all()
        assert (pixel_pos[:,1] < height).all()
        assert (pixel_pos[:,0] >= 0).all()
        assert (pixel_pos[:,0] < width).all()

        return pixel_pos

class Actor():

    def __init__(self,
            device,
            env,
            batch_size,
            layer_number,
            FLAGS,
            vpn,
            learning_rate=0.001,
            tau=0.05):
        super().__init__()
        self.device = device
        
        self.actor_name = 'actor_' + str(layer_number)
        self.learning_rate = learning_rate
        self.time_scale = FLAGS.time_scale
        # self.exploration_policies = exploration_policies
        self.tau = tau
        self.sigma_val = 2. if FLAGS.gaussian_attention else None
        self.vpn_masking = FLAGS.vpn_masking
        # self.batch_size = batch_size

        self.vpn = vpn
        self.infer_net = ActorModel(layer_number, env, FLAGS).to(device=self.device)

        # Create target actor network
        self.target_net = self.infer_net

        self.get_pos_image = lambda states, images: env.pos_image(states[..., :2], images[:, 0])
        self.get_image_location = lambda states, images: torch.stack(env.get_image_position(states[..., :2], images), dim=-1)
        self.get_env_location = lambda states, images: torch.stack(env.get_env_position(states[..., :2], images), dim=-1)
    
    def sigma(self, vpn_values, state, image):
        return self.sigma_val

    def _action(self, net, state, image):
        pos_image = self.get_pos_image(state, image)
        image_location = self.get_image_location(state, image)
        vpn_values, vpn_probs = self.vpn.actor(image, pos_image)
        if self.vpn_masking:
            vpn_values = self.vpn.mask_image(vpn_values, vpn_probs, pos_image, image_location)[0]
        sigma = self.sigma(vpn_values.squeeze(1), state, image)
        return net(vpn_values, image_location, sigma)

    def get_action(self, state, goal, image, noise=False, symbolic=False):
        if not symbolic:
            with torch.no_grad():
                pixel_pos = self._action(self.infer_net, state, image)
                return self.get_env_location(pixel_pos, image)
        else:
            pixel_pos = self._action(self.infer_net, state, image)
            return self.get_env_location(pixel_pos, image)

    def get_target_action(self, state, goal, image, symbolic=False):
        if not symbolic:
            with torch.no_grad():
                pixel_pos = self._action(self.target_net, state, image)
                return self.get_env_location(pixel_pos, image), None
        else:
            pixel_pos = self._action(self.target_net, state, image)
            return self.get_env_location(pixel_pos, image), None

    def get_target_action_for_goal_grads(self, state, image):
        pixel_pos = self._action(self.target_net, state, image)
        return self.get_env_location(pixel_pos, image)

    def update_target_weights(self):
        for source, target in zip(self.infer_net.parameters(), self.target_net.parameters()):
            target.data.copy_(self.tau * source + (1.0 - self.tau) * target)

    def state_dict(self):
        return {}
        
    def load_state_dict(self, state_dict):
        assert len(state_dict) == 0
        pass

    def update(self, state, goal, action_derivs, next_batch_size, metrics):
        pass
