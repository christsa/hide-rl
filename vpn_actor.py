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

        self.cnn1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.cnn2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.cnn3 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=True)

    def get_image_probs(self, v_image):
        # v_image shape [batch_size, height, width]
        batch_size = v_image.shape[0]
        cnn1 = F.relu(self.cnn1(v_image.unsqueeze(1)))
        cnn2 = F.relu(self.cnn2(cnn1))
        cnn3 = self.cnn3(cnn2 + cnn1).view(batch_size, -1)
        output = F.softmax(cnn3, dim=-1).view(batch_size, v_image.shape[-2], v_image.shape[-1])
        return output

    def forward(self, v_image, pos_coords, sigma=None, pixel_probs=False):
        batch_size = v_image.shape[0]
        if self.gaussian_attention:
            assert sigma is not None
            if self.covariance:
                masked_v = utils.multivariate_gaussian_attention(v_image, pos_coords, cov=sigma)[0]
            else:
                masked_v = utils.gaussian_attention(v_image, pos_coords, sigma=sigma)[0]
            x_coords = torch.arange(v_image.shape[-2], dtype=torch.float32, device=v_image.device).unsqueeze(0).expand(batch_size, -1)
            y_coords = torch.arange(v_image.shape[-1], dtype=torch.float32, device=v_image.device).unsqueeze(0).expand(batch_size, -1)
        elif self.no_attention:
            masked_v = v_image
            x_coords = torch.arange(v_image.shape[-2], dtype=torch.float32, device=v_image.device).unsqueeze(0).expand(batch_size, -1)
            y_coords = torch.arange(v_image.shape[-1], dtype=torch.float32, device=v_image.device).unsqueeze(0).expand(batch_size, -1)
        else:
            masked_v, x_coords, y_coords = utils.attention(v_image, pos_coords, offset=self.offset)
        probs_image = self.get_image_probs(masked_v)
        if pixel_probs:
            return probs_image
        pixel_pos = utils.softargmax(probs_image, x_coords, y_coords)

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
        self.sigma_val = 2 if FLAGS.gaussian_attention else None
        self.vpn_masking = FLAGS.vpn_masking
        # self.batch_size = batch_size

        self.vpn = vpn
        self.infer_net = ActorModel(layer_number, env, FLAGS).to(device=self.device)

        # Create target actor network
        if FLAGS.no_target_net:
            self.target_net = self.infer_net
        else:
            self.target_net = ActorModel(layer_number, env, FLAGS).to(device=self.device)
        
        opt_class = RAdam if FLAGS.radam else torch.optim.Adam
        self.optimizer = opt_class(self.infer_net.parameters(), lr=learning_rate)

        self.get_pos_image = lambda states, images: env.pos_image(states[..., :2], images[:, 0])
        self.get_image_location = lambda states, images: torch.stack(env.get_image_position(states[..., :2], images), dim=-1)
        self.get_env_location = lambda states, images: torch.stack(env.get_env_position(states[..., :2], images), dim=-1)

    def sigma(self, vpn_values, state, image, noise=True):
        return self.sigma_val

    def _vpn_values(self, state, image, image_location):
        pos_image = self.get_pos_image(state, image)
        vpn_values, vpn_probs = self.vpn.actor(image, pos_image)
        if self.vpn_masking:
            vpn_values = self.vpn.mask_image(vpn_values, vpn_probs, pos_image, image_location)[0]
        return vpn_values

    def _action_with_intermediate_results(self, net, state, image, noise=True, pixel_probs=False):
        image_location = self.get_image_location(state, image)
        vpn_values = self._vpn_values(state, image, image_location)
        sigma = self.sigma(vpn_values.squeeze(1), state, image, noise)
        return net(vpn_values, image_location, sigma, pixel_probs=pixel_probs), image_location, vpn_values, sigma

    def _action(self, net, state, image, noise=True, pixel_probs=False):
        return self._action_with_intermediate_results(net, state, image, noise, pixel_probs)[0]

    def get_action(self, state, goal, image, noise=True, symbolic=False):
        if not symbolic:
            with torch.no_grad():
                pixel_pos = self._action(self.infer_net, state, image, noise=noise, pixel_probs=False)
                return self.get_env_location(pixel_pos, image)
        else:
            pixel_probs = self._action(self.infer_net, state, image, pixel_probs=True)
            return pixel_probs

    def get_target_action(self, state, goal, image, symbolic=False):
        if not symbolic:
            with torch.no_grad():
                pixel_pos = self._action(self.target_net, state, image, pixel_probs=False)
                return self.get_env_location(pixel_pos, image), None
        else:
            pixel_probs = self._action(self.target_net, state, image, pixel_probs=True)
            return pixel_probs, None

    def get_target_action_for_goal_grads(self, state, image):
        pixel_pos = self._action(self.target_net, state, image)
        return self.get_env_location(pixel_pos, image)

    def update_target_weights(self):
        for source, target in zip(self.infer_net.parameters(), self.target_net.parameters()):
            target.data.copy_(self.tau * source + (1.0 - self.tau) * target)


    def state_dict(self):
        result = {
            'target_net': self.target_net.state_dict(),
            'infer_net': self.infer_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return result

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
