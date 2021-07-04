import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_gaussian import GaussianSmoothing

from utils import attention, gaussian_attention, multivariate_gaussian_attention

class MVProp(nn.Module):
    def __init__(self, gamma, FLAGS, env):
        super(MVProp, self).__init__()
        self.no_weights = FLAGS.no_vpn_weights
        self.gamma = 0.75
        self.time_scale = FLAGS.time_scale
        self.h = 10
        self.k=35
        self.blur = GaussianSmoothing(1, sigma=0.5)
        self.vpn_double_conv = FLAGS.vpn_double_conv
        self.vpn_masking = FLAGS.vpn_masking
        self.vpn_cnn_masking = FLAGS.vpn_cnn_masking
        self.vpn_cnn_masking_times = FLAGS.vpn_cnn_masking_times
        self.vpn_direction_masking = FLAGS.vpn_direction_masking
        self.reconstruction = FLAGS.reconstruction
        self.wall_thresh = FLAGS.wall_thresh
        self.noisy = FLAGS.noisy

        def _clip_but_pass_gradient(x, l=0., u=1.):
            clip_up = (x > u).float()
            clip_low = (x < l).float()
            return x + ((u - x) * clip_up + (l - x) * clip_low).detach()

        self.wall_prob_act_fn = (lambda x: _clip_but_pass_gradient(x)) if FLAGS.vpn_masking_act else torch.sigmoid
        if self.vpn_double_conv:
            self.h_p = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True)
            self.p = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 3),
                stride=1,
                padding=self._get_padding("SAME", [3,3]),
                bias=True)
        else:
            self.p = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 3),
                stride=1,
                padding=self._get_padding("SAME", [3,3]),
                bias=True)

        # Lower initial weights for more stable training
        self.h_p.weight.data.uniform_(0.0, 0.5)
        self.p.weight.data.uniform_(0.0, 0.5)

        self.vpn_covariance = FLAGS.covariance
        self.mask_cnn1 = nn.Conv2d(
            in_channels=2,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=True)
        self.mask_cnn2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=True)
        self.mask_cnn3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=True)
        def reduction_big(dim):
            x = int((dim - 3)/1 + 1)  # Conv [3x3] Stride 1
            x = int((x -2)/2  +1)  # MaxPooling [2x2]
            x = int((x - 3)/1 + 1)  # Conv [3x3] Stride 1
            x = int((x -2)/2  +1)  # MaxPooling [2x2]
            x = int((x - 3)/1 + 1)  # Conv [3x3] Stride 1
            return x
        def reduction_small(dim):
            x = int((dim - 3)/1 + 1)  # Conv [3x3] Stride 1
            x = int((x - 3)/1 + 1)  # Conv [3x3] Stride 1
            x = int((x - 3)/1 + 1)  # Conv [3x3] Stride 1
            return x
        reduction = reduction_small if env.image_size[0] == 9 else reduction_big
        num_features = reduction(env.image_size[0]) * reduction(env.image_size[1]) * 32
        self.mask_fc1 = nn.Linear(num_features, 64)
        if self.vpn_covariance:
            num_outputs = 5 if self.reconstruction else 3
            self.mask_fc2 = nn.Linear(64, num_outputs)
        else:
            self.mask_fc2 = nn.Linear(64, 1)
        # self.mask_conv1_bn = nn.BatchNorm2d(32)
        # self.mask_conv2_bn = nn.BatchNorm2d(32)
        self.mask_max_pool = nn.MaxPool2d(kernel_size=(2,2))
        self.post_process = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(5,5), stride=1, padding=2, bias=False) if FLAGS.vpn_post_processing else None
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1)

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def _normalize(self, map):
        bs = map.shape[0]
        ones_dim = [1] * (len(map.shape)-1)
        map_min = map.view(bs, -1).min(1)[0]
        map = map - map_min.view(bs, *ones_dim)
        map_max = map.view(bs, -1).max(1)[0]
        map = map / map_max.view(bs, *ones_dim)
        return map

    def mask_image(self, v_map, wall_map, pos_image, image_pos, return_kernel=False):
        assert self.vpn_cnn_masking or self.vpn_masking
        if self.vpn_direction_masking:
            def rot(angle):
                assert len(angle.shape) == 2
                cos_angle = torch.cos(angle)  # [batch, 1]
                sin_angle = torch.sin(angle)  # [batch, 1]
                return torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle], dim=1).view(-1, 2, 2)
            agent_cropout, _, _ = attention(v_map, image_pos, offset=1)
            assert list(agent_cropout.shape) == [v_map.shape[0], 3, 3], agent_cropout.shape
            # Compute the gradient on the immediate agent neighborhood
            g_y = (self.grad_y * agent_cropout).sum(dim=[1, 2]).unsqueeze(1)
            g_x = (self.grad_x * agent_cropout).sum(dim=[1, 2]).unsqueeze(1)
            angle = -torch.atan2(g_y, g_x)
            r = rot(angle)
            rT = rot(angle).transpose(-1, -2)
            cov = torch.bmm(torch.bmm(r, self.cov_base.unsqueeze(0).expand(r.shape[0], -1, -1)), rT)
            att = multivariate_gaussian_attention(v_map, image_pos, cov)[0]
            return att-1 # Move back between -1 and 0

        elif self.vpn_cnn_masking:
            inputs = torch.stack([v_map]+[gaussian_attention(wall_map-1, image_pos, sigma)[0]-1 for sigma in self.sigmas], dim=1)
            cnn1 = F.relu(self.mask_cnn1(inputs)) # 16 channels
            cnn2 = F.relu(self.mask_cnn2(cnn1)) # 32 outputs
            cnn3 = F.relu(self.mask_cnn3(cnn2)) # 16 channels
            cnn4 = self.mask_cnn4(cnn3+cnn1)
            if self.vpn_cnn_masking_times:
                return self._normalize((torch.sigmoid(cnn4).squeeze(1)) * (1+v_map)) - 1

            return (cnn4.squeeze(1)+v_map)

        elif self.vpn_masking:
            if self.noisy:
                v_map_noisy = torch.where(torch.rand_like(v_map) > 0.8, -torch.rand_like(v_map), v_map)
            else:
                v_map_noisy = v_map
            inputs = torch.stack([v_map_noisy, pos_image], dim=1)
            if inputs.shape[-1] > 9 and inputs.shape[-2] > 9:
                x = F.relu(self.mask_max_pool(self.mask_cnn1(inputs)))
                x = F.relu(self.mask_max_pool(self.mask_cnn2(x)))
                flatten = self.mask_cnn3(x).view(inputs.shape[0], -1)
            else:
                x = F.relu(self.mask_cnn1(inputs))
                x = F.relu(self.mask_cnn2(x))
                flatten = self.mask_cnn3(x).view(inputs.shape[0], -1)
            if self.vpn_covariance:
                extra_loss = 0
                if self.reconstruction:
                    sigma1, sigma2, rho, posx, posy = self.mask_fc2(F.relu(self.mask_fc1(flatten))).unbind(1)
                    sigma1, sigma2, rho, posx, posy = F.softplus(sigma1) + 0.01, F.softplus(sigma2) + 0.01, torch.tanh(rho), F.softplus(posx), F.softplus(posy)
                    pos = torch.stack([posx, posy], dim=-1)
                    assert pos.shape == image_pos.shape
                    extra_loss = F.mse_loss(pos, image_pos)
                else:
                    sigma1, sigma2, rho = self.mask_fc2(F.relu(self.mask_fc1(flatten))).unbind(1)
                    sigma1, sigma2, rho = F.softplus(sigma1) + 0.01, F.softplus(sigma2) + 0.01, torch.tanh(rho)
                correlation = rho*sigma1*sigma2
                cov = torch.stack([sigma1*sigma1, correlation, correlation, sigma2*sigma2], dim=1).view(sigma1.shape[0], 2, 2)
                v_modified, kernel = multivariate_gaussian_attention(v_map, image_pos, cov)
            else:
                sigma = F.softplus(self.mask_fc2(F.relu(self.mask_fc1(flatten)))) + 0.01
                v_modified, kernel = gaussian_attention(v_map, image_pos, sigma)
            # Normalize the v_modified
            v_modified = v_modified - 1
            assert (v_modified <= 0).all()
            assert (v_modified >= -1).all()
            
            if return_kernel:
                return kernel

            return v_modified, extra_loss


    def process_image(self, X):
        if self.no_weights:
            p, r = torch.unbind(X, dim=1)
            assert (r>=0).all()
            assert (r<=1).all()
            p = self.blur(p.unsqueeze(1)*self.gamma)
            r = -(1.-r.unsqueeze(1))
        else:
            grid, goal = X.unbind(1)
            assert (goal>=0).all(), goal.min()
            assert (goal<=1).all(), goal.max()
            r = -(1.-goal).unsqueeze(1)
            assert (r<=0).all(), r.max()
            assert (r>=-1).all(), r.min()
            if self.vpn_double_conv:
                h_p = F.relu(self.h_p(grid.unsqueeze(1)))
                p = self.wall_prob_act_fn(self.p(h_p))
            else:
                p = self.wall_prob_act_fn(self.p(grid.unsqueeze(1)))
            if self.wall_thresh:
                p = torch.where(p < 0.4, torch.zeros_like(p), p)
        assert not torch.isnan(p).any()
        return r, p

    def forward(self, X):
        # prep = self.prep2(self.prep(X.unsqueeze(1)))
        r, p = self.process_image(X)

        # MVProp algorithm
        v = self._value_iteration(r, p).squeeze(1)
        # V has shape: [batch_size, img_size, img_size]
        #import pdb
        import matplotlib.pyplot as plt

        return v, p.squeeze(1)

    def _value_iteration(self, r, p):
        v = r
        for i in range(0, self.k):
            v = torch.max(v, r + p * (self.pool(v) - r))
        
        assert (v<=0).all(), v.max()
        return v

    def critic(self, images, pos_image):
        v, _ = self(images)
        if self.post_process is not None:
            v = self.post_process(torch.stack([v, pos_image], dim=1)).squeeze(1)
        return v

    def actor(self, images, pos_image):
        v, p = self(images)
        if self.post_process is not None:
            v = self.post_process(torch.stack([v, pos_image], dim=1)).squeeze(1)
        return v, p
        # near_v, x_coord, y_coords = self._attention(v, pos)
        # return near_v, x_coord, y_coords

    def get_info(self, X):
        with torch.no_grad():
            # prep = self.prep2(self.prep(X.unsqueeze(1)))
            r, p = self.process_image(X)

            # MVProp algorithm
            v = self._value_iteration(r, p)
            return -r.squeeze(), p.squeeze(), v.squeeze()
