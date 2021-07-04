import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete

class BaseConvNet(nn.Module):
    def __init__(self):
        super(BaseConvNet, self).__init__()
        self.cnn1 = nn.Conv2d(
            in_channels=2,
            out_channels=32,
            kernel_size=(4, 4),
            stride=2,
            padding=0,
            bias=True)
        self.cnn2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=2,
            padding=0,
            bias=True)
        self.cnn3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=True)

    def forward(self, inputs):
        # print("Inputs", inputs.shape)
        x = F.relu((self.cnn1(inputs)))
        # print("CNN1", x.shape)
        x = F.relu(self.cnn2(x))
        # print("CNN2", x.shape)
        return self.cnn3(x).view(inputs.shape[0], -1)

    def get_output_shape(self, shape):
        def reduction(dim):
            x = int((dim - 4)/2 + 1)  # Conv [4x4] Stride 2
            x = int((x - 3)/2 + 1)  # Conv [3x3] Stride 2
            x = int(x - 2)  # Conv [3x3] Stride 1
            return x
        return reduction(shape[0]) * reduction(shape[1]) * 32

class MLP(nn.Module):
    def __init__(self,
                 layers,
                 activation=torch.tanh,
                 output_activation=None,
                 output_squeeze=False, 
                 init_bias=0):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        self.init_bias = init_bias

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) + self.init_bias
        else:
            x = self.output_activation(self.layers[-1](x)) + self.init_bias
        return x.squeeze() if self.output_squeeze else x

class ActorCritic(nn.Module):
    def __init__(self,
                 img_shape,
                 action_space,
                 covariance=False,
                 hidden_sizes=(64,),
                 activation=torch.relu,
                 policy=None):
        super(ActorCritic, self).__init__()

        self.base_net = BaseConvNet()
        num_features = self.base_net.get_output_shape(img_shape)
        self.policy = MLP(layers=[num_features]+list(hidden_sizes)+[action_space.shape[0]], 
            activation=torch.relu, output_activation=None, init_bias=0.0)
        self.covariance = covariance

        self.q = MLP(
            layers=[num_features + action_space.shape[0]] + list(hidden_sizes) + [1],
            activation=activation,
            output_squeeze=True)

    def q_function_forward(self, x, action):
        embedding = self.base_net(x)
        return self.q(torch.cat([embedding, action], dim=-1))

    def _pi(self, embedding):
        pi = self.policy(embedding)
        if self.covariance:
            sigma1, sigma2, rho = pi.unbind(dim=1)
            return torch.stack([F.softplus(sigma1)+0.01, F.softplus(sigma2)+0.01, torch.tanh(rho)], dim=1)
        else:
            return F.softplus(pi)+0.01

    def policy_forward(self, x):
        embedding = self.base_net(x)
        return self._pi(embedding)

    def forward(self, x, a):
        embedding = self.base_net(x)
        pi = self._pi(embedding)
        q = self.q(torch.cat([embedding, a], dim=-1))
        q_pi = self.q(torch.cat([embedding, pi], dim=-1))

        return pi, q, q_pi