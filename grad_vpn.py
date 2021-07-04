"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""

from options import parse_options
from agent import Agent
from run_HAC import run_HAC
import torch
import torchvision
import os
import math
from utils import gaussian_attention, attention, argmax, softargmax, multivariate_gaussian_attention

import matplotlib.pyplot as plt

import importlib

def show(situation, maze_image, v, att, v_att, position, goal_position):
    fig, ax = plt.subplots(5, 1)
    im3 = ax[0].imshow(maze_image)
    plt.colorbar(im3, ax=ax[0])
    ax[0].scatter(position[0], position[1], c='red', marker='x')
    ax[0].scatter(goal_position[0], goal_position[1], c='black', marker='x')
    im0 = ax[1].imshow(situation)
    plt.colorbar(im0, ax=ax[1])
    im1 = ax[2].imshow(v_att)
    plt.colorbar(im1, ax=ax[2])
    ax[2].scatter(position[0], position[1], c='red', marker='x')
    im2 = ax[3].imshow(att.cpu().numpy())
    plt.colorbar(im2, ax=ax[3])

    #Black and white input image x, 1x1xHxW
    a = torch.Tensor([[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]]).to(device=att.device)

    def rot(angle):
        return torch.tensor([[torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]], device=att.device)

    a = a.view((1,1,3,3))
    G_x = torch.nn.functional.conv2d(att.unsqueeze(0).unsqueeze(1), a)

    b = torch.Tensor([[1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]]).to(device=att.device)

    b = b.view((1,1,3,3))
    # G_y = torch.nn.functional.conv2d(att.unsqueeze(0).unsqueeze(1), b)
    G_y = (b * att.unsqueeze(0).unsqueeze(1)).sum()
    G_x = (a * att.unsqueeze(0).unsqueeze(1)).sum()
    print(G_y.shape, G_x.shape)

    # G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))

    # Compute the angle
    print("gradients", G_x, G_y)
    angle = -torch.atan2(G_y, G_x)
    r = rot(angle)
    print("Angle: ", angle, angle*57.2957795)
    cov = torch.tensor([[.1, 0.], [0., 3.]], device=v.device)
    cov = torch.matmul(torch.matmul(r, cov), r.t())
    new_att = multivariate_gaussian_attention(v.unsqueeze(0), torch.tensor(position, device=att.device, dtype=torch.float32).unsqueeze(0), cov.unsqueeze(0))[0]
    new_att = new_att.unsqueeze(1)
    # new_att = torch.where(new_att >= torch.mean(new_att), torch.ones_like(new_att), torch.zeros_like(new_att))
    # new_att *= (1+v)
    argmax_pos = argmax(new_att.squeeze(1), 
                        torch.zeros(1,1, device=att.device, dtype=torch.int64).expand(1, -1),
                        torch.zeros(1,1, device=att.device, dtype=torch.int64).expand(1, -1)).squeeze().cpu().numpy()

    probs = torch.softmax((new_att*1e10).view(new_att.shape[0], -1), dim=1).view(*new_att.shape).squeeze(1)
    print("probs shape", probs.shape)
    pixel_pos = softargmax(probs, 
                    torch.arange(v_att.shape[-2], dtype=torch.float32, device=att.device).unsqueeze(0), 
                    torch.arange(v_att.shape[-1], dtype=torch.float32, device=att.device).unsqueeze(0)).squeeze().cpu().numpy()
    print('pixel pos ', pixel_pos.shape)
    im4 = ax[4].imshow(new_att.squeeze().cpu().numpy())
    plt.colorbar(im4, ax=ax[4])
    ax[4].scatter(position[0], position[1], c='red', marker='x')
    ax[4].scatter(argmax_pos[0], argmax_pos[1], c='black', marker='x')
    ax[4].scatter(pixel_pos[0], pixel_pos[1], c='green', marker='x')
    print(argmax_pos, pixel_pos)

    plt.show()

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()

# Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file. 
# Load the variant dynamically from the variant folder based on the name.
module = importlib.import_module(f"variants.{FLAGS.variant}", __name__)

# Begin training
if FLAGS.exp_num >= 0 or FLAGS.test:
    agent, env = module.design_agent_and_env(FLAGS)
    vpn = agent.layers[agent.FLAGS.layers-1].critic.vpn
    current_state = torch.tensor(env.reset_sim(agent.goal_array[agent.FLAGS.layers - 1]), device=agent.sess, dtype=torch.float32)
    goal = torch.tensor(env.get_next_goal(None), dtype=torch.float32, device=agent.sess)
    current_image = torch.tensor(env.take_snapshot(), device=agent.sess, dtype=torch.float32)
    position_image = env.pos_image(current_state[:2], current_image)
    current_image = torch.stack([current_image, env.pos_image(goal, current_image)], dim=0)
    r, p, v = buffer_images_v = vpn.get_info(current_image.unsqueeze(0))
    image_position = torch.stack(env.get_image_position(current_state.unsqueeze(0), current_image.unsqueeze(0)), dim=-1)
    goal_image_position = torch.stack(env.get_image_position(goal.unsqueeze(0), current_image.unsqueeze(0)), dim=-1)
    v_att = gaussian_attention(v.unsqueeze(0), image_position, sigma=2)[0].unsqueeze(1)
    # v_att = v_att - v_att.min()
    # print("Max", v_att.max().item(), "min", v_att.min().item())
    att, _, _ = attention(v.unsqueeze(0), image_position, offset=1)
    print(att.shape)
    probs = torch.softmax((v_att*1000000).view(v_att.shape[0], -1), dim=1).view(*v_att.shape)

    argmax_pos = argmax(v_att.squeeze(1), 
                        torch.zeros(1,1, device=v_att.device, dtype=torch.int64),
                        torch.zeros(1,1, device=v_att.device, dtype=torch.int64))
    pixel_pos = softargmax(probs.squeeze(1), 
                    torch.arange(v_att.shape[-2], dtype=torch.float32, device=v_att.device).unsqueeze(0).expand(v_att.shape[0], -1), 
                    torch.arange(v_att.shape[-1], dtype=torch.float32, device=v_att.device).unsqueeze(0).expand(v_att.shape[0], -1))

    print(argmax_pos, env.get_env_position(argmax_pos, current_image), current_state[:2])
    show(torch.cat([position_image, r, p, 1+v], dim=-1).cpu().numpy(), 
        current_image[0].squeeze().cpu().numpy(),
        v,
        att.squeeze(), 
        v_att.squeeze().cpu().numpy(), 
        image_position.squeeze().cpu().numpy(),
        goal_image_position.squeeze().cpu().numpy()
        )
    r, p, v = r.unsqueeze(0), p.unsqueeze(0), v.unsqueeze(0)
    torchvision.utils.save_image([r,p,1+v], os.path.join(agent.model_dir, "vpn_render.png"))
    del agent
    del env