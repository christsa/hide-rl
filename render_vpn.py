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
from utils import gaussian_attention, attention, argmax, softargmax

import matplotlib.pyplot as plt

import importlib

def show(situation, maze_image, att, v_att, position, goal_position):
    fig, ax = plt.subplots(4, 1)
    im3 = ax[0].imshow(maze_image)
    plt.colorbar(im3, ax=ax[0])
    ax[0].scatter(position[0], position[1], c='red', marker='x')
    ax[0].scatter(goal_position[0], goal_position[1], c='black', marker='x')
    im0 = ax[1].imshow(situation)
    plt.colorbar(im0, ax=ax[1])
    im1 = ax[2].imshow(v_att)
    plt.colorbar(im1, ax=ax[2])
    ax[2].scatter(position[0], position[1], c='red', marker='x')
    im2 = ax[3].imshow(att)
    plt.colorbar(im2, ax=ax[3])
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
    current_state = torch.tensor(env.reset_sim(agent.goal_array[agent.FLAGS.layers - 1]), device=agent.device, dtype=torch.float32)
    goal = torch.tensor(env.get_next_goal(None), dtype=torch.float32, device=agent.device)
    current_image = torch.tensor(env.take_snapshot(), device=agent.device, dtype=torch.float32)
    position_image = env.pos_image(current_state[:2], current_image)
    current_image = torch.stack([current_image, env.pos_image(goal, current_image)], dim=0)
    r, p, v = buffer_images_v = vpn.get_info(current_image.unsqueeze(0))
    image_position = torch.stack(env.get_image_position(current_state.unsqueeze(0), current_image.unsqueeze(0)), dim=-1)
    goal_image_position = torch.stack(env.get_image_position(goal.unsqueeze(0), current_image.unsqueeze(0)), dim=-1)
    v_att = gaussian_attention(v.unsqueeze(0), image_position, sigma=2)[0].unsqueeze(1)
    att, _, _ = attention(v.unsqueeze(0), image_position, offset=2)
    probs = torch.softmax((v_att*1000000).view(v_att.shape[0], -1), dim=1).view(*v_att.shape)

    argmax_pos = argmax(v_att, 
                        torch.zeros(1,1, device=v_att.device, dtype=torch.int64).expand(v_att.shape[0], -1),
                        torch.zeros(1,1, device=v_att.device, dtype=torch.int64).expand(v_att.shape[0], -1))
    pixel_pos = softargmax(probs.squeeze(1), 
                    torch.arange(v_att.shape[-2], dtype=torch.float32, device=v_att.device).unsqueeze(0).expand(v_att.shape[0], -1), 
                    torch.arange(v_att.shape[-1], dtype=torch.float32, device=v_att.device).unsqueeze(0).expand(v_att.shape[0], -1))

    print(argmax_pos, env.get_env_position(argmax_pos, current_image), current_state[:2])
    show(torch.cat([position_image, r, p, 1+v], dim=-1).cpu().numpy(), 
        current_image[0].squeeze().cpu().numpy(),
        att.squeeze().cpu().numpy(), 
        v_att.squeeze().cpu().numpy(), 
        image_position.squeeze().cpu().numpy(),
        goal_image_position.squeeze().cpu().numpy()
        )
    r, p, v = r.unsqueeze(0), p.unsqueeze(0), v.unsqueeze(0)
    torchvision.utils.save_image([r,p,1+v], os.path.join(agent.model_dir, "vpn_render.png"))
    del agent
    del env