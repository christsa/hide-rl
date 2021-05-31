import torch
import numpy as np
import os
import math

def attention(v, pos, offset):
    # V has shape: [batch_size, img_size, img_size]
    # Axis X and Y are flipped.
    pos_x = pos[:, 1].long()
    pos_y = pos[:, 0].long()
    assert ((pos_x-offset) >= 0).all()
    assert ((pos_y-offset) >= 0).all()
    assert ((pos_x+offset) < v.shape[1]).all()
    assert ((pos_y+offset) < v.shape[2]).all(), (torch.max(pos_x), torch.max(pos_y), v.shape)
    # x_neighbor [batchsize, 2*offset+1]
    x_neighbor = pos_x.unsqueeze(1).expand(-1, 2*offset+1) + torch.arange(-offset, offset+1, dtype=torch.int64, device=v.device).unsqueeze(0).expand(len(pos_x), -1)
    # y_neighbor [batchsize, 2*offset+1]
    y_neighbor = pos_y.unsqueeze(1).expand(-1, 2*offset+1) + torch.arange(-offset, offset+1, dtype=torch.int64, device=v.device).unsqueeze(0).expand(len(pos_y), -1)
    assert (x_neighbor >= 0).all()
    assert (y_neighbor >= 0).all()
    assert (x_neighbor < v.shape[1]).all()
    assert (y_neighbor >= 0).all()
    assert (y_neighbor < v.shape[2]).all()

    slice_s1 = x_neighbor.unsqueeze(-1).expand(-1, -1, v.shape[2])
    action = v.gather(1, slice_s1)  # batch_size, img_size

    slice_s2 = y_neighbor.unsqueeze(1).expand(-1, 2*offset+1, -1)
    action = action.gather(2, slice_s2)
    assert list(action.size()[1:]) == [2*offset+1, 2*offset+1]
    return action, x_neighbor, y_neighbor

def gaussian_attention(v, pos, sigma):
    assert (v <= 0).all()
    # V has shape: [batch_size, height, width]
    # Axis X and Y are flipped.
    assert len(v.shape) == 3
    batch_size, height, width = v.shape
    if isinstance(sigma, torch.Tensor):
        assert list(sigma.shape) == [v.shape[0], 1], sigma.shape
        sigma = sigma.unsqueeze(-1)
    else:
        assert np.isscalar(sigma)

    # Initialize the kernel and prepare the mesh grids
    pos_x = pos[:, 1]
    pos_y = pos[:, 0]
    kernel = torch.ones_like(v)
    grid_mesh_height = torch.arange(height, device=v.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    grid_mesh_width = torch.arange(width, device=v.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)

    # Shift the mesh grids
    grid_mesh_height = grid_mesh_height - pos_x.unsqueeze(1)
    grid_mesh_width = grid_mesh_width - pos_y.unsqueeze(1)
    
    # Expand them to the other dimension
    grid_mesh_height = grid_mesh_height.unsqueeze(2).expand(-1, -1, width)
    grid_mesh_width = grid_mesh_width.unsqueeze(1).expand(-1, height, -1)
    
    distribution = torch.distributions.Normal(0, sigma)
    
    kernel = (distribution.log_prob(grid_mesh_height) + distribution.log_prob(grid_mesh_width)).exp()
    kernel = torch.where(kernel >= kernel.mean(dim=[1,2], keepdim=True), torch.ones_like(kernel), torch.zeros_like(kernel))    
    assert not torch.isnan(kernel).any()

    return kernel * (1+v), kernel

def multivariate_gaussian_attention(v, pos, cov):
    assert (v <= 0).all()
    # V has shape: [batch_size, height, width]
    # Axis X and Y are flipped.
    assert len(v.shape) == 3, v.shape
    batch_size, height, width = v.shape
    assert list(cov.shape) == [v.shape[0], 2, 2], (cov.shape, v.shape)

    # Initialize the kernel and prepare the mesh grids
    pos_x = pos[:, 1]
    pos_y = pos[:, 0]
    kernel = torch.ones_like(v)
    grid_mesh_height = torch.arange(height, device=v.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    grid_mesh_width = torch.arange(width, device=v.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)

    # Shift the mesh grids
    grid_mesh_height = grid_mesh_height - pos_x.unsqueeze(1)
    grid_mesh_width = grid_mesh_width - pos_y.unsqueeze(1)
    
    # Expand them to the other dimension
    grid_mesh_height = grid_mesh_height.unsqueeze(2).repeat(1, 1, width)
    grid_mesh_width = grid_mesh_width.unsqueeze(1).repeat(1, height, 1)
    
    grid_mesh_height_sq = grid_mesh_height.view(-1, 1)
    grid_mesh_width_sq = grid_mesh_width.view(-1, 1)
    grid_mesh_sq = torch.cat([grid_mesh_height_sq, grid_mesh_width_sq], dim=1)
    
    # Expand cov matrix as well.
    cov_expanded = cov.unsqueeze(1).repeat(1, width*height, 1, 1).view(-1, 2, 2)
    zero_mean = torch.zeros_like(grid_mesh_sq)
    
    distribution = torch.distributions.MultivariateNormal(loc=zero_mean, covariance_matrix=cov_expanded)
      
    kernel = distribution.log_prob(grid_mesh_sq).exp()
    kernel = kernel.view(*v.shape)
    kernel = torch.where(kernel >= kernel.mean(dim=[1,2], keepdim=True), torch.ones_like(kernel), torch.zeros_like(kernel))
    assert not torch.isnan(kernel).any()
    # assert torch.allclose(kernel.sum(), torch.tensor(kernel.shape[0], device=kernel.device, dtype=kernel.dtype))

    return kernel * (1+v), kernel

def softargmax(probs_grid, x_coords, y_coords):
    height, width = probs_grid.shape[-2:]
    # height_range = torch.arange(0, height, dtype=torch.float32, device=probs_grid.device)
    # width_range = torch.arange(0, width, dtype=torch.float32, device=probs_grid.device)
    batch_size = probs_grid.shape[0]
    y_pos = (probs_grid * x_coords.float().unsqueeze(2).expand(-1, -1, width)).sum(dim=[1,2])
    x_pos = (probs_grid * y_coords.float().unsqueeze(1).expand(batch_size, height, -1)).sum(dim=[1,2])
    
    return torch.stack([x_pos, y_pos], dim=-1)

def argmax(value_grid, x_coords, y_coords):
    assert len(value_grid.shape) == 3, value_grid.shape
    assert len(x_coords.shape) == 2 and len(y_coords.shape) == 2, (x_coords.shape, y_coords.shape)

    assert value_grid.shape[0] == x_coords.shape[0] and value_grid.shape[0] == y_coords.shape[0], (value_grid.shape, x_coords.shape, y_coords.shape)
    height, width = value_grid.shape[-2:]
    # height_range = torch.arange(0, height, dtype=torch.float32, device=probs_grid.device)
    # width_range = torch.arange(0, width, dtype=torch.float32, device=probs_grid.device)
    batch_size = value_grid.shape[0]
    max_val, max_idx = value_grid.view(batch_size, -1).max(1)
    max_idx = max_idx.view(-1, 1)
    
    x_pos = (max_idx % value_grid.shape[-1]) + y_coords[:, 0].unsqueeze(1)
    y_pos = (max_idx // value_grid.shape[-1]) + x_coords[:, 0].unsqueeze(1)
    
    return torch.cat([x_pos, y_pos], dim=-1).to(dtype=torch.float32)

def render_image_for_video(env, FLAGS, agent, state):
    def to_torch(value):
        return torch.tensor(value, dtype=torch.float32, device=agent.device)
    real_image = env.crop_raw(env.render(mode='rgb_array'))
    if FLAGS.vpn and FLAGS.vpn_masking and FLAGS.sigma_overlay:
        vpn = agent.layers[-1].critic.vpn
        video_current_image = to_torch(env.take_snapshot())
        video_current_goal_image = torch.stack([video_current_image, env.pos_image(agent.layers[-1].goal, video_current_image)], dim=0)
        video_pos_image = env.pos_image(state[:2], video_current_image).unsqueeze(0)
        video_image_position = torch.stack(env.get_image_position(state[:2], video_current_image), dim=-1).unsqueeze(0)

        real_image_torch = to_torch(real_image[...,0].copy())
        env_featurize_setup = env.featurize_image
        env.featurize_image = False
        real_image_position = torch.stack(env.get_image_position(state[:2], real_image_torch), dim=-1)
        env.featurize_image = env_featurize_setup
        with torch.no_grad():
            import cv2
            v_map, p = vpn(video_current_goal_image.unsqueeze(0))
            kernel = vpn.mask_image(v_map, p, video_pos_image, video_image_position, return_kernel=True)
            kernel = torch.where(kernel < 0.01, torch.ones_like(kernel)*0.8, kernel)
            overlay = kernel.squeeze().cpu().numpy()
            scaled_overlay = cv2.resize(overlay, dsize=(real_image.shape[0], real_image.shape[1]))
        real_image = (real_image.astype(np.float32) * scaled_overlay[...,None]).astype(np.uint8)
    return real_image

def project_state(env, FLAGS, layer, state):
    if layer == FLAGS.layers - 1 or (FLAGS.oracle and layer == FLAGS.layers - 2):
        return env.project_state_to_end_goal(None, state)
    else:
        return env.project_state_to_subgoal(None, state)

def layer_goal_nn(input_layer, num_next_neurons, is_output=False):
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]
    
    
    fan_in_init = 1 / num_prev_neurons ** 0.5
    weight_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init)
    bias_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init) 

    weights = tf.get_variable("weights", shape, initializer=weight_init)
    biases = tf.get_variable("biases", [num_next_neurons], initializer=bias_init)

    dot = tf.matmul(input_layer, weights) + biases

    if is_output:
        return dot

    relu = tf.nn.relu(dot)
    return relu


# Below function prints out options and environment specified by user
def print_summary(FLAGS,env):

    print("\n- - - - - - - - - - -")
    print("Task Summary: ","\n")
    print("Environment: ", env.name)
    print("Number of Layers: ", FLAGS.layers)
    print("Time Limit per Layer: ", FLAGS.time_scale)
    print("Max Episode Time Steps: ", env.max_actions)
    print("Retrain: ", FLAGS.retrain)
    print("Test: ", FLAGS.test)
    print("Visualize: ", FLAGS.show)
    print("- - - - - - - - - - -", "\n\n")


# Below function ensures environment configurations were properly entered
def check_validity(model_name, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space, subgoal_bounds, subgoal_thresholds, max_actions, timesteps_per_action):

    # Ensure model file is an ".xml" file
    assert model_name[-4:] == ".xml", "Mujoco model must be an \".xml\" file"

    # Ensure upper bounds of range is >= lower bound of range
    if goal_space_train is not None:
        for i in range(len(goal_space_train)):
            assert goal_space_train[i][1] >= goal_space_train[i][0], "In the training goal space, upper bound must be >= lower bound"

    if goal_space_test is not None:
        for i in range(len(goal_space_test)):
            assert goal_space_test[i][1] >= goal_space_test[i][0], "In the training goal space, upper bound must be >= lower bound"

    for i in range(len(initial_state_space)):
        assert initial_state_space[i][1] >= initial_state_space[i][0], "In initial state space, upper bound must be >= lower bound"
    
    for i in range(len(subgoal_bounds)):
        assert subgoal_bounds[i][1] >= subgoal_bounds[i][0], "In subgoal space, upper bound must be >= lower bound" 

    # Make sure end goal spaces and thresholds have same first dimension
    if goal_space_train is not None and goal_space_test is not None:
        assert len(goal_space_train) == len(goal_space_test) == len(end_goal_thresholds), "End goal space and thresholds must have same first dimension"

    # Makde sure suboal spaces and thresholds have same dimensions
    assert len(subgoal_bounds) == len(subgoal_thresholds), "Subgoal space and thresholds must have same first dimension"

    # Ensure max action and timesteps_per_action are postive integers
    assert max_actions > 0, "Max actions should be a positive integer"

    assert timesteps_per_action > 0, "Timesteps per action should be a positive integer"


def oracle_action(FLAGS, current_state, env_goal, env):
    def distance_subgoal_propotional(pos, goal):
        distances = np.sqrt(np.sum((pos - goal) * (pos - goal), axis=0)+1e-10)
        # https://www.desmos.com/calculator/mb3rlpkbsp
        return pos + np.exp(-1/20*(distances)) * (goal-pos)

    def distance_subgoal_constant(pos, goal):
        distances = np.sqrt(np.sum((pos - goal) * (pos - goal), axis=0)+1e-10)
        if distances > 4:
            result = pos + ((goal-pos) / distances) * 4
            return pos + ((goal-pos) / distances) * 4
        else:
            return goal

    distance_subgoal = distance_subgoal_constant if FLAGS.new_oracle else distance_subgoal_propotional
    pos = env.project_state_to_end_goal(env.sim, current_state)
    goal = env_goal

    if env.name in ["ant_reacher.xml", "AntEmptyNegDistDict"]:
        if FLAGS.relative_subgoals:
            return distance_subgoal(pos, goal) - pos
        else:
            return distance_subgoal(pos, goal)
    elif env.name == "AntKeyGateDict" or env.name == "PointKeyGateDict" or env.name == "AntKeyGatePassageDict":
        torso_x, torso_y = env._env._init_torso_x, env._env._init_torso_y
        struct, scale = env._env.MAZE_STRUCTURE, env._env.MAZE_SIZE_SCALING
        gate = env._env.gate
        def coords_to_cell(coords):
            return round((coords[0] + torso_x) / scale), round((coords[1] + torso_y) / scale)
        def cell_to_coords(cell):
            return cell[0] * scale - torso_x, cell[1] * scale - torso_y
        def keygate_subgoal(pos, goal):
            pos_cell, goal_cell = coords_to_cell(pos[:2]), coords_to_cell(goal[:2])
            key_cell, gate_cell = coords_to_cell(gate['key_location'][:2]), gate['gate_cell_coords']
            # print("Pos:", pos_cell, pos[:2])
            # print("Goal:", goal_cell, goal[:2])
            # print("Key:", key_cell, gate['key_location'][:2])
            # print("Gate:", gate_cell)
            # If the spider is already in the correct cell
            if np.allclose(pos_cell, goal_cell):
                return distance_subgoal(pos, goal)

            # If the spider can reach the goal using by following a straight line.
            if pos_cell[0] > gate_cell[0] and goal_cell[0] > gate_cell[0]:
                return distance_subgoal(pos, goal)
            if pos_cell[0] < gate_cell[0] and goal_cell[0] < gate_cell[0] and not np.allclose(goal_cell, key_cell):
                return distance_subgoal(pos, goal)


            # If the goal is in the gate cell.
            if np.allclose(goal_cell, gate_cell):
                # If the gate is open.
                if not gate['closed']:
                    if np.allclose(pos_cell, (gate_cell[0]-1, gate_cell[1])):
                        return distance_subgoal(pos, goal)
                    else:
                        # Navigate in front of the gate.
                        new_goal = np.copy(goal)
                        new_goal[:2] = cell_to_coords(gate_cell)
                        new_goal[0] -= scale
                        return distance_subgoal(pos, new_goal)
                else:  # Gate is closed.
                    new_goal = np.copy(goal)
                    new_goal[:2] = gate['key_location']
                    return keygate_subgoal(pos, new_goal)  # Navigate to the key first.
        
            # If the goal is in the key cell and the spider is cell next to the key.
            if np.allclose(key_cell, goal_cell):
                if np.allclose(pos_cell, (goal_cell[0] + 1, goal_cell[1])):
                    return distance_subgoal(pos, goal)
                # If the goal is in the key cell and the spider is far.
                else:
                    new_goal = np.copy(goal)
                    new_goal[0] += scale
                    return distance_subgoal(pos, new_goal)
            
            # If the goal in the upper right part.
            if goal_cell[0] >= gate_cell[0] and goal_cell[1] > gate_cell[1]:
                # Take the long way upwards.
                passage_cell = (gate_cell[0], gate_cell[1] + 3)
                # If in or in front of the passage
                if np.allclose(pos_cell, passage_cell) or np.allclose(pos_cell, (passage_cell[0]-1, passage_cell[1])):
                    new_goal = np.copy(goal)
                    new_goal[:2] = cell_to_coords((passage_cell[0]+1, passage_cell[1]))
                    return distance_subgoal(pos, new_goal)
                elif pos_cell[0] < passage_cell[0]:  # Otherwise, navigate in front of the passage.
                    new_goal = np.copy(goal)
                    new_goal[:2] = cell_to_coords((passage_cell[0]-1, passage_cell[1]))
                    return distance_subgoal(pos, new_goal)

            # If the goal in the lower right part.
            if goal_cell[0] >= gate_cell[0] and goal_cell[1] <= gate_cell[1]:
                # Take the path through the gate.
                if gate['closed']:
                    new_goal = np.copy(goal)
                    new_goal[:2] = gate['key_location']
                    return keygate_subgoal(pos, new_goal)
                else:
                    # If in or in front of the gate
                    if np.allclose(pos_cell, gate_cell) or np.allclose(pos_cell, (gate_cell[0]-1, gate_cell[1])):
                        new_goal = np.copy(goal)
                        new_goal[:2] = cell_to_coords((gate_cell[0]+1, gate_cell[1]))
                        return distance_subgoal(pos, new_goal)
                    elif pos_cell[0] < gate_cell[0]:  # Otherwise, navigate in front of the gate.
                        new_goal = np.copy(goal)
                        new_goal[:2] = cell_to_coords((gate_cell[0]-1, gate_cell[1]))
                        return distance_subgoal(pos, new_goal)
            
            # If the spider has passed the wall.
            return distance_subgoal(pos, goal)

        if FLAGS.relative_subgoals:
            return keygate_subgoal(pos, goal) - pos
        else:
            return keygate_subgoal(pos, goal)

    elif env.name == "AntKeyGateBackwardsDict":
        torso_x, torso_y = env._env._init_torso_x, env._env._init_torso_y
        struct, scale = env._env.MAZE_STRUCTURE, env._env.MAZE_SIZE_SCALING
        gate = env._env.gate
        def coords_to_cell(coords):
            return round((coords[0] + torso_x) / scale), round((coords[1] + torso_y) / scale)
        def cell_to_coords(cell):
            return cell[0] * scale - torso_x, cell[1] * scale - torso_y
        def keygate_subgoal(pos, goal):
            pos_cell, goal_cell = coords_to_cell(pos[:2]), coords_to_cell(goal[:2])
            key_cell, gate_cell = coords_to_cell(gate['key_location'][:2]), gate['gate_cell_coords']
            # print("Pos:", pos_cell, pos[:2])
            # print("Goal:", goal_cell, goal[:2])
            # print("Key:", key_cell, gate['key_location'][:2])
            # print("Gate:", gate_cell)
            # If the spider is already in the correct cell
            if np.allclose(pos_cell, goal_cell):
                return distance_subgoal(pos, goal)

            # If the spider can reach the goal using by following a straight line.
            if pos_cell[0] > gate_cell[0] and goal_cell[0] > gate_cell[0]:
                return distance_subgoal(pos, goal)
            if pos_cell[0] < gate_cell[0] and goal_cell[0] < gate_cell[0] and not np.allclose(goal_cell, key_cell):
                return distance_subgoal(pos, goal)


            # If the goal is in the gate cell.
            if np.allclose(goal_cell, gate_cell):
                # If the gate is open.
                if not gate['closed']:
                    if np.allclose(pos_cell, (gate_cell[0]-1, gate_cell[1])):
                        return distance_subgoal(pos, goal)
                    else:
                        # Navigate in front of the gate.
                        new_goal = np.copy(goal)
                        new_goal[:2] = cell_to_coords(gate_cell)
                        new_goal[0] -= scale
                        return distance_subgoal(pos, new_goal)
                else:  # Gate is closed.
                    new_goal = np.copy(goal)
                    new_goal[:2] = gate['key_location']
                    return keygate_subgoal(pos, new_goal)  # Navigate to the key first.
        
            # If the goal is in the key cell and the spider is cell next to the key.
            if np.allclose(key_cell, goal_cell):
                if np.allclose(pos_cell, (goal_cell[0] + 1, goal_cell[1])):
                    return distance_subgoal(pos, goal)
                # If the goal is in the key cell and the spider is far.
                else:
                    new_goal = np.copy(goal)
                    new_goal[0] += scale
                    return distance_subgoal(pos, new_goal)
            
            # If the goal is in the left part.
            if goal_cell[0] < gate_cell[0] and goal_cell[1] < gate_cell[1]:
                # Take the long way upwards.
                passage_cell = (gate_cell[0], gate_cell[1] + 3)
                # If in or in front of the passage
                if np.allclose(pos_cell, passage_cell) or np.allclose(pos_cell, (passage_cell[0]+1, passage_cell[1])):
                    new_goal = np.copy(goal)
                    new_goal[:2] = cell_to_coords((passage_cell[0]-1, passage_cell[1]))
                    return distance_subgoal(pos, new_goal)
                elif pos_cell[0] > passage_cell[0]:  # Otherwise, navigate in front of the passage.
                    new_goal = np.copy(goal)
                    new_goal[:2] = cell_to_coords((passage_cell[0]+1, passage_cell[1]))
                    return distance_subgoal(pos, new_goal)

            # If the goal in the lower right part.
            if goal_cell[0] >= gate_cell[0] and goal_cell[1] <= gate_cell[1]:
                # Take the path through the gate.
                if gate['closed']:
                    new_goal = np.copy(goal)
                    new_goal[:2] = gate['key_location']
                    return keygate_subgoal(pos, new_goal)
                else:
                    # If in or in front of the gate
                    if np.allclose(pos_cell, gate_cell) or np.allclose(pos_cell, (gate_cell[0]-1, gate_cell[1])):
                        new_goal = np.copy(goal)
                        new_goal[:2] = cell_to_coords((gate_cell[0]+1, gate_cell[1]))
                        return distance_subgoal(pos, new_goal)
                    elif pos_cell[0] < gate_cell[0]:  # Otherwise, navigate in front of the gate.
                        new_goal = np.copy(goal)
                        new_goal[:2] = cell_to_coords((gate_cell[0]-1, gate_cell[1]))
                        return distance_subgoal(pos, new_goal)
            
            # If the spider has passed the wall.
            return distance_subgoal(pos, goal)

        if FLAGS.relative_subgoals:
            return keygate_subgoal(pos, goal) - pos
        else:
            return keygate_subgoal(pos, goal)
    else:
        raise ValueError(env.name)


def save_video(video_frames, filename, fps=30):
    assert fps == int(fps), fps
    import skvideo.io
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    skvideo.io.vwrite(filename, video_frames, inputdict={'-r': str(int(fps))})