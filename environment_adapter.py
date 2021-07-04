# from tkinter import *
# from tkinter import ttk
import time
import numpy as np
from sl_envs.hac.hac_env import HACEnv
# from sl_envs.reacher.reacher import ReacherEnv
from sl_envs.lj.create_maze_env import create_maze_env
# from sl_envs.fetch import create_maze as create_fetch_maze
from mujoco_py import load_model_from_path, MjSim, MjViewer
from map import generate_random_map
import pickle

class EnvironmentAdapter():

    def __init__(self, universe, domain, task, project_state_to_end_goal, project_state_to_subgoal, show = False, normalization_dict=None, featurize_image=False, seed=0):
        self.task = task
        self.universe = universe
        self.domain = domain
        self.featurize_image = featurize_image

        if universe == "HAC":
            assert domain == "Ant"
            self._env = env = HACEnv(task)
        elif universe == "reacher":
            #assert domain == "reacher"
            self._env = env = ReacherEnv()
        elif universe == "maze":
            assert domain in ["Ant", "Point", "Humanoid", "DMPoint"]
            if task == 'KeyGateHACDict':
                task = 'KeyGateDict'
                self._env = env = create_maze_env(domain+task, seed=seed, hac_mode=True, extra_dims=True)
            elif task in ['CustomDict']:
                self.maze_counter = 0
                self.mazes = pickle.load(open('RandomMazes.pkl', 'rb'))
                self.task = task = 'CustomDict'
                self._env = env = self.create_new_env()
            else:
                self._env = env = create_maze_env(domain+task, seed=seed, hac_mode=True)
            print("Subgoal space: ", env.subgoal_space.low, env.subgoal_space.high)
            self.env_len_x = (self._env.MAZE_SIZE_SCALING * len(self._env.MAZE_STRUCTURE[0]))
            self.env_len_y = (self._env.MAZE_SIZE_SCALING * len(self._env.MAZE_STRUCTURE))
        elif universe == "fetch":
            import gym
            assert domain in ["push", "ball"]
            if "random" in task:
                self.maze_counter = 0
                self.mazes = pickle.load(open('RandomMazesNoFence_08.pkl', 'rb'))
                self.task = task = 'CustomDict'
                self._env = env = self.create_new_env()
            else:
                self._env = env = create_fetch_maze(domain, task)


        if universe == "HAC":
            self.name = env._env.name
        else:
            self.name = domain+task

        self.normalization_dict = normalization_dict
        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = env.observation_space.spaces['observation'].shape[0]
        self.action_dim = env.action_space.shape[0] # low-level action dim
        self.action_bounds = env.action_space.high # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.end_goal_dim = env.observation_space.spaces['desired_goal'].shape[0]
        self.subgoal_dim = env.subgoal_space.shape[0]
        self.subgoal_bounds = list(zip(self._normalize_subgoal_list(env.subgoal_space.low), self._normalize_subgoal_list(env.subgoal_space.high)))

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal 
        # self.project_state_to_subgoal = lambda sim, state: np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))
        # self.project_state_to_subgoal = lambda sim, state: np.concatenate((state[:2], np.array([1 if state[2] > 1 else state[2]]), np.array([3 if state[i] > 3 else -3 if state[i] < -3 else state[i] for i in range(15, 17)])))
        self.project_state_to_subgoal = project_state_to_subgoal

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]


        # End goal/subgoal thresholds
        if self.universe == 'fetch':
            self.end_goal_thresholds = self._normalize_end_goal_list([0.035] * self.end_goal_dim)
            self.subgoal_thresholds = self._normalize_subgoal_list([0.035] * self.subgoal_dim)
        elif self.universe == 'reacher':
            self.end_goal_thresholds = self._normalize_end_goal_list([0.05] * self.end_goal_dim)
            self.subgoal_thresholds = self._normalize_subgoal_list([0.05] * self.subgoal_dim)
        else:
            self.end_goal_thresholds = self._normalize_end_goal_list([1.0] * self.end_goal_dim)
            self.subgoal_thresholds = self._normalize_subgoal_list([0.5] * self.subgoal_dim)

        if ("KeyGate" in task or "Wall" in task):
            self.max_actions = 1000
        elif self.universe == 'fetch':
            self.max_actions = 500
        else:
            self.max_actions = 500

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean

        self._current_obs = None
        self._subgoals = None
        self._img_size = None

    def create_new_env(self):
        # self.map = generate_random_map(size=4)
        self.map = self.mazes['mazes'][self.maze_counter]
        print("Maze distance: ", self.mazes['distances'][self.maze_counter])
        self.maze_counter += 1
        if self.universe == 'fetch':
            self._env = env = create_fetch_maze(self.domain, self.task, maze_structure=self.map)
        else:
            self._env = env = create_maze_env(self.domain+self.task, hac_mode=True, maze_structure=self.map)
        self._current_obs = None
        self._subgoals = None
        return self._env

    def get_obstacle_list(self):
        obstacles = []
        #self.map = self.map[::-1]
        start = []
        goal = []
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == 1:
                    obstacles.append((j*self._env.MAZE_SIZE_SCALING,i*self._env.MAZE_SIZE_SCALING,self._env.MAZE_SIZE_SCALING))
                elif self.map[i][j] == 'r':
                    start = [j*self._env.MAZE_SIZE_SCALING+self._env.MAZE_SIZE_SCALING/2,i*self._env.MAZE_SIZE_SCALING+self._env.MAZE_SIZE_SCALING/2]
                elif self.map[i][j] == '+':
                    goal = [j*self._env.MAZE_SIZE_SCALING+self._env.MAZE_SIZE_SCALING/2,i*self._env.MAZE_SIZE_SCALING+self._env.MAZE_SIZE_SCALING/2]
        return obstacles, start, goal
    # Get state, which concatenates joint positions and velocities
    def get_state(self):

        return self._normalize_observation(self._current_obs['observation'])

    @property
    def image_size(self):
        if self._img_size is None:
            assert self._current_obs is None
            self.reset_sim(None)
            self._img_size = self.take_snapshot().shape
        return self._img_size

    @property
    def healthy(self):
        if self.universe == 'fetch':
            return self._env.is_pos_valid(self._env._get_obs()['achieved_goal'])
        return True

    # Reset simulation to state within initial state specified by user
    def reset_sim(self, test):

        self._current_obs = self._env.reset()
        # Return state
        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def execute_action(self, action):

        self._current_obs, reward, done, info = self._env.step(action)
        if self.visualize:
            self._env.render(subgoals=self._subgoals, mode='human')

        return self.get_state()

    def render(self, subgoals=None, mode='rgb_array'):
        return self._env.render(mode=mode, subgoals=self._subgoals, width=1000, height=1000)

    @property
    def sim(self):
        return None

    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self,end_goal):
        pass


    # Function returns an end goal
    def get_next_goal(self,test):
        # self.reset_sim()
        end_goal = self._normalize_end_goal_list(self._current_obs['desired_goal'])

        return end_goal

    def crop_raw(self, image):
        top_x, bottom_x = 0, 0
        for i, row in enumerate(image):
            if np.mean(row) < 230:
                top_x = i
                break
        
        for i in range(len(image)-1, -1, -1):
            row = image[i]
            if np.mean(row) < 230:
                bottom_x = i
                break
        
        # Crop top/botom
        image = image[:bottom_x+1]
        image = image[top_x:]

        left_y, right_y = 0, 0
        for i in range(len(image[0])):
            row = image[:, i]
            if np.mean(row) < 230:
                left_y = i
                break
        
        for i in range(len(image[0])-1, -1, -1):
            row = image[:, i]
            if np.mean(row) < 230:
                right_y = i
                break

        # Crop right/left
        image = image[:, :right_y+1]
        image = image[:, left_y:]
        return image

    def take_snapshot(self):
        if self.universe == 'fetch': return self._env.take_snapshot()
        def preprocess_image(image):
            # from scipy import ndimage
            image = self.crop_raw(image)

            # Convert to Grayscale
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            # image = ndimage.gaussian_filter(image, sigma=0.7)
            # Scale between 0 and 1
            image = image / 255
            return image

        if self.featurize_image:
            len_x = len(self._env.MAZE_STRUCTURE[0] * self._env.MAZE_SIZE_SCALING)
            len_y = len(self._env.MAZE_STRUCTURE * self._env.MAZE_SIZE_SCALING)
            image = np.zeros((len_y, len_x), dtype=np.float32)
            for row in range(len_x):
                for col in range(len_y):
                    symbol = self._env.MAZE_STRUCTURE[int(col/self._env.MAZE_SIZE_SCALING)][int(row/self._env.MAZE_SIZE_SCALING)]
                    if symbol in [1, 'G']:
                        image[len_y-col-1, row] = 0.
                    else:
                        image[len_y-col-1, row] = 1.
            return image
        else:
            assert hasattr(self._env, 'snap')
            image = self._env.snap()
            return preprocess_image(image)

    def get_image_position(self, pos, image):
        if self.universe == "fetch": return self._env.get_image_position(pos, image)
        len_x = (self._env.MAZE_SIZE_SCALING * len(self._env.MAZE_STRUCTURE[0]))
        len_y = (self._env.MAZE_SIZE_SCALING * len(self._env.MAZE_STRUCTURE))
        # x_scale = image.shape[-1] / len_x
        # y_scale = image.shape[-2] / len_y
        im_pos_x = (pos[..., 0] + len_x/2) * image.shape[-1] / len_x
        im_pos_y = (pos[..., 1] + len_y/2) * image.shape[-2] / len_y
        # y coord is flipped
        im_pos_y = image.shape[-2] - im_pos_y
        assert (im_pos_x >= 0).all(), (pos[..., 0].min(), im_pos_x.min())
        assert (im_pos_y >= 0).all(), (pos[..., 1].max(), im_pos_y.min())
        assert (im_pos_x < image.shape[-1]).all(), (pos[..., 0].max(), im_pos_x.max())
        assert (im_pos_y < image.shape[-2]).all(), (pos[..., 1].min(), im_pos_y.max())
        return im_pos_x, im_pos_y

    def get_env_position(self, pos, image):
        if self.universe == "fetch": return self._env.get_env_position(pos, image)
        len_x = (self._env.MAZE_SIZE_SCALING * len(self._env.MAZE_STRUCTURE[0]))
        len_y = (self._env.MAZE_SIZE_SCALING * len(self._env.MAZE_STRUCTURE))

        # x_scale: len_x / image.shape[-1]
        # y_scale: len_y / image.shape[-2]
        env_pos_x = (pos[..., 0] * len_x / image.shape[-1]) - len_x/2
        env_pos_y = ((image.shape[-2] - pos[..., 1]) * len_y / image.shape[-2]) - len_y/2
        # assert (env_pos_x >= self.subgoal_bounds[0][0]).all(), env_pos_x.min()
        # assert (env_pos_x <= self.subgoal_bounds[0][1]).all(), env_pos_x.max()
        # assert (env_pos_y >= self.subgoal_bounds[1][0]).all(), env_pos_y.min()
        # assert (env_pos_y <= self.subgoal_bounds[1][1]).all(), env_pos_y.max()
        return env_pos_x, env_pos_y

    def pos_image(self, goal, image, offset=0, color=1.0):
        if self.universe == "fetch": return self._env.pos_image(goal, image, color)
        assert len(image.shape) in [2, 3], image.shape
        # Make a copy of the image.
        if hasattr(image, 'clone'):
            import torch
            result = torch.zeros_like(image)
        else:
            import numpy as np
            result = np.zeros_like(image)
        if self.featurize_image:
            center_x = (len(self._env.MAZE_STRUCTURE[0]) * self._env.MAZE_SIZE_SCALING)/2
            center_y = (len(self._env.MAZE_STRUCTURE) * self._env.MAZE_SIZE_SCALING)/2
            goal_x = (goal[..., 0] + center_x).long()
            goal_y = (goal[..., 1] + center_y).long()
            goal_y = image.shape[-2]-goal_y-1
            if len(result.shape)== 3:
                result[torch.arange(result.shape[0], device=result.device), goal_y, goal_x] = color
            else:
                result[goal_y, goal_x] = color
        else:
            if offset is None:
                offset = int((image.shape[-2] + image.shape[-1]) /200 * 3)
            x, y = self.get_image_position(goal, image)
            x, y = x.long(), y.long()
            if len(result.shape) == 3:
                result[torch.arange(result.shape[0], device=image.device), y, x] = color
            else:
                result[(y-offset):(y+offset+1), (x-offset):(x+offset+1)] = color
        return result

    # Visualize all subgoals
    def display_subgoals(self, subgoals, FLAGS):
        self._subgoals = {}
        for i, subgoal in enumerate(subgoals):
            # if i == FLAGS.layers -1:  # This is the env goal.
            #     continue
            if (i == FLAGS.layers -2 and FLAGS.oracle):
                if FLAGS.relative_subgoals:
                    pos = self.project_state_to_end_goal(None, self.get_state())
                    self._subgoals[i] = self._denormalize_end_goal_list(subgoal + pos)
                else:
                    self._subgoals[i] = self._denormalize_end_goal_list(subgoal)
            else:
                if FLAGS.relative_subgoals:
                    pos = self.project_state_to_subgoal(None, self.get_state())
                    self._subgoals[i] = self._denormalize_subgoal_list(subgoal + pos)
                else:
                    self._subgoals[i] = self._denormalize_subgoal_list(subgoal)
    
    def _unscale(self, number, low, high):
        assert (low is None) == (high is None)
        if low is None:
            return number
        return np.clip(2*(number - low)/(high - low) - 1., -1., 1.)

    def _scale(self, number, low, high):
        assert (low is None) == (high is None)
        if low is None:
            return number
        return np.clip(low + (number + 1.) * (high - low) / 2.0, low, high)

    def _unscale_list(self, lst, indices, dims):
        new_lst = lst.copy()
        for i, dim in zip(indices, dims):
            low = self.normalization_dict['lows'][dim]
            high = self.normalization_dict['highs'][dim]
            new_lst[i] = self._unscale(lst[i], low, high)
        return new_lst

    def _scale_list(self, lst, indices, dims):
        new_lst = lst.copy()
        for i, dim in zip(indices, dims):
            low = self.normalization_dict['lows'][dim]
            high = self.normalization_dict['highs'][dim]
            new_lst[i] = self._scale(lst[i], low, high)
        return new_lst

    def _normalize_end_goal_list(self, end_goal_list):
        if self.normalization_dict is None:
            return end_goal_list
        dims = self.normalization_dict['end_goal_dims']
        return self._unscale_list(end_goal_list, range(len(dims)), dims)
        
    def _normalize_subgoal_list(self, subgoal_list):
        if self.normalization_dict is None:
            return subgoal_list
        dims = self.normalization_dict['subgoal_dims']
        return self._unscale_list(subgoal_list, range(len(dims)), dims)

    def _normalize_observation(self, observation):
        if self.normalization_dict is None:
            return observation
        dims = list(self.normalization_dict['lows'].keys())
        return self._unscale_list(observation, dims, dims)
    
    def _denormalize_subgoal_list(self, subgoal_list):
        if self.normalization_dict is None:
            return subgoal_list
        dims = self.normalization_dict['subgoal_dims']        
        return self._scale_list(subgoal_list, range(len(dims)), dims)

    def _denormalize_end_goal_list(self, end_goal_list):
        if self.normalization_dict is None:
            return end_goal_list
        dims = self.normalization_dict['end_goal_dims']        
        return self._scale_list(end_goal_list, range(len(dims)), dims)
        
        
