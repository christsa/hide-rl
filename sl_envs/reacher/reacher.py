import numpy as np
import gym
from gym import spaces
from gym import utils
from sl_envs.reacher import mujoco_env
import torch

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.goal = np.array([0.0, 0.0])

        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

        self.n_bins = 24
        self.MAZE_STRUCTURE = [[0, 0, 0, 0, 0, 0],
        [0, '+', 0, 0, 0, 0],
        [0, 0, 0, 'r', 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        ]
        self.MAZE_SIZE_SCALING = 4 #4*np.pi/self.n_bins
        self.max_torque = 1.
        self.max_speed = 10.
        high = np.array([1., 1., self.max_speed, self.max_speed])

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(-high, high, dtype=np.float32),
            'achieved_goal': gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            'desired_goal': gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        })
        self.subgoal_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def _set_goal(self, goal):
        self.goal = goal

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([self.sim.data.qpos.flat[:2],self.sim.data.qvel.flat[:2]])
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2].copy(),
            'desired_goal': self.goal.copy(),
        }


    def get_image_position(self, pos, image):
        pos2d = pos[..., :2] / self.MAZE_SIZE_SCALING
        return pos2d[..., -2], pos2d[..., -1]

    def get_env_position(self, pos, image):
        pos2dx = (pos[..., -2]) * self.MAZE_SIZE_SCALING
        pos2dy = (pos[..., -1]) * self.MAZE_SIZE_SCALING
        return pos2dx, pos2dy

    def take_snapshot(self):
        len_x = len(self.MAZE_STRUCTURE[0] * self.MAZE_SIZE_SCALING)
        len_y = len(self.MAZE_STRUCTURE * self.MAZE_SIZE_SCALING)
        image = np.zeros((len_y, len_x), dtype=np.float32)
        for row in range(len_x):
            for col in range(len_y):
                symbol = self.MAZE_STRUCTURE[int(col / self.MAZE_SIZE_SCALING)][
                    int(row / self.MAZE_SIZE_SCALING)]
                if symbol in [1, 'G']:
                    image[len_y - col - 1, row] = 0.
                else:
                    image[len_y - col - 1, row] = 1.
        return image

