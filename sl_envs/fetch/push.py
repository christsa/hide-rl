from . import fetch_env
import gym.envs.robotics.utils as robo_utils
import gym.envs.robotics.rotations as rotations
import numpy as np
import gym
import os
import tempfile
import xml.etree.ElementTree as ET

class PushEnv(fetch_env.FetchEnv, gym.utils.EzPickle):
    def __init__(self, maze_structure):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'robot0:shoulder_pan_joint': -0.05,
            'robot0:shoulder_lift_joint': -0.95,
            'robot0:upperarm_roll_joint': 0.0,
            'robot0:elbow_flex_joint': 1.8,
            'robot0:forearm_roll_joint': 0.0,
            'robot0:wrist_flex_joint': 1.0,
            'robot0:wrist_roll_joint': 0.0,
            'object0:joint': [1.25, 0.53, 0.43, 1., 0., 0., 0.],
        }
        self.images = []
        self.object_init_pos = None
        self.center = np.array([1.09944425, 0.7441,     0.42544991])
        self.MAZE_SIZE_SCALING = 0.025 * 2
        model_file_handle, model_file_path = self.create_maze(self.center, maze_structure, os.path.join(os.path.dirname(__file__), "assets", "push.xml"))
        fetch_env.FetchEnv.__init__(
            self, model_file_path, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type="sparse")
        os.close(model_file_handle)
        print(model_file_path)
        os.remove(model_file_path)
        
        gym.utils.EzPickle.__init__(self)
        # super(PushEnv, self).__init__(*args, **kwards, model_path=model_path)
        # self.observation_space = self.new_observation_space()
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(-np.inf, np.inf, shape=(len(self._get_obs()['observation']), ), dtype=np.float32), 
            'achieved_goal': gym.spaces.Box(-0.2, 0.2, shape=(2, ), dtype=np.float32), 
            'desired_goal': gym.spaces.Box(-0.2, 0.2, shape=(2, ), dtype=np.float32), 
        })
        self.subgoal_space = gym.spaces.Box(-0.2, 0.2, shape=(2, ), dtype=np.float32)

    def step(self, *args, **kwargs):
        res = super().step(*args, **kwargs)
        # self.images.append(self.render('rgb_array', subgoals=None))
        return res

    def create_maze(self, center_coords, MAZE_STRUCTURE, model_path):
        self.MAZE_STRUCTURE = MAZE_STRUCTURE
        tree = ET.parse(model_path)
        assets_dir = os.path.join(gym.__path__[0], "envs", "robotics", "assets")
        print(assets_dir)
        compiler = tree.find(".//compiler")
        compiler.attrib['meshdir'] = os.path.join(assets_dir, compiler.attrib['meshdir'])
        compiler.attrib['texturedir'] = os.path.join(assets_dir, compiler.attrib['texturedir'])
        worldbody = tree.find(".//worldbody")
        self.MAZE_STRUCTURE = MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        torso_x = -center_coords[0]
        torso_y = -center_coords[1]
        height_offset = 0.4
        self.target_coords = None
        self.object_coords = None
        i_center = 4
        j_center = 4
        for i in range(len(MAZE_STRUCTURE)):
            for j in range(len(MAZE_STRUCTURE[0])):
                struct = MAZE_STRUCTURE[len(MAZE_STRUCTURE) - i - 1][j]
                if struct in [1, 2]:  # Unmovable block.
                    if struct == 2: 
                        print ("....", (j-j_center) * size_scaling - torso_x,
                                          (i - i_center) * size_scaling - torso_y,
                                          height_offset +
                                          0)
                    # Offset all coordinates so that robot starts at the origin ( in the middle).
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i-i_center, j-j_center),
                        pos="%f %f %f" % ((j-j_center) * size_scaling - torso_x,
                                          (i - i_center) * size_scaling - torso_y,
                                          height_offset +
                                          0),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           0.03),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1",
                    )
                elif struct == "+":
                    self.target_coords = (i-i_center , j-j_center)
                elif struct == "O":
                    self.object_coords = (i-i_center , j-j_center)
        file_handle, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)
        return file_handle, file_path

    def compute_reward(self, achieved_goal, goal, info):
        return None

    def is_pos_valid(self, pos):
        if (pos < -0.25).any() or (pos > 0.25).any():
            print("Returning invalid position: {pos}")
            return False
        else:
            return True

    def get_image_position(self, pos, image):
        pos2d = pos[..., :2] / self.MAZE_SIZE_SCALING + 4.5
        return pos2d[..., -2], 9-pos2d[..., -1]

    def get_env_position(self, pos, image):
        assert image.shape[-1] == 9
        assert image.shape[-2] == 9
        pos2dx = (pos[..., -2] - 4.5) * self.MAZE_SIZE_SCALING
        pos2dy = (4.5 - pos[..., -1]) * self.MAZE_SIZE_SCALING
        return pos2dx, pos2dy

    def take_snapshot(self):
        np_arr = np.zeros((len(self.MAZE_STRUCTURE), len(self.MAZE_STRUCTURE[0])), dtype=np.float32)
        for i in range(len(self.MAZE_STRUCTURE)):
            for j in range(len(self.MAZE_STRUCTURE[0])):
                if self.MAZE_STRUCTURE[i][j] in [1, 2]:
                    np_arr[i][j] = 0
                else:
                    np_arr[i][j] = 1
        return np_arr

    def pos_image(self, pos, image, color):
        import torch
        result = torch.zeros_like(image)
        image_pos_x, image_pos_y = self.get_image_position(pos, image)

        if len(result.shape)== 3:
            image_pos_x = torch.clamp(image_pos_y, 0, 8)
            image_pos_y= torch.clamp(image_pos_y, 0, 8)
            result[torch.arange(result.shape[0], device=result.device), image_pos_y.long(), image_pos_x.long()] = color
        else:
            if image_pos_x < 0 or image_pos_x >= 9 or image_pos_y < 0 or image_pos_y >= 9:
                print("problematic pos:", pos, "image pos:", image_pos_x, image_pos_y)
                image = self.render('rgb_array', subgoals=None)
                import skimage
                from utils import save_video
                skimage.io.imsave("videos/problem.png", image)
                save_video(self.images, "videos/problem_video.avi")
            assert image_pos_x >= 0, (image_pos_x, pos)
            assert image_pos_x < 9, (image_pos_x, pos)
            assert image_pos_y >= 0, (image_pos_y, pos)
            assert image_pos_y < 9, (image_pos_y, pos)
            result[image_pos_y.long(), image_pos_x.long()] = color
        return result

    def _is_success(self, achieved_goal, desired_goal):
        return False
        # d = goal_distance(achieved_goal[:2], desired_goal[:2])
        # return (d < self.distance_threshold).astype(np.float32)
    
    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:gripper_link')
        # lookat = self.sim.data.body_xpos[body_id]
        lookat = self.center
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 4.0
        # self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -90.


    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robo_utils.robot_get_obs(self.sim)
        assert self.has_object
        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            object_pos.ravel()[:2] - self.center[:2], object_rel_pos, gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal[:2].copy() - self.center[:2],
            'desired_goal': self.goal.copy()[:2] - self.center[:2],
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        object_xpos = self.center[:2]
        if self.object_coords is None:
            center_x, center_y = int(len(self.MAZE_STRUCTURE)/2), int(len(self.MAZE_STRUCTURE[0])/2)
            x, y = None, None
            while x is None or self.MAZE_STRUCTURE[len(self.MAZE_STRUCTURE)-x-1][y] != 0 or ((x-center_x) == int(len(self.MAZE_STRUCTURE)/2) and (y-center_y) == int(len(self.MAZE_STRUCTURE[0])/2)):
                x, y = np.random.randint(2, len(self.MAZE_STRUCTURE)-2), np.random.randint(2, len(self.MAZE_STRUCTURE[0])-2)
                self.object_init_pos = (x-center_x, y-center_y)
                object_xpos = ((y-center_y) * self.MAZE_SIZE_SCALING + self.center[0], (x - center_x) * self.MAZE_SIZE_SCALING + self.center[1])
        else:
            object_xpos = (self.object_coords[1] * self.MAZE_SIZE_SCALING + self.center[0], self.object_coords[0] * self.MAZE_SIZE_SCALING + self.center[1])
            self.object_init_pos = (self.object_coords[0], self.object_coords[1])
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        # self.images = [self.render('rgb_array', subgoals=None)]
        return True

    def _sample_goal(self):
        self.goal = np.zeros(3, dtype=np.float32)
        if self.object_init_pos is None:
            return self.goal
        if self.target_coords is None:
            center_x, center_y = int(len(self.MAZE_STRUCTURE)/2), int(len(self.MAZE_STRUCTURE[0])/2)
            x, y = None, None
            while x is None or self.MAZE_STRUCTURE[len(self.MAZE_STRUCTURE)-x-1][y] != 0 or ((x-center_x) == self.object_init_pos[0] and (y-center_y) == self.object_init_pos[1]):
                # Keep margins of size 2 when sample.
                x, y = np.random.randint(2, len(self.MAZE_STRUCTURE)-2), np.random.randint(2, len(self.MAZE_STRUCTURE[0])-2)
                self.goal[:2] = ((y - center_y) * self.MAZE_SIZE_SCALING + self.center[0], (x-center_x) * self.MAZE_SIZE_SCALING + self.center[1])
                # self.target_coords = (x-center_x, y-center_y)
        else:    
            self.goal[:2] = (self.target_coords[1] * self.MAZE_SIZE_SCALING + self.center[0], self.target_coords[0] * self.MAZE_SIZE_SCALING + self.center[1])

        # goal += self.target_offset
        self.goal[2] = self.height_offset
        
        return self.goal.copy()

    def _render_subgoals(self, viewer, subgoals):
        colors = {
            1: [0.729, 0.0117, 0.988, 1.0],  # purple
            0: [0.0, 1.0, 0., 1.0],  # green
        }
        # Shapes taken from http://mujoco.org/book/source/mjmodel.h (_mjtGeom)
        shapes = {
            0: 6,  # box
            1: 5,  # ellipsoid
        }
        for level, subgoal in subgoals.items():
            if subgoal is None or level == 2:
                continue
            if subgoal.size == 2:
                subgoal = subgoal.flatten()
                viewer.add_marker(pos=[subgoal[0]+self.center[0], subgoal[1]+self.center[1], 0.45], label='', size=[0.01, 0.01, 0.01], rgba=colors[level])
            else:
                print("WARNING: subgoal has wrong dimension", subgoal)

    def render(self, *args, **kwargs):
        if args:
            mode = args[0]
        else:
            mode = kwargs['mode']
        viewer = self._get_viewer(mode)
        del viewer._markers[:]
        if 'subgoals' in kwargs:
            subgoals = kwargs.pop("subgoals")
            if subgoals:
                self._render_subgoals(viewer, subgoals)
        return super(PushEnv, self).render(*args, **kwargs)
