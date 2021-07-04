import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type, n_actions=7
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.force_range = 5
        self.object_pos = np.array([0, 0])
        self.n_actions = n_actions

        self.sensor_names = ["robot0:Sjp_shoulder_pan",
                             "robot0:Sjp_shoulder_lift",
                             "robot0:Sjp_upperarm_roll",
                             "robot0:Sjp_elbow_elbow_flex",
                             "robot0:Sjp_forearm_roll",
                             "robot0:Sjp_wrist_flex",
                             "robot0:Sjp_wrist_roll"]

        self.act_names = ["robot0:act_shoulder_pan",
                          "robot0:act_shoulder_lift",
                          "robot0:act_upperarm_roll",
                          "robot0:act_elbow_flex",
                          "robot0:act_forearm_roll",
                          "robot0:act_wrist_flex",
                          "robot0:act_wrist_roll"]

        self.joint_names = ["robot0:shoulder_pan_joint",
                            "robot0:shoulder_lift_joint",
                            "robot0:upperarm_roll_joint",
                            "robot0:elbow_flex_joint",
                            "robot0:forearm_roll_joint",
                            "robot0:wrist_flex_joint",
                            "robot0:wrist_roll_joint"]

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # def compute_reward(self, achieved_goal, goal, info):
    #     # Compute distance between goal and the achieved goal.
    #     d = goal_distance(achieved_goal, goal)
    #
    #     # check if goal was reached, if yes, change goal to 0 force contact
    #     if np.abs(goal - achieved_goal).all() < self.force_range and goal.all() > 1:
    #         goal *= 0
    #
    #     return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        # ensure that we don't change the action outside of this scope
        action = action.copy()

        # because we do not want to change the gripper, I just take it out of the action space
        # # add gripper ctrl, let it be permanently blocked
        # assert gripper_ctrl.shape == (2,)
        # if self.block_gripper:
        #     gripper_ctrl = np.zeros_like(gripper_ctrl)
        # action = np.concatenate([pos_ctrl, gripper_ctrl])

        # control range of actuator
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.

        # Converting actions to control inputs
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range

        # Clipping actions according to allowed ctrl_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        # forces
        # force = self.sim.data.get_sensor('force_sensor_r') + self.sim.data.get_sensor('force_sensor_l')
        force = self.force
        # print('force: {}'.format(force))

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }



    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        # remove following line to learn forces
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        #self.sim.model.site_pos[site_id][1] += 0.01
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.done = False

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            # take out following 1 line to take out randomization of puck start position
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

            # test
            self.object_pos = object_xpos

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            # replace following line with next line and take out goal += self.target_offset to remove rand. of goal
            # goal = np.array([1.7, 0.75, 0.0])
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    # # only force
    # def _sample_goal(self):
    #     goal = 50  # + np.random.randint(-30, 31)
    #     goal = np.hstack((goal, self.object_pos))
    #     return goal.copy().flatten()

    # force and position
    # def _sample_goal(self):
    #     goal = 50  # + np.random.randint(-30, 31)
    #     goal = np.array([goal, goal])
    #     return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    # def _is_success(self, achieved_goal, desired_goal):
    #     d = goal_distance(achieved_goal, desired_goal)
    #     return (d < self.force_range).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper_xpos[0] += 0.1
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
