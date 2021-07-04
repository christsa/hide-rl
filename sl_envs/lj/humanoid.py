import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env
import pdb

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 1,
    'distance': 4.0,
    'lookat': np.array((0.0, 0.0, 2.0)),
    'elevation': -20.0,
}


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = "humanoid.xml"
    FILE_HAC = "humanoid.xml"
    ORI_IND = 1

    def __init__(self,
                 file_path=None,
                 forward_reward_weight=1.25,
                 ctrl_cost_weight=0.1,
                 contact_cost_weight=5e-7,
                 contact_cost_range=(-np.inf, 10.0),
                 healthy_reward=5.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.5, 2.5),
                 reset_noise_scale=1e-2,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self.count = 0

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        self.goal = np.array([0.0,0.0])

        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)


        # self.goal = self.sample_goal()

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward


    def goal_reward(self,xy_position):
        return np.linalg.norm(self.goal-xy_position,axis=-1)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(
            np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z

        return is_healthy

    @property
    def done(self):
        done = ((not self.is_healthy)
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        #pdb.set_trace()
        #site = self.sim.data.get_site_xpos("torso0")
        #ctrl_range = self.sim.model.actuator_ctrlrange[0,1]
        #self.joint_limits = np.count_nonzero(np.abs(j[0::2]) > ctrl_range)
        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        self.body_xy = mass_center(self.model, self.sim)#(parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z

        z = position[2]
        r, p, yaw = self.quaternion_to_euler(position[3],position[4],position[5],position[6])
        #if self.initial_z is None:
        #    self.initial_z = z

        #if self._exclude_current_positions_from_observation:
        position_agent = mass_center(self.model, self.sim)
        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
        ))

    def quaternion_to_euler(self, w, x, y, z):

        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.atan2(t3, t4)

        return X, Y, Z

    def step(self, action):


        #if self.count % 300 == 0:
        #    self.goal=self.sample_goal()
        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        #pdb.set_trace()
        #-2.0 * float(np.abs(action * x_velocity).mean())
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        vel_angle = np.arctan2(y_velocity,x_velocity)
        #print('ang',vel_angle)

        forward_reward = self._forward_reward_weight * np.sqrt((x_velocity**2+y_velocity**2))
        healthy_reward = self.healthy_reward
        goal_reward = self.goal_reward(xy_position_after)/2
        goal_reward_pos = self.tolerance(goal_reward, bounds=(0, 0.5), sigmoid='linear', value_at_margin=0.0, margin=10)
        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        #print('target ang', self.walk_target_theta)
        #pdb.set_trace()
        costs = ctrl_cost + contact_cost # + np.abs(self.walk_target_theta-vel_angle)

        reward = rewards*2.0 - costs #*(0.5+0.5*goal_reward_pos)
        #print((rewards - costs)*0.5*goal_reward_pos)
        #print('goal', goal_reward_pos)

        done = self.done
        info = {
            'reward_linvel': forward_reward,
            'reward_quadctrl': -ctrl_cost,
            'reward_alive': healthy_reward,
            'reward_impact': -contact_cost,
            # 'reward_dir': np.abs(self.angle_to_target),

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }
        self.count += 1
        # if self.count % 1000 == 0:
        #     self.sample_goal()

        return observation, reward, done, info

    def sample_goal(self):
        hum_pos = mass_center(self.model, self.sim).copy()
        hum_pos[0] += np.random.uniform(0.0,7.0)
        hum_pos[1] += np.random.uniform(-5.0,5.0)

        hum_pos_3d = np.concatenate([hum_pos.copy(),[0.5]])
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        #print('bef', self.sim.model.site_pos[site_id])
        self.sim.model.site_pos[site_id] = hum_pos_3d - sites_offset[0]
        #print('after', self.sim.model.site_pos[site_id])
        self.sim.forward()
        return hum_pos

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        # self.goal = self.sample_goal()
        return observation

    def viewer_setup(self):
      self.viewer.cam.lookat[0] = 0
      self.viewer.cam.lookat[1] = 0
      self.viewer.cam.lookat[2] = 0
      # self.viewer.cam.azimuth = 0
      # self.viewer.cam.elevation = -58 # Record video view.
      self.viewer.cam.elevation = -90 # Top down view.
      # self.viewer.cam.distance = 20
      self.viewer.cam.distance = 40.46 # self.model.stat.extent * 1.19
      # print(self.viewer.cam.distance)
      # for key, value in DEFAULT_CAMERA_CONFIG.items():
      #     if isinstance(value, np.ndarray):
      #         getattr(self.viewer.cam, key)[:] = value
      #     else:
      #         setattr(self.viewer.cam, key, value)

    def _sigmoids(self,x, value_at_1, sigmoid):
        """Returns 1 when `x` == 0, between 0 and 1 otherwise.
        Args:
          x: A scalar or numpy array.
          value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
          sigmoid: String, choice of sigmoid type.
        Returns:
          A numpy array with values between 0.0 and 1.0.
        Raises:
          ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
            `quadratic` sigmoids which allow `value_at_1` == 0.
          ValueError: If `sigmoid` is of an unknown type.
        """
        if sigmoid in ('cosine', 'linear', 'quadratic'):
            if not 0 <= value_at_1 < 1:
                raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                                 'got {}.'.format(value_at_1))
        else:
            if not 0 < value_at_1 < 1:
                raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                                 'got {}.'.format(value_at_1))

        if sigmoid == 'gaussian':
            scale = np.sqrt(-2 * np.log(value_at_1))
            return np.exp(-0.5 * (x * scale) ** 2)

        elif sigmoid == 'hyperbolic':
            scale = np.arccosh(1 / value_at_1)
            return 1 / np.cosh(x * scale)

        elif sigmoid == 'long_tail':
            scale = np.sqrt(1 / value_at_1 - 1)
            return 1 / ((x * scale) ** 2 + 1)

        elif sigmoid == 'cosine':
            scale = np.arccos(2 * value_at_1 - 1) / np.pi
            scaled_x = x * scale
            return np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)

        elif sigmoid == 'linear':
            scale = 1 - value_at_1
            scaled_x = x * scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

        elif sigmoid == 'quadratic':
            scale = np.sqrt(1 - value_at_1)
            scaled_x = x * scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0)

        elif sigmoid == 'tanh_squared':
            scale = np.arctanh(np.sqrt(1 - value_at_1))
            return 1 - np.tanh(x * scale) ** 2

        else:
            raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))

    def tolerance(self,x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
                  value_at_margin=0.1):
      """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
      Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
          the target interval. These can be infinite if the interval is unbounded
          at one or both ends, or they can be equal to one another if the target
          value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
          `x` moves out-of-bounds.
          * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
          * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
           'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
          the distance from `x` to the nearest bound is equal to `margin`. Ignored
          if `margin == 0`.
      Returns:
        A float or numpy array with values between 0.0 and 1.0.
      Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
      """
      lower, upper = bounds
      if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
      if margin < 0:
        raise ValueError('`margin` must be non-negative.')

      in_bounds = np.logical_and(lower <= x, x <= upper)
      if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
      else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin, sigmoid))

      return float(value) if np.isscalar(x) else value

    def set_xy(self, xy):
      qpos = np.copy(self.sim.data.qpos)
      qpos[0] = xy[0]
      qpos[1] = xy[1]

      qvel = self.sim.data.qvel
      self.set_state(qpos, qvel)

    def get_xy(self):
      return self.sim.data.qpos[:2]
