import gym
import numpy as np
import importlib
from mujoco_py import MjViewer

class HACEnv(gym.Env):
    def __init__(self, task_id, eval_mode=False, **kwargs):
        module = importlib.import_module(".." + task_id + ".design_agent_and_env", __name__)
        self._env = env = module.design_env()
        goal_space = np.array(env.goal_space_train)
        self.metadata = {'render.subgoals' : True, 'render.modes': ['human']}
        self.observation_space = gym.spaces.Dict({
            'desired_goal': gym.spaces.Box(goal_space[:,0], goal_space[:,1]),
            'achieved_goal': gym.spaces.Box(goal_space[:,0], goal_space[:,1]),
            'achieved_subgoal': gym.spaces.Box(low=env.subgoal_bounds[:,0], high=env.subgoal_bounds[:,1]),
            'observation': gym.spaces.Box(low=env.initial_state_space[:,0], high=env.initial_state_space[:,1]),
        })
        self.observation_keys = ['desired_goal', 'observation']
        self.subgoal_space = gym.spaces.Box(low=env.subgoal_bounds[:,0], high=env.subgoal_bounds[:,1])
        self.action_space = gym.spaces.Box(low=-np.array(env.action_bounds), high=np.array(env.action_bounds))

        self._obs = None
        self._end_goal = None
        self._viewer = None
        self._eval_mode = eval_mode

        from collections import deque
        self.last_obs = deque(maxlen=50)
        self.last_actions = deque(maxlen=50)

    def step(self, action):
        next_obs = self._env.execute_action(action)
        reward = self.compute_reward(
            self._env.project_state_to_end_goal(self._env.sim, next_obs),
            self._end_goal, {})
        self._obs = next_obs
        self.last_actions.append(action)
        self.last_obs.appendleft(next_obs)
        return self._get_obs(), reward, False, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        return -np.sqrt(np.sum(np.square(achieved_goal - desired_goal)) + 1e-8)

    def _get_obs(self):
        return {
                'observation': self._obs,
                'achieved_goal': self._env.project_state_to_end_goal(self._env.sim, self._obs),
                'achieved_subgoal': self._env.project_state_to_subgoal(self._env.sim, self._obs),
                'desired_goal': self._end_goal,
            }

    def reset(self):
        self._end_goal = self._env.get_next_goal(test=self._eval_mode)
        self._env.display_end_goal(self._end_goal)
        self._obs = self._env.reset_sim(self._end_goal, test=self._eval_mode)
        return self._get_obs()


    def key_callback(self, window, key, scancode, action, mods):
        import glfw
        from itertools import islice
        if action == glfw.RELEASE and key == glfw.KEY_A:
            print("\n\n QPOS:\n", self._env.sim.data.qpos)
            print("\n\n QVEL:\n", self._env.sim.data.qvel)
            print("\n\n ControlRange: \n", self._env.model.actuator_ctrlrange, "\n\n")
            last_n_obs = tuple(islice(self.last_obs, None, 49))
            last_n_actions = tuple(islice(self.last_actions, None, 49))
            for i, (obs,action) in enumerate(zip(last_n_obs, last_n_actions)):
                print(i, "\n", obs, action)

    def render(self, *args, **kwargs):
        mode = 'human'
        if args:
            mode = args[0]
        elif 'mode' in kwargs:
            mode = kwargs['mode']
        assert mode == 'human'
        subgoals = kwargs.pop('subgoals', None)
        if subgoals:
            subgoals = [np.squeeze(subgoal) for _,subgoal in subgoals.items() if subgoal is not None]
            self._env.display_subgoals(subgoals)
        if self._viewer is None:
            import glfw
            self._viewer = MjViewer(self._env.sim)
            # glfw.set_key_callback(self._viewer.window, self.key_callback)
        ant_pos = self._get_obs()['achieved_goal']
        self._viewer.add_marker(pos=[ant_pos[0], ant_pos[1], 1], label=("ant"+str(ant_pos)))
        self._viewer.render()

    def close(self):
        self._viewer = None