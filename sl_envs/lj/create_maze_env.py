# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from .ant_maze_env import AntMazeEnv
from .point_maze_env import PointMazeEnv
from .dm_point_maze_env import DMPointMazeEnv
from .humanoid_maze_env import HumanoidMazeEnv

#import tensorflow as tf
# from tf_agents.environments import gym_wrapper
# from tf_agents.environments import tf_py_environment


# @gin.configurable
def create_maze_env(env_name=None, top_down_view=False, eval_mode=False, hac_mode=False, extra_dims=False, seed=0, maze_structure=None):
  n_bins = 0
  manual_collision = False
  if env_name.startswith('Ego'):
    n_bins = 8
    env_name = env_name[3:]
  if env_name.startswith('Ant'):
    cls = AntMazeEnv
    env_name = env_name[3:]
    maze_size_scaling = 4
  elif env_name.startswith('Point'):
    cls = PointMazeEnv
    manual_collision = True
    env_name = env_name[5:]
    maze_size_scaling = 4
  elif env_name.startswith('DMPoint'):
    cls = DMPointMazeEnv
    manual_collision = True
    env_name = env_name[7:]
    maze_size_scaling = 4
  elif env_name.startswith("Humanoid"):
    cls = HumanoidMazeEnv
    env_name = env_name[8:]
    maze_size_scaling = 4
  else:
    assert False, 'unknown env %s' % env_name

  dict_observation = False
  if env_name.endswith('Dict'):
    env_name = env_name[:-4]
    dict_observation = True

  maze_id = None
  observe_blocks = False
  observe_passage = False
  put_spin_near_agent = False
  random_goal = False
  relative_goal = False
  start_pos = None
  goal_pos = None
  key_gate_passage = False
  reward_fn = 'innner'
  if env_name in ['Maze', "MazeRandom", 'MazeFlipped', 'MazeBackwards']:
    maze_id = env_name
    random_goal = True
    reward_fn = 'negative_distance'
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (1 * maze_size_scaling, 3 * maze_size_scaling)
  elif env_name == 'Passage':
    maze_id = 'Passage'
    random_goal = True
    observe_passage = True
    reward_fn = 'negative_distance'
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (5 * maze_size_scaling, 5 * maze_size_scaling)
  elif env_name == 'EmptyNegDist':
    maze_id = 'Empty'
    random_goal = True
    reward_fn = 'negative_distance'
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (3 * maze_size_scaling, 4 * maze_size_scaling)
  elif env_name == 'EmptyNegDistRelative':
    maze_id = 'Empty'
    random_goal = True
    reward_fn = 'negative_distance'
    relative_goal = True
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (3 * maze_size_scaling, 4 * maze_size_scaling)
  elif env_name == 'Simple':
    maze_id = 'Simple'
    random_goal = True
    reward_fn = 'negative_distance'
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 2 * maze_size_scaling)
      goal_pos = (3 * maze_size_scaling, 3 * maze_size_scaling)
  elif env_name == 'KeyGate':
    maze_id = 'KeyGate'
    random_goal = True
    reward_fn = 'negative_distance'
    if eval_mode:
      start_pos = (2 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (6 * maze_size_scaling, 1 * maze_size_scaling)
  elif env_name == 'KeyGateBackwards':
    eval_mode = True
    maze_id = 'KeyGate'
    random_goal = True
    reward_fn = 'negative_distance'
    if eval_mode:
      goal_pos = (2 * maze_size_scaling, 1 * maze_size_scaling)
      start_pos = (5 * maze_size_scaling, 1 * maze_size_scaling)
  elif env_name == 'KeyGatePassage':
    maze_id = 'KeyGate'
    random_goal = True
    reward_fn = 'negative_distance'
    key_gate_passage = True
    if eval_mode:
      start_pos = (2 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (6 * maze_size_scaling, 1 * maze_size_scaling)
  elif env_name == 'SimpleRelative':
    maze_id = 'Simple'
    random_goal = True
    reward_fn = 'negative_distance'
    relative_goal = True
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 2 * maze_size_scaling)
      goal_pos = (3 * maze_size_scaling, 3 * maze_size_scaling)
  elif env_name == 'EmptySparse':
    maze_id = 'Empty'
    random_goal = True
    reward_fn = 'sparse'
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (3 * maze_size_scaling, 4 * maze_size_scaling)
  elif env_name == 'EmptyNegDistDiff':
    maze_id = 'Empty'
    random_goal = True
    reward_fn = 'negative_distance_diff'
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (3 * maze_size_scaling, 4 * maze_size_scaling)
  elif env_name == 'EmptyInner':
    maze_id = 'Empty'
    random_goal = True
    reward_fn = 'inner'
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (3 * maze_size_scaling, 4 * maze_size_scaling)
  elif env_name == 'Push':
    maze_id = 'Push'
    random_goal = True
    reward_fn = 'negative_distance'
    if eval_mode:
      start_pos = (2 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (2 * maze_size_scaling, 3 * maze_size_scaling)
  elif env_name == 'Fall':
    maze_id = 'Fall'
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (1 * maze_size_scaling, 4 * maze_size_scaling)
  elif env_name == 'Block':
    maze_id = 'Block'
    put_spin_near_agent = True
    observe_blocks = True
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (3 * maze_size_scaling, 3 * maze_size_scaling)
  elif env_name == 'BlockMaze':
    maze_id = 'BlockMaze'
    put_spin_near_agent = True
    observe_blocks = True
    if eval_mode:
      start_pos = (1 * maze_size_scaling, 1 * maze_size_scaling)
      goal_pos = (1 * maze_size_scaling, 3 * maze_size_scaling)
  elif env_name == 'Custom':
    maze_id = 'Custom'
    random_goal = True
    reward_fn = 'negative_distance'
    assert maze_structure is not None
  elif env_name in ['Wall', 'WallRandom', 'WallTestFlipped', 'WallTestBackwards', 'WallBig', 'WallBigger', 'WallBiggest', 'WallGiant', 'MazeExp', 'MazeExpTest']:
    maze_id = env_name
    random_goal = True
    reward_fn = 'negative_distance'
  else:
    raise ValueError('Unknown maze environment %s' % env_name)

  gym_mujoco_kwargs = {
      'maze_id': maze_id,
      'n_bins': n_bins,
      'observe_blocks': observe_blocks,
      'put_spin_near_agent': put_spin_near_agent,
      'top_down_view': top_down_view,
      'manual_collision': manual_collision,
      'maze_size_scaling': maze_size_scaling,
      'random_goal': random_goal, 
      'observe_passage' : observe_passage,
      'reward_fn' : reward_fn,
      'dict_observation' : dict_observation,
      'relative_goal': relative_goal,
      'start_pos': start_pos,
      'goal_pos': goal_pos,
      'hac_mode': hac_mode,
      'extra_dims': extra_dims,
      'key_gate_passage': key_gate_passage,
      'maze_structure': maze_structure,
      'seed': seed
  }
  gym_env = cls(**gym_mujoco_kwargs)
  gym_env.reset()
  # wrapped_env = gym_wrapper.GymWrapper(gym_env)
  return gym_env
