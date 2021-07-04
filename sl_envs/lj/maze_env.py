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

"""Adapted from rllab maze_env.py."""

import os
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np
import gym
import random

from . import maze_env_utils

# Directory that contains mujoco xml files.
# MODEL_DIR = '/home/lukas/miniconda3/envs/tamer/lib/python3.6/site-packages/gym/envs/mujoco/assets/'
MODULE_DIR, _ = os.path.split(os.path.realpath(__file__))
MODEL_DIR = os.path.join(MODULE_DIR, 'assets/')
GATE_HEIGHT = 50

class MazeEnv(gym.Env):
    MODEL_CLASS = None

    MAZE_HEIGHT = None
    MAZE_SIZE_SCALING = None

    def __init__(
        self,
        maze_id=None,
        maze_height=0.5,
        passage_width=0.3,
        maze_size_scaling=8,
        n_bins=0,
        sensor_range=3.,
        sensor_span=2 * math.pi,
        observe_blocks=False,
        put_spin_near_agent=False,
        top_down_view=False,
        manual_collision=False,
        random_goal=False,
        dict_observation=False,
        observe_passage=False,
        relative_goal=False,
        start_pos=None,
        goal_pos=None,
        reward_fn='inner',
        hac_mode=False,
        extra_dims=False,
        key_gate_passage=False,
        maze_structure=None,
        seed=0,
        *args,
        **kwargs):
        self._maze_id = maze_id

        self.metadata['render.subgoals'] = True
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        print("model_dir", MODEL_DIR, MODULE_DIR, flush=True)
        xml_path = os.path.join(MODEL_DIR, model_cls.FILE)
        if hac_mode:
            xml_path = os.path.join(MODEL_DIR, model_cls.FILE_HAC)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self.MAZE_HEIGHT = height = maze_height
        self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
        self._n_bins = n_bins
        self._sensor_range = sensor_range * size_scaling
        self._sensor_span = sensor_span
        self._observe_blocks = observe_blocks
        self._observe_passage = observe_passage
        self._put_spin_near_agent = put_spin_near_agent
        self._top_down_view = top_down_view
        self._manual_collision = manual_collision
        self._random_goal = random_goal
        self._passage_width = passage_width
        self._passage_wall_width = 1 - self._passage_width
        self._reward_fn = reward_fn
        self._dict_observation = dict_observation
        self._relative_goal = relative_goal
        self._hac_mode = hac_mode
        self._extra_dims = extra_dims
        self._key_gate_passage = key_gate_passage

        self.MAZE_STRUCTURE = structure = maze_env_utils.construct_maze(maze_id=self._maze_id) if maze_structure is None else maze_structure
        self.elevated = any(-1 in row for row in structure)  # Elevate the maze to allow for falling.
        self.blocks = any(
            any(maze_env_utils.can_move(r) for r in row)
            for row in structure)  # Are there any movable blocks?

        torso_x, torso_y = self._find_robot()
        print(torso_x, torso_y)
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y
        if start_pos is None:
            self._init_positions = [
                (x - torso_x, y - torso_y)
                for x, y in self._find_all_robots()]
            if key_gate_passage:
                self._init_positions = [(x*size_scaling - torso_x, y*size_scaling-torso_y) for (x,y) in [(2, 6), (3, 6), (2, 5), (3, 5)]]
        else:
            self._init_positions = [(start_pos[0] - torso_x, start_pos[1] - torso_y)]
            print("****** init positions", self._init_positions)
        self.goal = None
        self._goal_pos = goal_pos
        if goal_pos is None:
            self._goal_positions = [
                (x - torso_x, y - torso_y)
                for x, y in self._find_all_goals()]
        else:
            self._goal_positions = [(goal_pos[0] - torso_x, goal_pos[1] - torso_y)]

        self._xy_to_rowcol = lambda x, y: (2 + (y + size_scaling / 2) / size_scaling,
                                           2 + (x + size_scaling / 2) / size_scaling)
        # A diagram of the maze in a small top_down_view.
        self._view = np.zeros([5, 5, 3])  # walls (immovable), chasms (fall), movable blocks

        height_offset = 0.
        if self.elevated:
            # Increase initial z-pos of ant.
            height_offset = height * size_scaling
            torso = tree.find(".//body[@name='torso']")
            torso.set('pos', '0 0 %.2f' % (0.75 + height_offset))
        if self.blocks:
            # If there are movable blocks, change simulation settings to perform
            # better contact detection.
            default = tree.find(".//default")
            default.find('.//geom').set('solimp', '.995 .995 .01')

        self.movable_blocks = []
        self.passage = None
        self.gate = None
        key_location = None
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                struct = structure[i][j]
                if struct == 'K':
                    key_location = [j * size_scaling - torso_x, i * size_scaling - torso_y]
                if struct == 'r' and self._put_spin_near_agent:
                    struct = maze_env_utils.Move.SpinXY
                if self.elevated and struct not in [-1]:
                    # Create elevated platform.
                    ET.SubElement(
                        worldbody, "geom",
                        name="elevated_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.9 0.9 1",
                    )
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height_offset +
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1",
                    )
                if struct == '|':  # Thin passage through unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    name = "passage_%d_%d" % (i, j)
                    self.passage = [j * size_scaling - torso_x, i * size_scaling - torso_y - (self._passage_width/2) * size_scaling]
                    print("Setting passage", self.passage)
                    ET.SubElement(
                        worldbody, "geom",
                        name=name,
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y - (self._passage_width/2) * size_scaling,
                                          height_offset +
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           (self._passage_wall_width/2) * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.4 0.4 1",
                    )
                elif maze_env_utils.can_move(struct):  # Movable block.
                    # The "falling" blocks are shrunk slightly and increased in mass to
                    # ensure that it can fall easily through a gap in the platform blocks.
                    name = "movable_%d_%d" % (i, j)
                    self.movable_blocks.append((name, struct))
                    falling = maze_env_utils.can_move_z(struct)
                    spinning = maze_env_utils.can_spin(struct)
                    x_offset = 0.25 * size_scaling if spinning else 0.0
                    y_offset = 0.0
                    shrink = 0.1 if spinning else 0.99 if falling else 1.0
                    height_shrink = 0.1 if spinning else 1.0
                    movable_body = ET.SubElement(
                        worldbody, "body",
                        name=name,
                        pos="%f %f %f" % (j * size_scaling - torso_x + x_offset,
                                          i * size_scaling - torso_y + y_offset,
                                          height_offset +
                                          height / 2 * size_scaling * height_shrink),
                    )
                    ET.SubElement(
                        movable_body, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="0 0 0",
                        size="%f %f %f" % (0.5 * size_scaling * shrink,
                                           0.5 * size_scaling * shrink,
                                           height / 2 * size_scaling * height_shrink),
                        type="box",
                        material="",
                        mass="0.001" if falling else "0.0002",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.1 0.1 1"
                    )
                    if maze_env_utils.can_move_x(struct):
                        ET.SubElement(
                            movable_body, "joint",
                            armature="0",
                            axis="1 0 0",
                            damping="0.0",
                            limited="true" if falling else "false",
                            range="%f %f" % (-size_scaling, size_scaling),
                            margin="0.01",
                            name="movable_x_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide"
                        )
                    if maze_env_utils.can_move_y(struct):
                        ET.SubElement(
                            movable_body, "joint",
                            armature="0",
                            axis="0 1 0",
                            damping="0.0",
                            limited="true" if falling else "false",
                            range="%f %f" % (-size_scaling, size_scaling),
                            margin="0.01",
                            name="movable_y_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide"
                        )
                    if maze_env_utils.can_move_z(struct):
                        ET.SubElement(
                            movable_body, "joint",
                            armature="0",
                            axis="0 0 1",
                            damping="0.0",
                            limited="true",
                            range="%f 0" % (-height_offset),
                            margin="0.01",
                            name="movable_z_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide"
                        )
                    if maze_env_utils.can_spin(struct):
                        ET.SubElement(
                            movable_body, "joint",
                            armature="0",
                            axis="0 0 1",
                            damping="0.0",
                            limited="false",
                            name="spinable_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="ball"
                        )
                if struct == 'G':  # Wall that can be lifted and fixed in the air.
                    # The "falling" blocks are shrunk slightly and increased in mass to
                    # ensure that it can fall easily through a gap in the platform blocks.
                    name = "gate_%d_%d" % (i, j)
                    shrink = 0.99
                    movable_body = ET.SubElement(
                        worldbody, "body",
                        name=name,
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height_offset + GATE_HEIGHT +
                                          height / 2 * size_scaling),
                    )
                    ET.SubElement(
                        movable_body, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="0 0 0",  # Relative coordinates.
                        size="%f %f %f" % (0.5 * size_scaling * shrink,
                                           0.5 * size_scaling * shrink,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        # mass="0.001" if falling else "0.0002",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.1 0.1 1"
                    )
                    ET.SubElement(
                        movable_body, "joint",
                        armature="0",
                        axis="0 0 1",
                        damping="0.0",
                        limited="true",
                        range="%f %f" % (-GATE_HEIGHT, 0),
                        margin="0.01",
                        name="movable_z_%d_%d" % (i, j),
                        pos="0 0 0",
                        type="slide"
                    )
                    self.gate = {
                        'gate_name': name, 
                        'moveable_name': "movable_z_%d_%d" % (i, j),
                        'gate_cell_coords': (j ,i), 
                        'closed':True
                    }
        
        assert (self.gate is None) == (key_location is None), "Both gate and key location need to be defined or neither."
        if self.gate is not None:
            self.gate['key_location'] = key_location
            mujoco_node = tree.getroot()
            equality_node = ET.SubElement(mujoco_node, "equality")
            gate_name = self.gate['gate_name']
            ET.SubElement(equality_node, "weld", body1=gate_name, active='true', name='eq_'+gate_name)


        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        file_handle, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        self.wrapped_env = model_cls(*args, file_path=file_path, **kwargs)
        self.wrapped_env.seed(seed)
        os.close(file_handle)
        os.remove(file_path)

    def get_ori(self):
        """ Returns the orientation of the underlying robot (Ant) class. """
        return self.wrapped_env.get_ori()

    def get_top_down_view(self):
        """
    For the ‘Images’ versions of these environments, we zero-out the x, y coordinates in the observation
    and append a low-resolution 5 × 5 × 3 top-down view of the environment. The view is centered on
    the agent and each pixel covers the size of a large block (size equal to width of the corridor in Ant
    Maze). The 3 channels correspond to (1) immovable blocks (walls, gray in the videos), (2) movable
    blocks (shown in red in videos), and (3) chasms where the agent may fall.
    """
        self._view = np.zeros_like(self._view)

        def valid(row, col):
            return self._view.shape[0] > row >= 0 and self._view.shape[1] > col >= 0

        def update_view(x, y, d, row=None, col=None):
            """ Updates the voxel at x,y,d coordinates of the view that we are creating. Row and Col are coordinates in the structure. """
            if row is None or col is None:
                x = x - self._robot_x
                y = y - self._robot_y
                th = self._robot_ori

                row, col = self._xy_to_rowcol(x, y)
                update_view(x, y, d, row=row, col=col)
                return

            row, row_frac, col, col_frac = int(row), row % 1, int(col), col % 1
            if row_frac < 0:
                row_frac += 1
            if col_frac < 0:
                col_frac += 1

            if valid(row, col):
                self._view[row, col, d] += (
                    (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
                    (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
            if valid(row - 1, col):
                self._view[row - 1, col, d] += (
                    (max(0., 0.5 - row_frac)) *
                    (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
            if valid(row + 1, col):
                self._view[row + 1, col, d] += (
                    (max(0., row_frac - 0.5)) *
                    (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
            if valid(row, col - 1):
                self._view[row, col - 1, d] += (
                    (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
                    (max(0., 0.5 - col_frac)))
            if valid(row, col + 1):
                self._view[row, col + 1, d] += (
                    (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
                    (max(0., col_frac - 0.5)))
            if valid(row - 1, col - 1):
                self._view[row - 1, col - 1, d] += (
                    (max(0., 0.5 - row_frac)) * max(0., 0.5 - col_frac))
            if valid(row - 1, col + 1):
                self._view[row - 1, col + 1, d] += (
                    (max(0., 0.5 - row_frac)) * max(0., col_frac - 0.5))
            if valid(row + 1, col + 1):
                self._view[row + 1, col + 1, d] += (
                    (max(0., row_frac - 0.5)) * max(0., col_frac - 0.5))
            if valid(row + 1, col - 1):
                self._view[row + 1, col - 1, d] += (
                    (max(0., row_frac - 0.5)) * max(0., 0.5 - col_frac))

        # Draw ant.
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        self._robot_x = robot_x
        self._robot_y = robot_y
        self._robot_ori = self.get_ori()

        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        height = self.MAZE_HEIGHT

        # Draw immovable blocks and chasms.
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:  # Wall.
                    update_view(j * size_scaling - self._init_torso_x,
                                i * size_scaling - self._init_torso_y,
                                0)
                if structure[i][j] == -1 or structure[i][j] == 'P':  # Chasm or passage.
                    update_view(j * size_scaling - self._init_torso_x,
                                i * size_scaling - self._init_torso_y,
                                1)

        # Draw movable blocks.
        for block_name, block_type in self.movable_blocks:
            block_x, block_y = self.wrapped_env.get_body_com(block_name)[:2]
            update_view(block_x, block_y, 2)

        return self._view

    def get_range_sensor_obs(self):
        """Returns egocentric range sensor observations of maze."""
        robot_x, robot_y, robot_z = self.wrapped_env.get_body_com("torso")[:3]
        ori = self.get_ori()

        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        height = self.MAZE_HEIGHT

        segments = []
        # Get line segments (corresponding to outer boundary) of each immovable
        # block or drop-off.
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] in [1, -1, '|', 'G']:  # There's a wall, drop-off or passage.
                    if structure[i][j] == '|':
                        cx = j * size_scaling - self._init_torso_x
                        cy = i * size_scaling - self._init_torso_y - (self._passage_width/2) * size_scaling
                        x1 = cx - 0.5 * size_scaling
                        x2 = cx + 0.5 * size_scaling
                        y1 = cy - (self._passage_wall_width/2) * size_scaling
                        y2 = cy + (self._passage_wall_width/2) * size_scaling
                    else:
                        if structure[i][j] == 'G' and not self.gate['closed']:
                            continue
                        cx = j * size_scaling - self._init_torso_x
                        cy = i * size_scaling - self._init_torso_y
                        x1 = cx - 0.5 * size_scaling
                        x2 = cx + 0.5 * size_scaling
                        y1 = cy - 0.5 * size_scaling
                        y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))
        # Get line segments (corresponding to outer boundary) of each movable
        # block within the agent's z-view.
        for block_name, block_type in self.movable_blocks:
            block_x, block_y, block_z = self.wrapped_env.get_body_com(block_name)[:3]
            if (block_z + height * size_scaling / 2 >= robot_z and
                robot_z >= block_z - height * size_scaling / 2):  # Block in view.
                x1 = block_x - 0.5 * size_scaling
                x2 = block_x + 0.5 * size_scaling
                y1 = block_y - 0.5 * size_scaling
                y2 = block_y + 0.5 * size_scaling
                struct_segments = [
                    ((x1, y1), (x2, y1)),
                    ((x2, y1), (x2, y2)),
                    ((x2, y2), (x1, y2)),
                    ((x1, y2), (x1, y1)),
                ]
                for seg in struct_segments:
                    segments.append(dict(
                        segment=seg,
                        type=block_type,
                    ))

        sensor_readings = np.zeros((self._n_bins, 3))  # 3 for wall, drop-off, block
        for ray_idx in range(self._n_bins):
            ray_ori = (ori - self._sensor_span * 0.5 +
                       (2 * ray_idx + 1.0) / (2 * self._n_bins) * self._sensor_span)
            ray_segments = []
            # Get all segments that intersect with ray.
            for seg in segments:
                p = maze_env_utils.ray_segment_intersect(
                    ray=((robot_x, robot_y), ray_ori),
                    segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=maze_env_utils.point_distance(p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                # Find out which segment is intersected first.
                first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
                seg_type = first_seg["type"]
                idx = (0 if seg_type in [1,'|', 'G'] else  # Wall or passage wall or gate.
                       1 if seg_type == -1 else  # Drop-off.
                       2 if maze_env_utils.can_move(seg_type) else  # Block.
                       None)
                if first_seg["distance"] <= self._sensor_range:
                    sensor_readings[ray_idx][idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range

        return sensor_readings

    def _get_obs_components(self, append_goal):
        wrapped_obs = self.wrapped_env._get_obs()
        if self._top_down_view:
            view = [self.get_top_down_view().flat]
        else:
            view = []

        if append_goal:
            goal = [self.goal]
        else:
            goal = []

        key_collected = []
        if self.gate is not None:
            if self.gate['closed']:
                key_collected = [[0]]
            else:
                key_collected = [[1]]

        if self._observe_blocks:
            additional_obs = []
            for block_name, block_type in self.movable_blocks:
                additional_obs.append(self.wrapped_env.get_body_com(block_name))
            wrapped_obs = np.concatenate([wrapped_obs[:3]] + additional_obs +
                                         [wrapped_obs[3:]])
        if self._observe_passage:
            wrapped_obs = np.concatenate([wrapped_obs[:3]] + [self.passage] +
                                         [wrapped_obs[3:]])

        if self._hac_mode:
            return np.concatenate([wrapped_obs] +
                                   view + goal + key_collected)
                                   
        range_sensor_obs = self.get_range_sensor_obs()
        return np.concatenate([wrapped_obs,
                               range_sensor_obs.flat] +
                               view + goal + key_collected + [[self.t * 0.001]])

    def _get_obs(self):
        if self._dict_observation:
            obs = self._get_obs_components(append_goal=False)
            desired_goal = np.array(self.goal, dtype=np.float32) - np.array(self.wrapped_env.get_xy(), dtype=np.float32) if self._relative_goal else np.array(self.goal, dtype=np.float32)
            return {
                'observation': obs,
                'achieved_goal': np.array(self.wrapped_env.get_xy()),
                'desired_goal': desired_goal,
            }
        else:
            return self._get_obs_components(append_goal=True)

    def reset(self):
        self.t = 0
        self.trajectory = []
        self.wrapped_env.reset()
        if self.gate is not None:
            self._close_gate()
        xy = [self._init_torso_x, self._init_torso_y]
        if len(self._init_positions) > 0:
            xy = random.choice(self._init_positions)
            self.wrapped_env.set_xy(xy)
        if self._random_goal:
            if self._maze_id == "KeyGate" and not self._goal_pos and self._key_gate_passage:
                goal_blocks = [(6, 5), (6, 6), (5, 5), (5, 6)]
                block_x, block_y = goal_blocks[np.random.randint(len(goal_blocks))]
                start_x = block_x * self.MAZE_SIZE_SCALING - self._init_torso_x - 0.4 * self.MAZE_SIZE_SCALING
                end_x = block_x * self.MAZE_SIZE_SCALING - self._init_torso_x + 0.4 * self.MAZE_SIZE_SCALING
                start_y = block_y * self.MAZE_SIZE_SCALING - self._init_torso_y - 0.4 * self.MAZE_SIZE_SCALING
                end_y = block_y * self.MAZE_SIZE_SCALING - self._init_torso_y + 0.4 * self.MAZE_SIZE_SCALING
                self.goal = np.random.uniform([start_x,start_y],[end_x,end_y])
            elif self._maze_id in ["KeyGate", "WallRandom", "MazeRandom"] and not self._goal_pos:
                block_x, block_y = 0, 0
                while self.MAZE_STRUCTURE[block_y][block_x] == 1:
                    block_x = np.random.randint(0, len(self.MAZE_STRUCTURE[0]))
                    block_y = np.random.randint(0, len(self.MAZE_STRUCTURE))
                start_x = block_x * self.MAZE_SIZE_SCALING - self._init_torso_x - 0.4 * self.MAZE_SIZE_SCALING
                end_x = block_x * self.MAZE_SIZE_SCALING - self._init_torso_x + 0.4 * self.MAZE_SIZE_SCALING
                start_y = block_y * self.MAZE_SIZE_SCALING - self._init_torso_y - 0.4 * self.MAZE_SIZE_SCALING
                end_y = block_y * self.MAZE_SIZE_SCALING - self._init_torso_y + 0.4 * self.MAZE_SIZE_SCALING
                self.goal = np.random.uniform([start_x,start_y],[end_x,end_y])
            elif self._maze_id in ["Empty"] and not self._goal_pos:
                center_x = ((len(self.MAZE_STRUCTURE[0]) * self.MAZE_SIZE_SCALING) - 1.8 * self.MAZE_SIZE_SCALING) - self._init_torso_x
                center_y = ((len(self.MAZE_STRUCTURE) * self.MAZE_SIZE_SCALING) - 1.8*self.MAZE_SIZE_SCALING) - self._init_torso_y
                self.goal = np.random.uniform([-center_x,-center_y],[center_x,center_y])
            else:
                self.goal = random.choice(self._goal_positions)
                while self.goal == xy:
                    self.goal = random.choice(self._goal_positions)
        return self._get_obs()

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    def _add_goal(self, viewer):
        goal_xy = self.goal
        viewer.add_marker(pos=[goal_xy[0], goal_xy[1], 0.5], label='', type=2, size=[.5, .5, .5], rgba=[1., 0., 0., 1.])
        # ant_pos = self.wrapped_env.get_xy()
        # viewer.add_marker(pos=[ant_pos[0], ant_pos[1], 1], label=("ant"+str(ant_pos)+", reward:"+str(self._compute_reward(0))))
        # if self._random_goal:
        # viewer.add_marker(pos=[self.passage[0], self.passage[1], 1], label=(str(self.passage)))

    def visualize_subgoals(self, subgoals, viewer):
        colors = {
            0: [0.729, 0.0117, 0.988, 1.0],  # purple
            1: [0.0, 1.0, 0., 0.0],  # green
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
                viewer.add_marker(pos=[subgoal[0], subgoal[1], 2.5], label='', size=[0.2, 0.2, 0.2], rgba=colors[level])
            else:
                print("WARNING: subgoal has wrong dimension", subgoal)

    def snap(self):
        mode = 'rgb_array'
        viewer = self.wrapped_env._get_viewer(mode)
        del viewer._markers[:]
        # self._add_goal(viewer)
        return self.wrapped_env.render(mode=mode, width=32, height=32)

    def render(self, *args, **kwargs):
        mode = 'human'
        if args:
            mode = args[0]
        elif 'mode' in kwargs:
            mode = kwargs['mode']
        viewer = self.wrapped_env._get_viewer(mode)
        del viewer._markers[:]
        if self._random_goal:
            self._add_goal(viewer)
        if self.gate is not None:
            #  TODO: Change type from box to ball or something else.
            key = self.gate['key_location']
            if self.gate['closed']:
                viewer.add_marker(pos=[key[0], key[1], 1], label=("KEY"))

        subgoals = kwargs.pop('subgoals', None)
        if subgoals:
            self.visualize_subgoals(subgoals, viewer)
        return self.wrapped_env.render(*args, **kwargs)

    def _set_constraint(self, constraint_name, value):
        from mujoco_py import functions
        model = self.wrapped_env.sim.model
        mjOBJ_EQUALITY = 16
        eq_constraint_id = functions.mj_name2id(model, mjOBJ_EQUALITY, constraint_name)
        assert eq_constraint_id >= 0, "Constraint %s not found" % constraint_name
        model.eq_active[eq_constraint_id] = value

    def _open_gate(self):
        assert self.gate is not None
        gate_name, moveable_name = self.gate['gate_name'], self.gate['moveable_name']
        self.wrapped_env.sim.data.set_joint_qpos(moveable_name, 0)
        self._set_constraint("eq_"+gate_name, True)
        self.gate['closed'] = False

    def _close_gate(self):
        assert self.gate is not None
        gate_name, moveable_name = self.gate['gate_name'], self.gate['moveable_name']
        self._set_constraint("eq_"+gate_name, False)
        self.wrapped_env.sim.data.set_joint_qpos(moveable_name, -GATE_HEIGHT)
        self.gate['closed'] = True

    @property
    def observation_space(self):
        def get_obs_space_per_arr(arr):
          shape = arr.shape
          high = np.inf * np.ones(shape)
          low = -high
          return gym.spaces.Box(low, high)

        if self._dict_observation:
            obs = self._get_obs()
            assert isinstance(obs, dict)
            return gym.spaces.Dict({
                key: get_obs_space_per_arr(value)
                for key, value in obs.items()
            })
        else:
            return get_obs_space_per_arr(self._get_obs())

    @property
    def observation_keys(self):
        return ['desired_goal', 'observation'] if self._dict_observation else None

    @property
    def subgoal_space(self):
        min_y, max_y = 0 - self._init_torso_y, (len(self.MAZE_STRUCTURE)-1) * self.MAZE_SIZE_SCALING - self._init_torso_y
        min_x, max_x = 0 - self._init_torso_x, (len(self.MAZE_STRUCTURE[0])-1) * self.MAZE_SIZE_SCALING - self._init_torso_x
        min_y, max_y = float(min_y), float(max_y)
        min_x, max_x = float(min_x), float(max_x)
        if self._extra_dims:
            return gym.spaces.Box(low=np.array([min_x, min_y, 0., -3., -3.]), high=np.array([max_x, max_y, 1., 3., 3.]))    
        return gym.spaces.Box(low=np.array([min_x, min_y]), high=np.array([max_x, max_y]))

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def _find_robot(self):
        if self._hac_mode:
            # Center the environment
            center_x = ((len(self.MAZE_STRUCTURE[0]) * self.MAZE_SIZE_SCALING) - self.MAZE_SIZE_SCALING)/2
            center_y = ((len(self.MAZE_STRUCTURE) * self.MAZE_SIZE_SCALING) - self.MAZE_SIZE_SCALING)/2
            return center_x, center_y
        else:
            structure = self.MAZE_STRUCTURE
            size_scaling = self.MAZE_SIZE_SCALING
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if structure[i][j] == 'r':
                        return j * size_scaling, i * size_scaling
            assert False, 'No robot in maze specification.'

    def _find_all_robots(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        coords = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    coords.append((j * size_scaling, i * size_scaling))
        return coords

    def _find_all_goals(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        coords = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == '+':
                    coords.append((j * size_scaling, i * size_scaling))
        if len(coords) == 0:
            return self._find_all_robots()
        return coords

    def _is_in_collision(self, pos):
        x, y = pos
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False

    def _compute_distance(self, x, y):
      return np.sqrt(np.sum(np.square(np.array(x) - np.array(y))) + 1e-8)  

    def compute_reward(self, achieved_goal, desired_goal, info={}):
      reward_fn = self._reward_fn
      if 'reward_fn' in info:
        reward_fn = info['reward_fn']
      assert reward_fn in ['negative_distance', 'sparse']

      reward = 0
      distance = self._compute_distance(desired_goal, achieved_goal)
      if distance <=1:
          info['done'] = True
      if reward_fn == 'negative_distance':
            reward = -distance
      elif reward_fn == 'sparse':
          if distance <= 0.5:
              reward = 0
          else:
              reward = -1
      return reward

    def _compute_reward(self, inner_reward, xy_old=None, info=None):
        if info is None:
            # If info is not specified then create an empty dict that will be discarded.
            info = {}
        info['inner_reward'] = inner_reward
        info['negative_distance'] = self.compute_reward(self.wrapped_env.get_xy(), self.goal, info={'reward_fn':'negative_distance'})
        if self._reward_fn == 'inner':
            return inner_reward
        elif self._reward_fn == 'negative_distance_diff':
            if xy_old is not None:
                old_reward = self.compute_reward(xy_old, self.goal, info={'reward_fn':'negative_distance'})
                new_reward = self.compute_reward(self.wrapped_env.get_xy(), self.goal, info={'reward_fn':'negative_distance'})
                return old_reward - new_reward
            else:
                return self.compute_reward(self.wrapped_env.get_xy(), self.goal, info={'reward_fn':'negative_distance'})
        else:
            return self.compute_reward(self.wrapped_env.get_xy(), self.goal)

    def step(self, action):
        self.t += 1
        old_pos = self.wrapped_env.get_xy().copy()
        if self._manual_collision:
            inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            if self._is_in_collision(new_pos):
                self.wrapped_env.set_xy(old_pos)
        else:
            inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)

        if self.gate is not None:
            if self.gate['closed'] and self._compute_distance(self.wrapped_env.get_xy(), self.gate['key_location']) < 0.8:
                self._open_gate()

        next_obs = self._get_obs()
        done = False
        reward = self._compute_reward(inner_reward, xy_old=old_pos, info=info)

        return next_obs, reward, done, info
