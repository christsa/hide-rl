"""
This file provides the template for designing the agent and environment.  The below hyperparameters must be assigned to a value for the algorithm to work properly.
"""

import numpy as np
from .environment import Environment
from ..utils import check_validity

def design_env():

    """
    1. DESIGN AGENT

    The key hyperparameters for agent construction are

        a. Number of levels in agent hierarchy
        b. Max sequence length in which each policy will specialize
        c. Max number of atomic actions allowed in an episode
        d. Environment timesteps per atomic action

    See Section 3 of this file for other agent hyperparameters that can be configured.
    """

    """
    2. DESIGN ENVIRONMENT

        a. Designer must provide the original UMDP (S,A,T,G,R).
            - The S,A,T components can be fulfilled by providing the Mujoco model.
            - The user must separately specifiy the initial state space.
            - G can be provided by specifying the end goal space.
            - R, which by default uses a shortest path {-1,0} reward function, can be implemented by specifying two components: (i) a function that maps the state space to the end goal space and (ii) the end goal achievement thresholds for each dimensions of the end goal.

        b.  In order to convert the original UMDP into a hierarchy of k UMDPs, the designer must also provide
            - The subgoal action space, A_i, for all higher-level UMDPs i > 0
            - R_i for levels 0 <= i < k-1 (i.e., all levels that try to achieve goals in the subgoal space).  As in the original UMDP, R_i can be implemented by providing two components:(i) a function that maps the state space to the subgoal space and (ii) the subgoal achievement thresholds.

        c.  Designer should also provide subgoal and end goal visualization functions in order to show video of training.  These can be updated in "display_subgoal" and "display_end_goal" methods in the "environment.py" file.

    """

    # Provide file name of Mujoco model(i.e., "pendulum.xml").  Make sure file is stored in "mujoco_files" folder
    model_name = "ant_reacher.xml"


    # Provide initial state space consisting of the ranges for all joint angles and velocities.  In the Ant Reacher task, we use a random initial torso position and use fixed values for the remainder.

    initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
    initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
    initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
    initial_joint_ranges[0] = np.array([2.5,21.5])
    initial_joint_ranges[1] = np.array([2.5,21.5])

    # Cocatenate velocity ranges
    initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

    initial_state_space_test = np.copy(initial_state_space)
    initial_state_space_test[0] = np.array([2.5,2.5])
    initial_state_space_test[1] = np.array([2.5,2.5])


    # Provide end goal space.  The code supports two types of end goal spaces if user would like to train on a larger end goal space.  If user needs to make additional customizations to the end goals, the "get_next_goal" method in "environment.py" can be updated.

    # In the UR5 reacher environment, the end goal will be the desired joint positions for the 3 main joints.
    max_range = 9.5
    goal_space_train = [[2.5,21.5],[2.5,21.5],[0.45,0.55]]
    goal_space_test = [[21.5,21.5],[21.5,21.5],[0.55,0.55]]


    # Provide a function that maps from the state space to the end goal space.  This is used to (i) determine whether the agent should be given the sparse reward and (ii) for Hindsight Experience Replay to determine which end goal was achieved after a sequence of actions.
    project_state_to_end_goal = lambda sim, state: state[:3]

    # Set end goal achievement thresholds.  If the agent is within the threshold for each dimension, the end goal has been achieved and the reward of 0 is granted.

    # For the Ant Reacher task, the end goal will be the desired (x,y) position of the torso
    len_threshold = 0.5
    height_threshold = 0.2
    end_goal_thresholds = np.array([len_threshold, len_threshold, height_threshold])


    # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal space can be the same as the state space or some other projection out of the state space.

    # The subgoal space in the Ant Reacher task is the desired (x,y,z) position and (x,y,z) translational velocity of the torso
    cage_max_dim = 11.75
    max_height = 1
    max_velo = 3
    subgoal_bounds = np.array([[0.25,23.75],[0.25,23.75],[0,max_height]])


    # Provide state to subgoal projection function.
    # a = np.concatenate((sim.data.qpos[:2], np.array([4 if sim.data.qvel[i] > 4 else -4 if sim.data.qvel[i] < -4 else sim.data.qvel[i] for i in range(3)])))
    project_state_to_subgoal = lambda sim, state: state[:3]
    # project_state_to_subgoal = lambda sim, state: np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))


    # Set subgoal achievement thresholds
    velo_threshold = 0.5
    quat_threshold = 0.5
    # subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, quat_threshold, quat_threshold, quat_threshold, quat_threshold, velo_threshold, velo_threshold, velo_threshold])
    subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold])


    # To properly visualize goals, update "display_end_goal" and "display_subgoals" methods in "environment.py"


    """
    3. SET MISCELLANEOUS HYPERPARAMETERS

    Below are some other agent hyperparameters that can affect results, including
        a. Subgoal testing percentage
        b. Subgoal penalty
        c. Exploration noise
        d. Replay buffer size
    """
    # Dummy params so that the function works. 
    max_actions = 500
    timesteps_per_action = 15 
    show = False

    # Ensure environment customization have been properly entered
    check_validity(model_name, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space, subgoal_bounds, subgoal_thresholds, max_actions, timesteps_per_action)


    # Instantiate and return agent and environment
    env = Environment(model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, initial_state_space_test, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions, timesteps_per_action, show)

    # agent = Agent(FLAGS,env,agent_params)

    return env
