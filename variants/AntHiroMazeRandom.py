"""
This file provides the template for designing the agent and environment.  The below hyperparameters must be assigned to a value for the algorithm to work properly.
"""

import numpy as np
# from environment import Environment
from environment_adapter import EnvironmentAdapter
from agent import Agent
from collections import OrderedDict

def design_agent_and_env(FLAGS):

    """
    1. DESIGN AGENT

    The key hyperparameters for agent construction are

        a. Number of levels in agent hierarchy
        b. Max sequence length in which each policy will specialize
        c. Max number of atomic actions allowed in an episode
        d. Environment timesteps per atomic action

    See Section 3 of this file for other agent hyperparameters that can be configured.
    """

    FLAGS.layers = 2    # Enter number of levels in agent hierarchy

    FLAGS.time_scale = 40    # Enter max sequence length in which each policy will specialize

    # Enter max number of atomic actions.  This will typically be FLAGS.time_scale**(FLAGS.layers).  However, in the UR5 Reacher task, we use a shorter episode length.


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

    project_state_to_end_goal = lambda sim, state: state[..., :2]
    project_state_to_subgoal = lambda sim, state: state[..., :2]
    env = EnvironmentAdapter("maze", "Ant", "MazeRandomDict", project_state_to_end_goal, project_state_to_subgoal, FLAGS.show, featurize_image=FLAGS.featurize_image)

    eval_env = EnvironmentAdapter("maze", "Ant", "MazeDict", project_state_to_end_goal, project_state_to_subgoal, FLAGS.show, featurize_image=FLAGS.featurize_image)

    # To properly visualize goals, update "display_end_goal" and "display_subgoals" methods in "environment.py"


    """
    3. SET MISCELLANEOUS HYPERPARAMETERS

    Below are some other agent hyperparameters that can affect results, including
        a. Subgoal testing percentage
        b. Subgoal penalty
        c. Exploration noise
        d. Replay buffer size
    """

    agent_params = {}

    # Define percentage of actions that a subgoal level (i.e. level i > 0) will test subgoal actions
    agent_params["subgoal_test_perc"] = 0.3

    # Define subgoal penalty for missing subgoal.  Please note that by default the Q value target for missed subgoals does not include Q-value of next state (i.e, discount rate = 0).  As a result, the Q-value target for missed subgoal just equals penalty.  For instance in this 3-level UR5 implementation, if a level proposes a subgoal and misses it, the Q target value for this action would be -10.  To incorporate the next state in the penalty, go to the "penalize_subgoal" method in the "layer.py" file.
    agent_params["subgoal_penalty"] = -FLAGS.time_scale

    # Define exploration noise that is added to both subgoal actions and atomic actions.  Noise added is Gaussian N(0, noise_percentage * action_dim_range)
    agent_params["atomic_noise"] = [0.1 for i in range(8)]
    agent_params["subgoal_noise"] = [0.01 for i in range(len(env.subgoal_thresholds))]
    agent_params["oracle_noise"] = [0.01 for i in range(len(env.subgoal_thresholds))]
    agent_params["vpn_noise"] = [0.05 for i in range(len(env.subgoal_thresholds))]

    # Define number of episodes of transitions to be stored by each level of the hierarchy
    agent_params["episodes_to_store"] = 500

    # Provide training schedule for agent.  Training by default will alternate between exploration and testing.  Hyperparameter below indicates number of exploration episodes.  Testing occurs for 100 episodes.  To change number of testing episodes, go to "ran_HAC.py".
    agent_params["num_exploration_episodes"] = 100

    # For other relavent agent hyperparameters, please refer to the "agent.py" and "layer.py" files
    agent_params["num_batches"] = 200


    # Ensure environment customization have been properly entered
    # check_validity(model_name, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space, subgoal_bounds, subgoal_thresholds, max_actions, timesteps_per_action)


    # Instantiate and return agent and environment
    # env = Environment(model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions, timesteps_per_action, FLAGS.show)

    agent = Agent(FLAGS,env,agent_params)

    return agent, [env, eval_env]
