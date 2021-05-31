"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""

from options import parse_options
from agent import Agent
from run_HAC import run_HAC

import importlib

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()

# Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file. 
# Load the variant dynamically from the variant folder based on the name.
module = importlib.import_module(f"variants.{FLAGS.variant}", __name__)

def get_agent_and_envs(FLAGS):
    agent, env = module.design_agent_and_env(FLAGS)
    if isinstance(env, list):
        train_env, eval_env = env
    else:
        train_env, eval_env = env, env
    return agent, train_env, eval_env

# Begin training
if FLAGS.exp_num >= 0 or FLAGS.test:
    agent, train_env, eval_env = get_agent_and_envs(FLAGS)
    run_HAC(FLAGS,train_env,agent, eval_env)
    del agent
    del train_env
    del eval_env
else:
    for exp_num in range(1, 6):
        print("Running experiment ", exp_num)
        FLAGS.exp_num = exp_num
        agent, train_env, eval_env = get_agent_and_envs(FLAGS)
        run_HAC(FLAGS,train_env,agent, eval_env)
        del agent
        del train_env
        del eval_env
