"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""

import pickle as cpickle
import agent as Agent
from utils import print_summary
import numpy as np

TEST_FREQ = 2

num_test_episodes = 100

def run_HAC(FLAGS, train_env, agent, eval_env):

    if not FLAGS.test:
        env = train_env
    else:
        env = eval_env

    # Print task summary
    print_summary(FLAGS, env)
    
    # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not FLAGS.test and not FLAGS.train_only:
        assert FLAGS.exp_name != '', "No experiment name specified."
        mix_train_test = True
     
    for batch in range(agent.other_params['num_batches']):

        num_episodes = agent.other_params["num_exploration_episodes"]
        
        # Reset successful episode counter
        successful_episodes = 0
        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and batch % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            env = eval_env
            agent.FLAGS.test = True
            num_episodes = num_test_episodes
        elif not mix_train_test and FLAGS.test:
            env = eval_env
            num_episodes = 500

        for episode in range(num_episodes):
            
            print("\nExp_num %s Batch %d, Episode %d" % (FLAGS.exp_num, batch, episode))
            
            # Train for an episode
            if vars(FLAGS).get('variable_env', False):
                env.create_new_env()
            success = agent.train(env, episode)

            if success:
                print("Exp_num %s Batch %d, Episode %d End Goal Achieved\n" % (FLAGS.exp_num, batch, episode))
                
                successful_episodes += 1            

        # Save agent
        if FLAGS.retrain and mix_train_test and batch % TEST_FREQ == 0:
            agent.save_model(batch, successful_episodes / num_test_episodes * 100)
        elif not FLAGS.test:
            agent.save_model(batch)
           
        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:
            env = train_env
            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            agent.log_performance(success_rate)
            print("\nTesting Success Rate so far ", agent.performance_log)
            
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")
        elif not mix_train_test and FLAGS.test:
            success_rate = successful_episodes / num_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)

