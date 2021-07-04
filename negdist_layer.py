import numpy as np
from experience_buffer import ExperienceBuffer, PrioritizedReplayBuffer
from sac_actor import SacActor
from bayesian_critic import BayesianCritic
from time import sleep
from collections import defaultdict
from utils import project_state
import torch

class NegDistLayer():
    def __init__(self, layer_number, FLAGS, env, sess, agent_params):
        self.FLAGS = FLAGS
        self.sess = sess
        self.layer_number = layer_number
        self.sl_oracle = False
        self.semi_oracle = False
        self.last_layer = False
        assert self.layer_number == 0
        self.relative_subgoals = self.FLAGS.relative_subgoals and (self.layer_number < self.FLAGS.layers-1)


        # Set time limit for each layer.  If agent uses only 1 layer, time limit is the max number of low-level actions allowed in the episode (i.e, env.max_actions).
        if FLAGS.layers > 1:
            self.time_limit = FLAGS.time_scale
        else:
            self.time_limit = env.max_actions

        if self.FLAGS.no_middle_level:
            self.time_limit = 50

        self.current_state = None
        self.goal = None

        # Initialize Replay Buffer.  Below variables determine size of replay buffer.

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**7

        # Number of full episodes stored in replay buffer
        self.episodes_to_store = agent_params["episodes_to_store"]

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = 2

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        self.trans_per_attempt = self.time_limit
    

        # Buffer size = transitions per attempt * # attempts per episode * num of episodes stored
        self.buffer_size = min(self.trans_per_attempt * self.time_limit**(self.FLAGS.layers-1 - self.layer_number) * self.episodes_to_store, self.buffer_size_ceiling)

        # self.buffer_size = 10000000
        self.batch_size = 1024
        if not FLAGS.test:
            buffer_class = PrioritizedReplayBuffer if (self.FLAGS.priority_replay and not self.sl_oracle) else ExperienceBuffer
            self.replay_buffer = buffer_class(self.buffer_size, self.batch_size, device=self.sess, FLAGS=FLAGS, env=env, layer_number=self.layer_number)

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        # Initialize actor and critic networks
        if self.FLAGS.torch:
            from torch_actor import Actor
            from torch_critic import Critic
        else:
            from tf_actor import Actor
            from tf_critic import Critic
        actor_class = SacActor if self.FLAGS.sac else Actor
        self.actor = actor_class(sess, env, self.batch_size, self.layer_number, FLAGS)
        critic_class = BayesianCritic if FLAGS.bayes else Critic
        self.critic = critic_class(sess, env, self.layer_number, FLAGS)

        # Parameter determines degree of noise added to actions during training
        # self.noise_perc = noise_perc
        self.noise_perc = self.to_torch(agent_params["atomic_noise"])
        self.action_bounds = self.to_torch(env.action_bounds)
        self.action_offset = self.to_torch(env.action_offset)
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        # Create flag to indicate when layer has ran out of attempts to achieve goal.  This will be important for subgoal testing
        self.maxed_out = False

        self.subgoal_penalty = agent_params["subgoal_penalty"]
        
        # Stores metrics for later aggregation
        self.agg_metrics = defaultdict(list)

    def to_torch(self, value):
        return torch.tensor(value, dtype=torch.float32, device=self.sess)

    def copy_transition(self, trans):
        return [None if arr is None else torch.clone(arr) if isinstance(arr, torch.Tensor) else arr for arr in trans]

    # Add noise to provided action
    def add_noise(self,action, env):
        # Noise added will be percentage of range
        assert len(action) == len(self.action_bounds), "Action bounds must have same dimension as action"
        assert len(action) == len(self.noise_perc), "Noise percentage vector must have same dimension as action"

        # Add noise to action and ensure remains within bounds
        action += torch.randn_like(action) * self.noise_perc * self.action_bounds
        # Clip the actions to be in range.
        action = torch.max(torch.min(action, self.action_bounds + self.action_offset), -self.action_bounds+self.action_offset)
        return action

    # Select random action
    def get_random_action(self, env):
        return torch.rand(len(self.action_bounds), dtype=torch.float32, device=self.sess) * 2 * (self.action_bounds) - self.action_bounds + self.action_offset

    # Function selects action using an epsilon-greedy policy
    def choose_action(self,agent, env, subgoal_test):
        action, next_subgoal_test = None, None
        # If testing mode or testing subgoals, action is output of actor network without noise
        if agent.FLAGS.test or subgoal_test:
            current_image = self.current_goal_image.unsqueeze(0) if (self.FLAGS.vpn and self.last_layer) else None
            action = self.actor.get_action(self.current_state.unsqueeze(0), self.goal.unsqueeze(0), current_image, noise=False).squeeze(0)
            next_subgoal_test = subgoal_test
        else:
            if np.random.random_sample() > 0.2:
                # Choose noisy action
                current_image = self.current_goal_image.unsqueeze(0) if (self.FLAGS.vpn and self.last_layer) else None
                action = self.actor.get_action(self.current_state.unsqueeze(0), self.goal.unsqueeze(0), current_image).squeeze(0)
                action = action if self.FLAGS.sac else self.add_noise(action, env)

            # Otherwise, choose random action
            else:
                action = self.get_random_action(env)
                if self.relative_subgoals and self.layer_number > 0:
                    action -= project_state(env, self.FLAGS, self.layer_number, self.current_state)

            # Determine whether to test upcoming subgoal
            if np.random.random_sample() < self.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False

        return action, next_subgoal_test


    # Determine whether layer is finished training
    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached, (ii) maxed out episode time steps (env.max_actions), (iii) not testing and layer is out of attempts, and (iv) testing, layer is not the highest level, and layer is out of attempts.  NOTE: during testing, highest level will continue to ouput subgoals until either (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.

        # Return to previous level when any higher level goal achieved.  NOTE: if not testing and agent achieves end goal, training will continue until out of time (i.e., out of time steps or highest level runs out of attempts).  This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True
        
        if not env.healthy:
            return True

        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.FLAGS.test and self.layer_number < agent.FLAGS.layers-1 and attempts_made >= self.time_limit:
            return True

        else:
            return False

    def get_reward(self, pos, next_pos, action, goal, state, next_state, total_steps_taken):
        if self.FLAGS.relative_subgoals:
            diff = goal + pos - next_pos
            l2_distance = -torch.sqrt(torch.sum(torch.mul(diff, diff))+1e-8)
        else:
            diff = next_pos - goal
            l2_distance = -torch.sqrt(torch.sum(torch.mul(diff, diff))+1e-8)
        if self.FLAGS.negative_distance:
            return l2_distance
        dt = 0.02 * 5  # timestamp * frameskip
        forward_reward = torch.sum(torch.abs((pos[:2] - next_pos[:2]) / dt))
        healthy_reward = 1 if next_state[2] > 0.28 and next_state[2] < 1 else -1
        cost_penalty = -0.05 * torch.sum(action*action)

        alpha = 1 - (min(total_steps_taken, 1e-6) / 1e-6)
        return alpha*(forward_reward + healthy_reward + cost_penalty + l2_distance) + (1-alpha)*l2_distance

    # Learn to achieve goals with actions belonging to appropriate time scale.  "goal_array" contains the goal states for the current layer and all higher layers
    def train(self, agent, env, metrics, subgoal_test = False, episode_num = None):

        # print("\nTraining Layer %d" % self.layer_number)

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number].clone()
        self.current_state = agent.current_state

        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        if self.layer_number == 0 and (agent.FLAGS.show or agent.FLAGS.save_video) and agent.FLAGS.layers > 1:
            env.display_subgoals([arr.cpu().numpy() for arr in agent.goal_array], agent.FLAGS)
            # env.sim.data.mocap_pos[3] = env.project_state_to_end_goal(env.sim,self.current_state)
            # print("Subgoal Pos: ", env.sim.data.mocap_pos[1])

        # Current layer has self.time_limit attempts to each its goal state.
        attempts_made = 0

        while True:

            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, action_type = self.choose_action(agent, env, subgoal_test)

            # Execute low-level action
            next_state = self.to_torch(env.execute_action(action.cpu().numpy()))
            if self.FLAGS.save_video:
                agent.image_path.append(env.render(mode='rgb_array'))

            # Increment steps taken
            agent.steps_taken += 1
            if not self.FLAGS.test:
                agent.total_steps_taken += 1
            # print("Num Actions Taken: ", agent.steps_taken)

            if agent.steps_taken >= env.max_actions:
                print("Out of actions (Steps: %d)" % agent.steps_taken)

            agent.current_state = next_state

            # Determine whether any of the goals from any layer was achieved and, if applicable, the highest layer whose goal was achieved
            if self.FLAGS.relative_subgoals:
                    for i_layer in range(self.FLAGS.layers - 1):
                        old_pos = project_state(env, self.FLAGS, i_layer, self.current_state)
                        new_pos = project_state(env, self.FLAGS, i_layer, agent.current_state)
                        agent.goal_array[i_layer] = agent.goal_array[i_layer] + old_pos - new_pos
            goal_status, max_lay_achieved = agent.check_goals(env)

            attempts_made += 1

            # Transition will take the form [old state, action, reward, next_state, goal, terminate boolean, None]
            if not agent.FLAGS.test and env.healthy:
                if self.layer_number == agent.FLAGS.layers - 1 or (self.layer_number == agent.FLAGS.layers -2 and agent.FLAGS.oracle):
                    position = env.project_state_to_end_goal(env.sim, self.current_state)
                    next_position = env.project_state_to_end_goal(env.sim, agent.current_state)
                else:
                    position = env.project_state_to_subgoal(env.sim, self.current_state)
                    next_position = env.project_state_to_subgoal(env.sim, agent.current_state)
                reward = self.get_reward(position, next_position, action, self.goal, self.current_state, agent.current_state, agent.total_steps_taken)
                transition = [self.current_state, action, reward, agent.current_state, self.goal, goal_status[self.layer_number], None, None]

                self.replay_buffer.add(self.copy_transition(transition))
            elif not agent.FLAGS.test and not env.healthy:
                transition = [self.current_state, action, -100000., agent.current_state, self.goal, goal_status[self.layer_number], None, None]

            # Update state of current layer
            self.current_state = agent.current_state
            if self.relative_subgoals:
                self.goal = agent.goal_array[self.layer_number].clone()
                if self.layer_number == 0 and (agent.FLAGS.show or agent.FLAGS.save_video) and agent.FLAGS.layers > 1:
                    env.display_subgoals([arr.cpu().numpy() for arr in agent.goal_array], agent.FLAGS)


            # Return to previous level to receive next subgoal if applicable
            # if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or attempts_made >= self.time_limit:

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True
                    # print("Layer %d Out of Attempts" % self.layer_number)


                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
                    return goal_status, max_lay_achieved



    # Update actor and critic networks
    def learn(self, env, agent, num_updates, metrics):

        # To use target networks comment for loop above and uncomment for loop below
        for j in range(num_updates):
            # Update weights of non-target networks
            if self.replay_buffer.size >= 250:
                idx, (old_states, actions, rewards, new_states, goals, is_terminals, oracle_actions, images), is_weights = self.replay_buffer.get_batch()
                if self.relative_subgoals:
                    new_goals = []
                    new_goals = goals + project_state(env, self.FLAGS, self.layer_number, old_states) - project_state(env, self.FLAGS, self.layer_number, new_states)
                else:
                    new_goals = goals

                next_batch_size = min(self.replay_buffer.size, self.replay_buffer.batch_size)

                next_action, next_entropy = self.actor.get_target_action(new_states,new_goals, images)
                errors = self.critic.update(old_states, actions, rewards, new_states, goals, new_goals, next_action, is_terminals, is_weights, next_entropy, images, metrics, total_steps_taken=agent.total_steps_taken)
                self.replay_buffer.batch_update(idx, errors)

                action_derivs = self.critic.get_gradients_for_actions(old_states, goals, self.actor.get_action(old_states, goals, images, symbolic=True), images)
                goal_derivs = None

                if (not self.FLAGS.td3) or (j % 2 == 0):
                    if self.sl_oracle or self.semi_oracle:
                        self.actor.update(old_states, goals, action_derivs, next_batch_size, oracle_actions, metrics, goal_derivs)
                    else:
                        self.actor.update(old_states, goals, action_derivs, next_batch_size, metrics, goal_derivs)

            # Update weights of target networks
            if not self.FLAGS.no_target_net:
                self.critic.update_target_weights()
                self.actor.update_target_weights()
