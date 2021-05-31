import numpy as np
from experience_buffer import ExperienceBuffer, PrioritizedReplayBuffer
import torch
from collections import defaultdict
from utils import oracle_action
from copy import deepcopy

from utils import project_state, render_image_for_video

class Layer():
    def __init__(self, layer_number, FLAGS, env, device, agent_params):
        self.layer_number = layer_number
        self.FLAGS = FLAGS
        self.device = device
        self.last_layer = self.layer_number == self.FLAGS.layers-1
        self.relative_subgoals = self.FLAGS.relative_subgoals and (self.layer_number < self.FLAGS.layers-1)

        # Set time limit for each layer.  If agent uses only 1 layer, time limit is the max number of low-level actions allowed in the episode (i.e, env.max_actions).
        if FLAGS.layers > 1:
            self.time_limit = FLAGS.time_scale
        else:
            self.time_limit = env.max_actions

        # if self.layer_number == 0 and self.FLAGS.no_middle_level:
        #     self.time_limit = 80

        self.current_state = None
        self.current_image = None
        self.current_goal_image = None
        self.goal = None

        # Initialize Replay Buffer.  Below variables determine size of replay buffer.

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**7

        # Number of full episodes stored in replay buffer
        self.episodes_to_store = agent_params["episodes_to_store"]

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = 2

        self.attempts_made = 0

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit/3)

        # Buffer size = transitions per attempt * # attempts per episode * num of episodes stored
        self.buffer_size = min(self.trans_per_attempt * self.time_limit**(self.FLAGS.layers-1 - self.layer_number) * self.episodes_to_store, self.buffer_size_ceiling)

        self.batch_size = 1024
        if not FLAGS.test:
            buffer_class = ExperienceBuffer
            self.replay_buffer = buffer_class(self.buffer_size, self.batch_size, device=self.device, FLAGS=FLAGS, env=env, layer_number=self.layer_number)

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        # Initialize actor and critic networks
        if self.last_layer and self.FLAGS.vpn:
            if self.FLAGS.vpn_dqn:
                from vpn_dqn_actor import Actor
                from vpn_dqn_critic import Critic
            else:
                from vpn_actor import Actor
                from vpn_critic import Critic
            self.vpn = True
        else:
            from torch_actor import Actor
            from torch_critic import Critic
            self.vpn = False
        self.critic = Critic(self.device, env, self.layer_number, FLAGS)
        self.actor = Actor(device, env, self.batch_size, self.layer_number, FLAGS, self.critic.vpn if self.vpn else None)

        # Stores metrics for later aggregation
        self.agg_metrics = defaultdict(list)

        # Parameter determines degree of noise added to actions during training
        # self.noise_perc = noise_perc
        if self.layer_number == 0:
            self.noise_perc = self.to_torch(agent_params["atomic_noise"])
        elif self.last_layer and self.FLAGS.vpn:
            self.noise_perc = self.to_torch(agent_params["vpn_noise"])
        else:
            self.noise_perc = self.to_torch(agent_params["subgoal_noise"])

        if self.layer_number == 0:
            self.action_bounds = self.to_torch(env.action_bounds)
            self.action_offset = self.to_torch(env.action_offset)
        else:
            self.action_bounds = self.to_torch(env.subgoal_bounds_symmetric)
            self.action_offset = self.to_torch(env.subgoal_bounds_offset)

        # Create flag to indicate when layer has ran out of attempts to achieve goal.  This will be important for subgoal testing
        self.maxed_out = False

        self.subgoal_penalty = agent_params["subgoal_penalty"]*2 if (FLAGS.high_penalty and self.last_layer) else agent_params["subgoal_penalty"]
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]
        if self.last_layer and (self.FLAGS.always_penalize or self.FLAGS.Q_penalize):
            self.subgoal_test_perc = 0.0

    def to_torch(self, value):
        return torch.tensor(value, dtype=torch.float32, device=self.device)

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

    # Add noise to provided action
    def add_target_noise(self, actions, env):
        if not self.FLAGS.td3:
            return actions

        # Noise added will be percentage of range
        if self.layer_number == 0:
            action_bounds = env.action_bounds
            action_offset = env.action_offset
        else:
            action_bounds = env.subgoal_bounds_symmetric
            action_offset = env.subgoal_bounds_offset

        for action in actions:
            assert len(action) == len(action_bounds), ("Action bounds must have same dimension as action", self.layer_number, len(action), len(action_bounds))
            assert len(action) == len(self.noise_perc), "Noise percentage vector must have same dimension as action"

            # Add noise to action and ensure remains within bounds
            for i in range(len(action)):
                if self.layer_number == 0:
                    noise_std = self.noise_perc[i] * action_bounds[i]
                    noise = np.clip(np.random.normal(0, 2*noise_std), -5*noise_std, 5*noise_std)
                else:
                    noise_std = self.noise_perc[i] * action_bounds[i]
                    noise = np.clip(np.random.normal(0, noise_std), -.5, .5)
                action[i] += noise
                action[i] = max(min(action[i], action_bounds[i]+action_offset[i]), -action_bounds[i]+action_offset[i])

        return actions

    # Select random action
    def get_random_action(self, env):
        return torch.rand(len(self.action_bounds), dtype=torch.float32, device=self.device) * 2 * (self.action_bounds) - self.action_bounds + self.action_offset

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
                action = self.add_noise(action, env)

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


    # Create action replay transition by evaluating hindsight action given original goal
    def perform_action_replay(self, hindsight_action, next_state, goal_status):

        # Determine reward (0 if goal achieved, -1 otherwise) and finished boolean.  The finished boolean is used for determining the target for Q-value updates
        if goal_status[self.layer_number]:
            reward = 0
            finished = True
        else:
            reward = -1
            finished = False

        # Transition will take the form [old state, hindsight_action, reward, next_state, goal, terminate boolean, None]
        transition = [self.current_state, hindsight_action, reward, next_state, self.goal, finished, None, self.current_goal_image]

        # Add action replay transition to layer's replay buffer
        self.replay_buffer.add(self.copy_transition(transition))


    # Create initial goal replay transitions
    def create_prelim_goal_replay_trans(self, hindsight_action, next_state, env, total_layers):

        # Create transition evaluating hindsight action for some goal to be determined in future.  Goal will be ultimately be selected from states layer has traversed through.  Transition will be in the form [old state, hindsight action, reward = None, next state, goal = None, finished = None, next state projeted to subgoal/end goal space]

        if self.layer_number == total_layers - 1 or (self.layer_number == total_layers -2 and self.FLAGS.oracle):
            hindsight_goal = env.project_state_to_end_goal(env.sim, next_state)
        else:
            hindsight_goal = env.project_state_to_subgoal(env.sim, next_state)

        # state, action, reward, next_state, goal, terminal, global_hindsight_goal
        transition = [self.current_state, hindsight_action, None, next_state, None, None, hindsight_goal, None, self.current_image]

        self.temp_goal_replay_storage.append(self.copy_transition(transition))

    # Return reward given provided goal and goal achieved in hindsight
    def get_reward(self, new_global_goal, global_hindsight_goal, goal_thresholds):
        assert len(new_global_goal) == len(global_hindsight_goal) == len(goal_thresholds), "Goal, hindsight goal, and goal thresholds do not have same dimensions"
        # If the difference in any dimension is greater than threshold, goal not achieved
        if (torch.abs(new_global_goal-global_hindsight_goal) > goal_thresholds).any():
            return -1
        # Else goal is achieved
        return 0

    # Finalize goal replay by filling in goal, reward, and finished boolean for the preliminary goal replay transitions created before
    def finalize_goal_replay(self, env, goal_thresholds):

        # Choose transitions to serve as goals during goal replay.  The last transition will always be used
        num_trans = len(self.temp_goal_replay_storage)
        if num_trans == 0:
            return

        # If fewer transitions that ordinary number of replay goals, lower number of replay goals
        num_replay_goals = min(self.num_replay_goals, num_trans)


        if self.FLAGS.all_trans or self.FLAGS.HER:
            print("\n\nPerforming Goal Replay for Level %d\n\n" % self.layer_number)
            print("Num Trans: ", num_trans, ", Num Replay Goals: ", num_replay_goals)


        # For each selected transition, update the goal dimension of the selected transition and all prior transitions by using the next state of the selected transition as the new goal.  Given new goal, update the reward and finished boolean as well.
        for index in range(num_trans):
            # trans_copy = np.copy(self.temp_goal_replay_storage)

            # if self.FLAGS.all_trans or self.FLAGS.HER:
                # print("GR Iteration: %d, Index %d" % (i, indices[i]))

            # new_goal = trans_copy[int(indices[i])][6]
            # for index in range(int(indices[i])+1):
            for i in range(num_replay_goals):
                if i == num_replay_goals -1:
                    future_index = num_trans-1
                else:
                    future_index = np.random.randint(index, num_trans)
                new_global_goal = torch.clone(self.temp_goal_replay_storage[future_index][6])
                trans_copy = [None if item is None else torch.clone(item) for item in self.temp_goal_replay_storage[index]]

                # Update goal to new goal
                if self.last_layer and self.FLAGS.vpn:
                    trans_copy[8] = torch.stack([trans_copy[8], env.pos_image(new_global_goal, trans_copy[8])], dim=0)
                if self.relative_subgoals:
                    state_pos = project_state(env, self.FLAGS, self.layer_number, trans_copy[0])
                    trans_copy[4] = (new_global_goal - state_pos)
                else:
                    trans_copy[4] = new_global_goal

                # Update reward
                trans_copy[2] = self.get_reward(new_global_goal, trans_copy[6], goal_thresholds)

                # Update finished boolean based on reward
                if trans_copy[2] == 0:
                    trans_copy[5] = True
                else:
                    trans_copy[5] = False

                # Add finished transition to replay buffer
                if self.FLAGS.all_trans or self.FLAGS.HER:
                    print("\nNew Goal: ", new_global_goal)
                    print("Upd Trans %d: " % index, trans_copy)

                self.replay_buffer.add(trans_copy)


        # Clear storage for preliminary goal replay transitions at end of goal replay
        self.temp_goal_replay_storage = []


    # Create transition penalizing subgoal if necessary.  The target Q-value when this transition is used will ignore next state as the finished boolena = True.  Change the finished boolean to False, if you would like the subgoal penalty to depend on the next state.
    def penalize_subgoal(self, subgoal, next_state, high_level_goal_achieved):

        transition = [self.current_state, subgoal, self.subgoal_penalty, next_state, self.goal, True, None, self.current_goal_image]

        self.replay_buffer.add(self.copy_transition(transition))



    # Determine whether layer is finished training
    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached, (ii) maxed out episode time steps (env.max_actions), (iii) not testing and layer is out of attempts, and (iv) testing, layer is not the highest level, and layer is out of attempts.  NOTE: during testing, highest level will continue to ouput subgoals until either (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.

        # Return to previous level when any higher level goal achieved.  NOTE: if not testing and agent achieves end goal, training will continue until out of time (i.e., out of time steps or highest level runs out of attempts).  This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
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


    # Learn to achieve goals with actions belonging to appropriate time scale.  "goal_array" contains the goal states for the current layer and all higher layers
    def train(self, agent, env, metrics, subgoal_test = False, episode_num = None):

        # print("\nTraining Layer %d" % self.layer_number)

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number].clone()
        self.current_state = agent.current_state
        if self.last_layer and self.FLAGS.vpn:
            self.current_image = self.to_torch(env.take_snapshot())
            self.current_goal_image = torch.stack([self.current_image, env.pos_image(self.goal, self.current_image)], dim=0)

        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        if self.layer_number == 0 and (agent.FLAGS.show or agent.FLAGS.save_video) and agent.FLAGS.layers > 1:
            env.display_subgoals([arr.cpu().numpy() for arr in agent.goal_array], agent.FLAGS)
            # env.sim.data.mocap_pos[3] = env.project_state_to_end_goal(env.sim,self.current_state)
            # print("Subgoal Pos: ", env.sim.data.mocap_pos[1])

        # Current layer has self.time_limit attempts to each its goal state.
        self.attempts_made = 0


        while True:

            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, next_subgoal_test = self.choose_action(agent, env, subgoal_test)

            if self.layer_number > 0:
                subgoal_distance = torch.norm(action[:2] if self.relative_subgoals else (action[:2] - self.current_state[:2])).item()
                self.agg_metrics['subgoal_distances'].append(subgoal_distance)
                goal_subgoal_distance = torch.norm((self.goal[:2] - self.current_state[:2] - action[:2]) if self.relative_subgoals else (self.goal[:2] - action[:2])).item()
                self.agg_metrics['goal_subgoal_distance'].append(goal_subgoal_distance)
                lower_layer = agent.layers[self.layer_number-1]
                lower_action = lower_layer.actor.get_action(self.current_state.unsqueeze(0), action.unsqueeze(0), None)[0]
                lower_Q = lower_layer.critic.get_Q_value(self.current_state.unsqueeze(0), action.unsqueeze(0), lower_action.unsqueeze(0), None).item()
                self.agg_metrics['lower_Q_val'].append(lower_Q)

            # If next layer is not bottom level, propose subgoal for next layer to achieve and determine whether that subgoal should be tested
            if self.layer_number > 0:
                action_copy = action.clone()
                if self.FLAGS.relative_subgoals and self.last_layer:
                    action_copy = action.clone()
                    action_copy -= project_state(env, self.FLAGS, self.layer_number, self.current_state)
                agent.goal_array[self.layer_number - 1] = action_copy

                goal_status, max_lay_achieved = agent.layers[self.layer_number - 1].train(agent, env, metrics, next_subgoal_test, episode_num)

            # If layer is bottom level, execute low-level action
            else:
                next_state = self.to_torch(env.execute_action(action.cpu().numpy()))
                if self.FLAGS.save_video:
                    real_image = render_image_for_video(env, self.FLAGS, agent, next_state)
                    agent.image_path.append(real_image)

                # Increment steps taken
                agent.steps_taken += 1
                if not self.FLAGS.test:
                    agent.total_steps_taken += 1
                # print("Num Actions Taken: ", agent.steps_taken)

                if agent.steps_taken >= env.max_actions:
                    print("Out of actions (Steps: %d)" % agent.steps_taken)

                agent.current_state = next_state

                if self.FLAGS.relative_subgoals:
                    for i_layer in range(self.FLAGS.layers - 1):
                        old_pos = project_state(env, self.FLAGS, i_layer, self.current_state)
                        new_pos = project_state(env, self.FLAGS, i_layer, agent.current_state)
                        agent.goal_array[i_layer] = agent.goal_array[i_layer] + old_pos - new_pos

                # Determine whether any of the goals from any layer was achieved and, if applicable, the highest layer whose goal was achieved
                goal_status, max_lay_achieved = agent.check_goals(env)

            self.attempts_made += 1

            # Perform hindsight learning using action actually executed (low-level action or hindsight subgoal)
            if self.layer_number == 0:
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as hindsight action
                if goal_status[self.layer_number-1]:
                    hindsight_action = action
                # Otherwise, use subgoal that was achieved in hindsight
                else:
                    if self.relative_subgoals:
                        hindsight_action = torch.clone(env.project_state_to_subgoal(env.sim, agent.current_state) - env.project_state_to_subgoal(env.sim, self.current_state))
                    else:
                        hindsight_action = torch.clone(env.project_state_to_subgoal(env.sim, agent.current_state))


            # Next, create hindsight transitions if not testing and env still healthy
            if not agent.FLAGS.test:
                # Create action replay transition by evaluating hindsight action given current goal
                self.perform_action_replay(hindsight_action, agent.current_state, goal_status)

                # Create preliminary goal replay transitions.  The goal and reward in these transitions will be finalized when this layer has run out of attempts or the goal has been achieved.
                self.create_prelim_goal_replay_trans(hindsight_action, agent.current_state, env, agent.FLAGS.layers)


                # Penalize subgoals if subgoal testing and subgoal was missed by lower layers after maximum number of attempts
                if self.layer_number > 0 and next_subgoal_test and agent.layers[self.layer_number-1].maxed_out:
                    self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])
                # Penalize subgoals for the highest level if always penalization on and the lower layers ran out of attempts. 
                elif self.last_layer and self.FLAGS.always_penalize and agent.layers[self.layer_number-1].maxed_out:
                    self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])
                # Penalize subgoals if the lower level think the goal is reachable, but it couldn't reach it. Probably a wall.
                elif self.last_layer and self.FLAGS.Q_penalize:
                    lower_layer = agent.layers[self.layer_number-1]
                    action_copy = action.clone()
                    if self.FLAGS.relative_subgoals:
                        action_copy -= project_state(env, self.FLAGS, self.layer_number, self.current_state)
                    lower_action,_ = lower_layer.actor.get_target_action(self.current_state.unsqueeze(0), action_copy.unsqueeze(0), None)
                    lower_Q_val = lower_layer.critic.get_target_Q_value(self.current_state.unsqueeze(0), action_copy.unsqueeze(0), lower_action, None).item()
                    if lower_Q_val >= -self.FLAGS.time_scale+2 and agent.layers[self.layer_number-1].maxed_out:
                        self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])
            elif not agent.FLAGS.test and self.layer_number == 0:
                self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])

            # Update state of current layer
            self.current_state = agent.current_state
            if self.relative_subgoals:
                self.goal = agent.goal_array[self.layer_number].clone()
                if self.layer_number == 0 and (agent.FLAGS.show or agent.FLAGS.save_video) and agent.FLAGS.layers > 1:
                    env.display_subgoals([arr.cpu().numpy() for arr in agent.goal_array], agent.FLAGS)
            if self.last_layer and self.FLAGS.vpn:
                self.current_image = self.to_torch(env.take_snapshot())
                self.current_goal_image = torch.stack([self.current_image, env.pos_image(self.goal, self.current_image)], dim=0)

            # Return to previous level to receive next subgoal if applicable
            # if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or self.attempts_made >= self.time_limit:

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if self.attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True
                    # print("Layer %d Out of Attempts" % self.layer_number)

                # If not testing, finish goal replay by filling in missing goal and reward values before returning to prior level.
                if not agent.FLAGS.test:
                    if self.layer_number == agent.FLAGS.layers - 1 or (self.layer_number == agent.FLAGS.layers -2 and self.FLAGS.oracle):
                        goal_thresholds = self.to_torch(env.end_goal_thresholds)
                    else:
                        goal_thresholds = self.to_torch(env.subgoal_thresholds)

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(max_lay_achieved, agent, env, self.attempts_made):
                    if self.layer_number == agent.FLAGS.layers-1 and agent.FLAGS.test:
                        print("HL Attempts Made: ", self.attempts_made)
                    return goal_status, max_lay_achieved



    # Update actor and critic networks
    def learn(self, env, agent, num_updates, metrics):

        # To use target networks comment for loop above and uncomment for loop below
        for j in range(num_updates):
            # Update weights of non-target networks
            if self.replay_buffer.size >= 250:
                idx, (old_states, actions, rewards, new_states, goals, is_terminals, images), is_weights = self.replay_buffer.get_batch()
                if self.relative_subgoals:
                    new_goals = []
                    new_goals = goals + project_state(env, self.FLAGS, self.layer_number, old_states) - project_state(env, self.FLAGS, self.layer_number, new_states)
                else:
                    new_goals = goals

                next_batch_size = min(self.replay_buffer.size, self.replay_buffer.batch_size)

                next_action, next_entropy = self.actor.get_target_action(new_states,new_goals, images)
                errors = self.critic.update(old_states, actions, rewards, new_states, goals, new_goals, next_action, is_terminals, is_weights, next_entropy, images, metrics, total_steps_taken=agent.total_steps_taken)
                self.replay_buffer.batch_update(idx, errors)

                action_derivs = self.critic.get_gradients_for_actions(old_states, goals, self.actor, images)
                if self.layer_number > 0:
                    lower_critic = agent.layers[self.layer_number-1].critic
                    lower_actor = agent.layers[self.layer_number-1].actor
                    subgoals = self.actor.get_target_action(old_states, goals, images)[0]
                    if self.FLAGS.relative_subgoals and self.last_layer:
                        assert len(subgoals) == len(old_states)
                        subgoals = subgoals - project_state(env, self.FLAGS, self.layer_number, old_states)
                    Q_val_lower = lower_critic.get_target_Q_value(old_states, subgoals, lower_actor.get_target_action(old_states, subgoals, None)[0], None)
                    metrics['buffer/Q_val_lower%d' % self.layer_number] = torch.mean(Q_val_lower).item()
                    metrics['buffer/Q_val_lower_clipped%d' % self.layer_number] = torch.mean((Q_val_lower < -self.FLAGS.time_scale+1e-6).float()).item()
                    metrics['buffer/Q_val_lower_too_low%d' % self.layer_number] = torch.mean((Q_val_lower > -1.5).float()).item()

                self.actor.update(old_states, goals, action_derivs, next_batch_size, metrics)

            # Update weights of target networks
            if not self.FLAGS.no_target_net:
                self.critic.update_target_weights()
                self.actor.update_target_weights()