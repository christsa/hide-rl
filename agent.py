import numpy as np
from layer import Layer
from oracle_layer import OracleLayer
import pickle as cpickle
import os, sys
import pickle as cpickle
import csv, json
import datetime, time
import torch
from tensorboardX import SummaryWriter
import torchvision

from utils import project_state, save_video, attention, gaussian_attention, multivariate_gaussian_attention, render_image_for_video

from collections import OrderedDict, defaultdict

# Below class instantiates an agent
class Agent():
    def __init__(self,FLAGS, env, agent_params):

        self.FLAGS = FLAGS
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        if self.FLAGS.no_middle_level:
            self.FLAGS.layers = 2
            self.FLAGS.time_limit = 25

        # Create agent with number of levels specified by user
        highest_layer_class = OracleLayer if self.FLAGS.oracle else Layer
        self.layers = [Layer(0, FLAGS, env, self.device, agent_params)]
        self.layers = self.layers + [Layer(i,FLAGS,env,self.device,agent_params) for i in range(1, FLAGS.layers-1)]
        self.layers.append(highest_layer_class(FLAGS.layers-1, FLAGS, env, self.device, agent_params))

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        self.initialize_networks()

        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for i in range(FLAGS.layers)]

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        self.total_steps_taken = 0

        self.image_path = None

        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = 40

        # Below parameters will be used to store performance results
        self.performance_log = []

        self.other_params = agent_params

        self.end_goal_thresholds = torch.tensor(env.end_goal_thresholds, dtype=torch.float32, device=self.device)
        self.subgoal_thresholds = torch.tensor(env.subgoal_thresholds, dtype=torch.float32, device=self.device)


    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self,env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.FLAGS.layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(None, self.current_state)
        proj_end_goal = env.project_state_to_end_goal(None, self.current_state)

        far_fn_glob = lambda goal, pos, thres: torch.abs(goal - pos) > thres
        far_fn_rel = lambda goal, pos, thres: torch.abs(goal) > thres

        for i in range(self.FLAGS.layers):

            goal_achieved = True
            far_fn = far_fn_rel if (self.layers[i].relative_subgoals) else far_fn_glob

            # If at highest layer, compare to end goal thresholds
            if i == self.FLAGS.layers - 1 or (i == self.FLAGS.layers - 2 and self.FLAGS.oracle):
                # Check dimensions are appropriate
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(self.end_goal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                if far_fn(self.goal_array[i], proj_end_goal, self.end_goal_thresholds).any():
                    goal_achieved = False

            # If not highest layer, compare to subgoal thresholds
            else:
                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(self.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                if far_fn(self.goal_array[i], proj_subgoal, self.subgoal_thresholds).any():
                    goal_achieved = False

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False


        return goal_status, max_lay_achieved

    def datetimestamp(self, divider='-', datetime_divider='T'):
        now = datetime.datetime.now()
        return now.strftime(
            '%Y{d}%m{d}%dT%H{d}%M{d}%S'
            ''.format(d=divider, dtd=datetime_divider))

    def initialize_networks(self):
        # Set up directory for saving models
        self.model_dir = os.path.join(os.getcwd(), 'models', self.FLAGS.exp_name, str(self.FLAGS.exp_num))
        os.makedirs(self.model_dir, exist_ok=True)
        model_working_dir = os.path.join(os.getcwd(), 'models_working')
        model_negdist_dir = os.path.join(os.getcwd(), 'models_negative_distance')
        self.model_loc = os.path.join(self.model_dir, 'HAC.pkl')
        if not self.FLAGS.test:
            self.tb_writter = SummaryWriter(self.model_dir)
        self.performance_path = os.path.join(self.model_dir, "performance_log.txt")
        self.metrics_path = os.path.join(self.model_dir, "progress.csv")
        self.metrics_keys = OrderedDict( {key:None for key in sorted([
            'critic_0/Q_val', 'critic_1/Q_val', 'critic_2/Q_val', 
            'critic_0/Q_loss', 'critic_1/Q_loss', 'critic_2/Q_loss', 
            'vpn_critic_2/Q_val', 'vpn_critic_2/Q_loss',
            'actor_0/alpha', 'actor_1/alpha', 'actor_2/alpha', 
            'actor_2/mask_percentage', 'actor_2/sl_loss', 
            'steps_taken', 'test/success_rate', 'total_steps_taken',
            'sample_time', 'train_time', 'epoch_time',
            'subgoal_distances1', 'subgoal_distances2',
            'goal_subgoal_distance1', 'goal_subgoal_distance2',
            'lower_Q_val1', 'lower_Q_val2',
            'buffer/Q_val_lower_clipped1', 'buffer/Q_val_lower1', 'buffer/Q_val_lower_too_low1',
            'buffer/Q_val_lower_clipped2', 'buffer/Q_val_lower2', 'buffer/Q_val_lower_too_low2',
            'actor_0/loss', 'actor_1/loss','actor_2/loss', 
        ])})
        
        if self.FLAGS.retrain:
            with open(self.metrics_path, 'w+') as f:
                print(','.join(self.metrics_keys.keys()), file=f)

            with open(os.path.join(self.model_dir, "params.json"), 'w+') as f:
                json.dump({
                    'run_id': "%s_%d_%s" % (self.FLAGS.exp_name, self.FLAGS.exp_num, self.datetimestamp()),
                    'run_command': ' '.join(sys.argv),
                    'target_networks': not self.FLAGS.no_target_net, 
                    'num_Qs': self.FLAGS.num_Qs, 
                    'exp_name': self.FLAGS.exp_name,
                    'exp_num': self.FLAGS.exp_num,
                    'oracle': self.FLAGS.oracle, 
                    'variant': self.FLAGS.variant,
                    'relative_subgoals': self.FLAGS.relative_subgoals,
                }, f, indent=4, sort_keys=True)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # If not retraining, restore weights
        # if we are not retraining from scratch, just restore weights
        if self.FLAGS.retrain == False:
            with open(os.path.join(self.model_dir, "params.json"), 'r') as f:
                variant = json.load(f)
            flag_dict = vars(self.FLAGS)
            for variant_key in variant:
                if variant_key in ['run_id', 'run_command', 'target_networks', 'variant', 'ddl']:
                    continue
                assert variant[variant_key] == flag_dict[variant_key], (variant_key, variant[variant_key, flag_dict[variant_key]])
            assert variant['target_networks'] == (not flag_dict['no_target_net'])
            print(self.model_dir)
            self.load_state_dict(torch.load(self.model_loc, self.device))

    def state_dict(self):
        result = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'actor'):
                result['layer_%d_actor' % i] = layer.actor.state_dict()
            if hasattr(layer, 'critic'):
                result['layer_%d_critic' % i] = layer.critic.state_dict()
        return result
    
    def load_state_dict(self, state_dict):
        result = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'actor'):
                layer.actor.load_state_dict(state_dict['layer_%d_actor' % i])
            if hasattr(layer, 'critic'):
                layer.critic.load_state_dict(state_dict['layer_%d_critic' % i])

    # Save neural network parameters
    def save_model(self, episode, success_rate = None):
        if success_rate is not None and success_rate >= 0:
            extra_location = '{}/HAC_{}_{}.pkl'.format(self.model_dir, episode, int(success_rate))
            torch.save(self.state_dict(), extra_location)
        torch.save(self.state_dict(), self.model_loc)

    # Update actor and critic networks for each layer
    def learn(self, env, metrics):
        for i in range(len(self.layers)):
            self.layers[i].learn(env, self, self.num_updates, metrics)
        # self.layers[0].learn(self.num_updates)
        
        metrics['total_steps_taken'] = self.total_steps_taken
        metrics['steps_taken'] = self.steps_taken
        if not self.FLAGS.train_only:
            metrics['test/success_rate'] = self.performance_log[-1]

    # Train agent for an episode
    def train(self, env, episode_num):
        metrics = {}
        epoch_start = time.time()
        # Select initial state from in initial state space, defined in environment.py
        self.current_state = torch.tensor(env.reset_sim(self.goal_array[self.FLAGS.layers - 1]), device=self.device, dtype=torch.float32)
        if "ant" in env.name.lower():
            print("Initial Ant Position: ", self.current_state[:3])
        # print("Initial State: ", self.current_state)

        if self.FLAGS.save_video:
            self.image_path = [env.crop_raw(env.render(mode='rgb_array'))]

        # Select final goal from final goal space, defined in "design_agent_and_env.py"
        self.goal_array[self.FLAGS.layers - 1] = torch.tensor(env.get_next_goal(self.FLAGS.test), dtype=torch.float32, device=self.device)
        print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, max_lay_achieved = self.layers[self.FLAGS.layers-1].train(self,env, metrics, episode_num = episode_num)
        
        sample_end = time.time()
        metrics['sample_time'] = sample_end - epoch_start

        for i_layer in range(self.FLAGS.layers):
            for key, values in self.layers[i_layer].agg_metrics.items():
                metrics[key+str(i_layer)] = np.mean(values)
            self.layers[i_layer].agg_metrics = defaultdict(list)

        metrics['sample_time'] = sample_end - epoch_start

        if self.FLAGS.save_video:
            save_video(self.image_path, os.path.join(self.model_dir, "test_episode_%d.avi"%episode_num))
            del self.image_path[:]

        # Update actor/critic networks if not testing
        print(self.steps_taken)
        if not self.FLAGS.test:
            self.learn(env, metrics)
            epoch_end = time.time()
            metrics['train_time'] = epoch_end - sample_end
            metrics['epoch_time'] = epoch_end - epoch_start
            self.log_metrics(metrics, episode_num, env)

        # Return whether end goal was achieved
        return goal_status[self.FLAGS.layers-1]

    # Save performance evaluations
    def log_performance(self, success_rate):
        # Add latest success_rate to list
        self.performance_log.append(success_rate)
        # Save log
        with open(self.performance_path, "w+") as f:
            print(self.performance_log, file=f)

    def log_metrics(self, metrics, episode_num, env):
        if self.FLAGS.test: return
        
        for key, metric in metrics.items():
            self.tb_writter.add_scalar(key, metric, self.total_steps_taken)
        if self.FLAGS.vpn and episode_num == 1: # Log once for every batch, i.e. every train 100 episodes
            def subtract_channels(tensor, dim):
                grid, pos = tensor.unbind(dim=dim)
                return (grid - pos).unsqueeze(dim)
            vpn = self.layers[self.FLAGS.layers-1].critic.vpn
            sampled_image = self.layers[self.FLAGS.layers-1].current_image.unsqueeze(0)
            sampled_image_with_goal = self.layers[self.FLAGS.layers-1].current_goal_image.unsqueeze(0)
            image_grid = torchvision.utils.make_grid([sampled_image, subtract_channels(sampled_image_with_goal, 1).squeeze(1)])
            self.tb_writter.add_image('sampled_imaged,sampled_imaged_with_goal', image_grid, self.total_steps_taken)

            batch = self.layers[self.FLAGS.layers-1].replay_buffer.get_batch()[1]
            buffer_images = batch[-1][:5]
            buffer_images_pos = torch.stack([env.pos_image(batch[0][i, :2], buffer_images[i,0]) for i in range(5)], dim=0).unsqueeze(1)
            buffer_images_r, buffer_images_p, buffer_images_v = vpn.get_info(buffer_images)
            buffer_images_r, buffer_images_p, buffer_images_v = list(map(lambda img: img.unsqueeze(1), [buffer_images_r, buffer_images_p, buffer_images_v]))
            assert (buffer_images >=0).all() and (buffer_images <= 1).all(), (torch.max(buffer_images), torch.min(buffer_images))
            assert (buffer_images_r >=0).all() and (buffer_images_r <= 1).all()
            assert (buffer_images_p >=0).all() and (buffer_images_p <= 1).all()
            assert (buffer_images_v <=0).all() and (buffer_images_v >= -1).all()
            row = [(buffer_images[:,:1]-buffer_images_pos), buffer_images_r, buffer_images_p, 1+buffer_images_v]
            if self.FLAGS.gaussian_attention:
                image_position = torch.stack(env.get_image_position(batch[0][:5, :2], buffer_images), dim=-1)
                sigma = 3
                if self.FLAGS.learn_sigma:
                    if self.FLAGS.covariance:
                        cov = self.layers[self.FLAGS.layers-1].actor.sigma(buffer_images_v.squeeze(1), batch[0][:5], buffer_images)
                        buffer_images_v_att = multivariate_gaussian_attention(buffer_images_v.squeeze(1), image_position, cov)[0].unsqueeze(1)
                    else:
                        sigma = self.layers[self.FLAGS.layers-1].actor.sigma(buffer_images_v.squeeze(1), batch[0][:5], buffer_images)
                        buffer_images_v_att = gaussian_attention(buffer_images_v.squeeze(1), image_position, sigma)[0].unsqueeze(1)
                buffer_images_v_att = gaussian_attention(buffer_images_v.squeeze(1), image_position, sigma=3)[0].unsqueeze(1)
                row.append(buffer_images_v_att)
            if self.FLAGS.vpn_masking:
                image_position = torch.stack(env.get_image_position(batch[0][:5, :2], buffer_images), dim=-1)
                pos_image = env.pos_image(batch[0][:5, :2], buffer_images[:, 0])
                print("adding to row.")
                row.append(1+vpn.mask_image(buffer_images_v.squeeze(1), buffer_images_p.squeeze(1), pos_image, image_position)[0].unsqueeze(1))

            if self.FLAGS.vpn_dqn:
                buffer_images_actor_probs = self.layers[self.FLAGS.layers-1].actor.get_action(batch[0][:5], None, buffer_images)
                actor_probs = env.pos_image(buffer_images_actor_probs, buffer_images[:,0])
            elif self.FLAGS.gaussian_attention:
                with torch.no_grad():
                    actor_probs = self.layers[self.FLAGS.layers-1].actor.get_action(batch[0][:5], None, buffer_images, symbolic=True)
            else:
                with torch.no_grad():
                    actor_probs = torch.zeros_like(buffer_images_v.squeeze(1))
                    _,x_coords,y_coords = attention(buffer_images_v.squeeze(1), self.layers[self.FLAGS.layers-1].actor.get_image_location(batch[0][:5], buffer_images), 2)
                    buffer_images_actor_probs = self.layers[self.FLAGS.layers-1].actor.get_action(batch[0][:5], None, buffer_images, symbolic=True)
                    for i in range(5):
                        for j in range(5):
                            for k in range(5):
                                actor_probs[i, x_coords[i, j], y_coords[i, k]] = buffer_images_actor_probs[i, j, k]
            assert (actor_probs >= 0).all() and (actor_probs <= 1).all()
            row.append(actor_probs.unsqueeze(1))
            image_grid = torchvision.utils.make_grid(torch.cat(row, dim=-1), nrow=1)
            self.tb_writter.add_image('img,r,p,v,actor_probs', image_grid, self.total_steps_taken)

        keys_extra = set(metrics.keys()) - set(self.metrics_keys)
        if len(keys_extra) > 0:
            print("WARNING, Extra keys found: ", keys_extra)
        if episode_num % 20 == 0:
            # Save metrics
            with open(self.metrics_path, 'a') as f:
                ordered_metrics = [str(metrics.get(key, "")) for key in self.metrics_keys]
                print(','.join(ordered_metrics), file=f)

        
