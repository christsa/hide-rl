#import tensorflow as tf
import numpy as np
from utils import layer

class BayesianCritic():

    def __init__(self, sess, env, layer_number, FLAGS, learning_rate=0.001, gamma=0.98, tau=0.05, mc_samples=50, dropout_tau=0.85, dropout_rate=0.01, alpha=0.5):
        self.sess = sess
        self.critic_name = 'critic_' + str(layer_number) + "_bayes"
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        self.mc_samples = mc_samples
        self.dropout_tau = dropout_tau
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        assert FLAGS.num_Qs == 1
       
        self.negative_distance = FLAGS.negative_distance
        if not self.negative_distance:
            self.q_limit = -FLAGS.time_scale
            # Set parameters to give critic optimistic initialization near q_init
            self.q_init = -0.067
            self.q_offset = -np.log(self.q_limit/self.q_init - 1)
        self.no_target_net = FLAGS.no_target_net

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1 or (layer_number == FLAGS.layers -2 and FLAGS.oracle):
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim
        self.state_ph = tf.placeholder(tf.float32, shape=(None, env.state_dim), name='state_ph')
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        self.action_ph = tf.placeholder(tf.float32, shape=(None, action_dim), name='action_ph')
        
        if FLAGS.mask_global_info and layer_number == 1:
            self.features_ph = tf.concat([self.goal_ph, self.action_ph], axis=1)
        elif FLAGS.mask_global_info and layer_number == 0:
            self.features_ph = tf.concat([self.state_ph[:, 2:], self.goal_ph, self.action_ph], axis=1)
        else:
            self.features_ph = tf.concat([self.state_ph, self.goal_ph, self.action_ph], axis=1)

        # Create critic network graph
        prefix = self.critic_name
        self.infer, self.infer_avg, self.infer_mc = self.create_nn(self.features_ph, num_Qs=FLAGS.num_Qs, name=prefix)
        self.weights = [v for v in tf.trainable_variables() if prefix in v.op.name]

        # Create target critic network graph.  Please note that by default the critic networks are not used and updated.  To use critic networks please follow instructions in the "update" method in this file and the "learn" method in the "layer.py" file.

        # Target network code "repurposed" from Patrick Emani :^)
        if self.no_target_net:
            self.target = tf.reduce_min(self.infer, axis=0)
        else:
            target_prefix = self.critic_name + '_target'
            self.target, self.target_avg, self.target_mc = self.create_nn(self.features_ph, num_Qs=FLAGS.num_Qs, name=target_prefix)
            self.target_weights = [v for v in tf.trainable_variables() if target_prefix in v.op.name]

            self.update_target_weights = \
            [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                    tf.multiply(self.target_weights[i], 1. - self.tau))
                        for i in range(len(self.target_weights))]
	
        self.wanted_qs = tf.placeholder(tf.float32, shape=(None, 1))

        if FLAGS.priority_replay:
            self.is_weights = tf.placeholder(tf.float32, shape=(None, 1))
        else:
            self.is_weights = tf.ones_like(self.wanted_qs)

        self.abs_errors = tf.abs(self.wanted_qs - self.infer[0])
        sumsq = (-0.5 * self.alpha * self.dropout_tau) * tf.reduce_sum(self.is_weights * tf.square(self.wanted_qs - self.infer_mc), axis=-1)
        self.loss = (-1.0 * self.alpha ** -1.0) * tf.reduce_logsumexp(sumsq, axis=0, keepdims=True) 
        # self.loss = tf.reduce_mean(tf.square(self.wanted_qs - self.infer[i]))

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.gradient_action = tf.gradients(self.infer_avg, self.action_ph)

        self.gradient_goal = tf.gradients(self.infer_avg, self.goal_ph)


    def get_Q_value(self,state, goal, action):
        return self.sess.run(self.infer,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })[0]

    def get_target_Q_value(self,state, goal, action):
        assert not self.no_target_net
        return self.sess.run(self.target,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })[0]


    def update(self, old_states, old_actions, rewards, new_states, old_goals, new_goals, new_actions, is_terminals, is_weights, metrics):

        # Be default, repo does not use target networks.  To use target networks, comment out "wanted_qs" line directly below and uncomment next "wanted_qs" line.  This will let the Bellman update use Q(next state, action) from target Q network instead of the regular Q network.  Make sure you also make the updates specified in the "learn" method in the "layer.py" file.  
        wanted_qs = self.sess.run(self.infer_avg if self.no_target_net else self.target_avg,
                feed_dict={
                    self.state_ph: new_states,
                    self.goal_ph: new_goals,
                    self.action_ph: new_actions
                })
       
        for i in range(len(wanted_qs)):
            if is_terminals[i]:
                wanted_qs[i] = rewards[i]
            else:
                wanted_qs[i] = rewards[i] + self.gamma * wanted_qs[i][0]

            # Ensure Q target is within bounds [-self.time_limit,0]
            if not self.negative_distance:
                wanted_qs[i] = max(min(wanted_qs[i],0), self.q_limit)
                assert wanted_qs[i] <= 0 and wanted_qs[i] >= self.q_limit, "Q-Value target not within proper bounds"

        feed_dict = {
                    self.state_ph: old_states,
                    self.goal_ph: old_goals,
                    self.action_ph: old_actions,
                    self.wanted_qs: wanted_qs 
                }
        if is_weights is not None:
            feed_dict[self.is_weights] = is_weights
        self.loss_val, _, abs_errors = self.sess.run([self.loss, self.train, self.abs_errors], feed_dict=feed_dict)
        
        metrics[self.critic_name.replace("_bayes", "") + '/Q_loss'] = np.mean(self.loss_val)
        metrics[self.critic_name.replace("_bayes", "") + '/Q_val'] = np.mean(wanted_qs)
        return abs_errors

    def get_gradients_for_actions(self, state, goal, action):
        grads = self.sess.run(self.gradient_action,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })

        return grads[0]

    def get_gradients_for_goals(self, state, goal, action):
        grads = self.sess.run(self.gradient_goal,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })

        return grads[0]
    
    # Function creates the graph for the critic function.  The output uses a sigmoid, which bounds the Q-values to between [-Policy Length, 0].
    def create_nn(self, features, num_Qs=1, name=None):

        def reapply_layer(name, fn, nets, dropout):
            new_nets = []
            for net in nets:
                if name is None:
                    new_net = fn(net)
                    if dropout:
                        new_net = tf.nn.dropout(new_net, rate=self.dropout_rate)
                else:
                    with tf.variable_scope(name, reuse=True):
                        new_net = fn(net)
                        if dropout:
                            new_net = tf.nn.dropout(new_net, rate=self.dropout_rate)
                new_nets.append(new_net)
            return new_nets

        if name is None:
            name = self.critic_name
        
        output = None
        dropout_nets = [features] * self.mc_samples
    
        layer_name = name + '_fc_1'
        with tf.variable_scope(layer_name):
            fc1 = layer(features, 64)
        dropout_nets = reapply_layer(layer_name, lambda x: layer(x, 64), dropout_nets, dropout=True)
        
        layer_name = name + '_fc_2'
        with tf.variable_scope(layer_name):
            fc2 = layer(fc1, 64)
        dropout_nets = reapply_layer(layer_name, lambda x: layer(x, 64), dropout_nets, dropout=True)

        layer_name = name + '_fc_3'
        with tf.variable_scope(layer_name):
            fc3 = layer(fc2, 64)
        dropout_nets = reapply_layer(layer_name, lambda x: layer(x, 64), dropout_nets, dropout=True)

        layer_name = name + '_fc_4'
        with tf.variable_scope(layer_name):
            fc4 = layer(fc3, 1, is_output=True)
        dropout_nets = reapply_layer(layer_name, lambda x: layer(x, 1, is_output=True), dropout_nets, dropout=True)

        if self.negative_distance:
            output = fc4
        else:
            # A q_offset is used to give the critic function an optimistic initialization near 0
            output = tf.sigmoid(fc4 + self.q_offset) * self.q_limit
            dropout_nets = reapply_layer(None, lambda x: tf.sigmoid(x + self.q_offset) * self.q_limit, dropout_nets, dropout=False)

        return output, tf.add_n(dropout_nets) / float(len(dropout_nets)), tf.stack(dropout_nets, axis=0)
