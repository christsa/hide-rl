#import tensorflow as tf
import numpy as np
from utils import layer


class SacActor():

    def __init__(self,
            sess,
            env,
            batch_size,
            layer_number,
            FLAGS,
            learning_rate=0.001,
            tau=0.05):

        self.sess = sess
        self.actor_grads = FLAGS.actor_grads and layer_number > 0

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = env.action_bounds
            self.action_offset = env.action_offset
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim
        self.target_entropy = -self.action_space_size

        self.actor_name = 'actor_' + str(layer_number)

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1 or (layer_number == FLAGS.layers -2 and FLAGS.oracle):
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.state_dim = env.state_dim

        self.learning_rate = learning_rate
        # self.exploration_policies = exploration_policies
        self.tau = tau
        # self.batch_size = batch_size
        self.batch_size = tf.placeholder(tf.float32)

        self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))
        if FLAGS.mask_global_info and layer_number == 1:
            self.features_ph = tf.concat([self.goal_ph], axis=1)
        elif FLAGS.mask_global_info and layer_number == 0:
            self.features_ph = tf.concat([self.state_ph[:, 2:], self.goal_ph], axis=1)
        else:
            self.features_ph = tf.concat([self.state_ph, self.goal_ph], axis=1)

        # Create actor network
        self.infer_det, self.infer_act, self.infer_pi = self.create_nn(self.features_ph)

        # Target network code "repurposed" from Patrick Emani :^)
        self.weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name]
        # self.num_weights = len(self.weights)

        # Create target actor network
        # self.target_det, self.target_act, self.target_pi = self.create_nn(self.features_ph, name = self.actor_name + '_target')
        # self.target_weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name][len(self.weights):]

        self.update_target_weights = tf.no_op() # \
	    # [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
        #                                           tf.multiply(self.target_weights[i], 1. - self.tau))
        #             for i in range(len(self.target_weights))]

        self.action_derivs = tf.placeholder(tf.float32, shape=(None, self.action_space_size))

        if self.actor_grads:
            self.goal_derivs = tf.placeholder(tf.float32, shape=(None, env.subgoal_dim))
            gradients_from_subgoal = list(map(lambda x: tf.scalar_mul(1., x), tf.gradients(self.infer, self.weights, -self.goal_derivs)))
            gradients_from_actions = tf.gradients(self.infer, self.weights, -self.action_derivs)
            assert len(gradients_from_actions) == len(gradients_from_subgoal)
            self.unnormalized_actor_gradients = list(map(lambda x: x[0]+x[1], zip(gradients_from_actions, gradients_from_subgoal)))
        else:
            self.unnormalized_actor_gradients = tf.gradients(self.infer_act, self.weights, -self.action_derivs)

        self.policy_gradient = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        self.alpha = tf.get_variable(self.actor_name + "/alpha", shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
        self.alpha_loss = -tf.reduce_mean(self.alpha * (self.infer_pi + self.target_entropy))
        self.alpha_train = tf.train.AdamOptimizer(learning_rate).minimize(self.alpha_loss, var_list=[self.alpha])

        self.entropy_target = tf.exp(self.alpha) * self.infer_pi
        entropy_grads = list(map(lambda x: tf.div(x, self.batch_size), tf.gradients(tf.exp(self.alpha) * self.infer_pi, self.weights)))
        self.grads = list(map(lambda x: x[0]+x[1], zip(self.policy_gradient, entropy_grads)))

        # self.policy_gradient = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.train = tf.group([tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.grads, self.weights)), self.alpha_train])


    def get_action(self, state, goal, noise=True):
        actions = self.sess.run(self.infer_act if noise else self.infer_det,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })

        return actions

    def get_target_action(self, state, goal):
        actions, entropy_target = self.sess.run([self.infer_act, self.entropy_target],
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })

        return actions, entropy_target

    def update(self, state, goal, action_derivs, next_batch_size, metrics, goal_derivs=None):
        feed_dict={
                        self.state_ph: state,
                        self.goal_ph: goal,
                        self.action_derivs: action_derivs,
                        self.batch_size: next_batch_size
                    }
        if self.actor_grads:
            assert goal_derivs is not None
            feed_dict[self.goal_derivs] = goal_derivs
        weights, policy_grad, _, alpha = self.sess.run([self.weights, self.policy_gradient, self.train, self.alpha], feed_dict= feed_dict)
        metrics[self.actor_name+"/alpha"] = np.exp(alpha)

        return len(weights)

    # def create_nn(self, state, goal, name='actor'):
    def create_nn(self, features, name=None):
        def clip_but_pass_grad(x, l=1., u=1.):
            clip_up = tf.cast((x > u), tf.float32)
            clip_low = tf.cast(x < l, tf.float32)
            return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)
        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            shift, log_scale = tf.split(layer(fc3, 2*self.action_space_size, is_output=True), 2, axis=-1)
            log_scale = tf.clip_by_value(log_scale, -20., 2)
        distribution = tf.distributions.Normal(shift, tf.exp(log_scale))
        raw_actions = distribution.sample()

        log_likelihood = tf.reduce_sum(distribution.log_prob(raw_actions), axis=1)
        actions = tf.tanh(raw_actions)
        det_actions = tf.tanh(shift)
        log_likelihood -= tf.reduce_sum(tf.log(clip_but_pass_grad(1-actions ** 2 + 1e-6)), axis=1)

        return det_actions, actions, log_likelihood
