#import tensorflow as tf
import numpy as np
from utils import layer


class SLActor():

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
        self.semi_oracle = FLAGS.semi_oracle

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
        if FLAGS.mask_middle and layer_number == 1:
            self.features_ph = tf.concat([self.goal_ph], axis=1)
        else:
            self.features_ph = tf.concat([self.state_ph, self.goal_ph], axis=1)

        # Create actor network
        self.infer = self.create_nn(self.features_ph)

        # Target network code "repurposed" from Patrick Emani :^)
        self.weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name]
        # self.num_weights = len(self.weights)

        # Create target actor network
        if FLAGS.no_target_net:
            self.target = self.infer
        else:
            self.target = self.create_nn(self.features_ph, name = self.actor_name + '_target')
            self.target_weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name][len(self.weights):]

            self.update_target_weights = \
            [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                    tf.multiply(self.target_weights[i], 1. - self.tau))
                        for i in range(len(self.target_weights))]

        self.action_derivs = tf.placeholder(tf.float32, shape=(None, self.action_space_size))

        if self.actor_grads:
            self.goal_derivs = tf.placeholder(tf.float32, shape=(None, env.subgoal_dim))
            gradients_from_subgoal = list(map(lambda x: tf.scalar_mul(1., x), tf.gradients(self.infer, self.weights, -self.goal_derivs)))
            gradients_from_actions = tf.gradients(self.infer, self.weights, -self.action_derivs)
            assert len(gradients_from_actions) == len(gradients_from_subgoal)
            self.unnormalized_actor_gradients = list(map(lambda x: x[0]+x[1], zip(gradients_from_actions, gradients_from_subgoal)))
        else:
            self.unnormalized_actor_gradients = tf.gradients(self.infer, self.weights, -self.action_derivs)

        self.action_label_ph = tf.placeholder(tf.float32, shape=(None, self.action_space_size))
        
        if FLAGS.semi_oracle:
            self.mask_label_ph = tf.placeholder(tf.float32, shape=(None,))
            self.sl_loss = tf.losses.mean_squared_error(labels=self.action_label_ph, predictions=self.infer, weights=tf.expand_dims(self.mask_label_ph, axis=-1))
            self.policy_gradient = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
            self.policy_gradient = list(map(lambda x: x[0]+x[1], zip(self.policy_gradient, tf.gradients(self.sl_loss, self.weights))))
            self.train = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.policy_gradient, self.weights))
        else:
            self.sl_loss = tf.losses.mean_squared_error(labels=self.action_label_ph, predictions=self.infer, weights=self.mask_label_ph)
            self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.sl_loss) # .apply_gradients(zip(self.policy_gradient, self.weights))


    def get_action(self, state, goal, noise=False):
        actions = self.sess.run(self.infer,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })

        return actions

    def get_target_action(self, state, goal):
        actions = self.sess.run(self.target,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })

        return actions, None

    def update(self, state, goal, action_derivs, next_batch_size, action_labels, metrics, goal_derivs=None):
        feed_dict =  {
                self.state_ph: state,
                self.goal_ph: goal,
                self.action_derivs: action_derivs,
                self.batch_size: next_batch_size,
                self.action_label_ph: action_labels
            }
        
        if self.actor_grads:
            assert goal_derivs is not None
            feed_dict[self.goal_derivs] = goal_derivs
        if self.semi_oracle:
            masks = np.array([0.0 if label is None else 1.0 for label in action_labels], dtype=np.float32)
            metrics[self.actor_name+"/mask_percentage"] = np.mean(masks)
            feed_dict[self.mask_label_ph] = masks
            feed_dict[self.action_label_ph] = [np.zeros(self.action_space_size) if label is None else label for label in action_labels]

        weights, sl_loss, _ = self.sess.run([self.weights, self.sl_loss, self.train], feed_dict=feed_dict)
                    
        metrics[self.actor_name+"/sl_loss"] = sl_loss
        return len(weights)

        # self.sess.run(self.update_target_weights)

    # def create_nn(self, state, goal, name='actor'):
    def create_nn(self, features, name=None):

        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset

        return output
