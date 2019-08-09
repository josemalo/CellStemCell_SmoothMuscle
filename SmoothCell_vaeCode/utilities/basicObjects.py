
"""
basicOjects: Objects required to run a zero-inflated negative binomial Variational Auto-Encoder
"""

import feather
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from copy import deepcopy
from keras import metrics
from math import floor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

module_logger = logging.getLogger('VAE')


def add_name(name, v_name):
    return name + '_' + v_name

class BaseTF(object):
    """
    Activation functions
    """
    def __init__(self, args):
        self.args = args
        # Dict to call correct activation function
        self.layer_fnc = {'tanh': self.tanh_layer, 'relu': self.relu_layer, 'elu': self.elu_layer,
                          'swish': self.swish_layer, 'dense': self.dense_layer,
                          'sigmoid': self.sigmoid_layer, 'gated': self.gated_layer}

    @staticmethod
    def log_init(dim_1, dim_2, act_fnc):
        module_logger.info('Creating Layer...')
        module_logger.info('Layer: ' + repr(act_fnc) + ' Activation - Dim: (' + repr(dim_1) + ',' + repr(dim_2) + ')')

    @staticmethod
    def gated_layer(data, weight, bias, phase, dropout):
        """
        GatedDense Activation Function (Dauphin et al. 2016)

        https://arxiv.org/abs/1612.08083

        Args:
            data: input data matrix
            weight: dict of weight matrix
            bias: dict of bias vector
            phase (boolean): training or testing phase
            dropout: fraction of dropout (1 is none, 0 is all)

        Returns:
            gated: data processed through activation function

        """
        # Fully Connected h
        fc_h = tf.matmul(data, weight.get('h_weight')) + bias.get('h_bias')
        # Fully Connected g
        fc_g = tf.matmul(data, weight.get('g_weight')) + bias.get('g_bias')
        # Sigmoid g
        g = tf.sigmoid(fc_g)
        # Calculate gated activation
        gated = tf.multiply(fc_h, g)
        # Dropout
        gated = tf.contrib.layers.dropout(gated, keep_prob=dropout, is_training=phase)
        return gated

    @staticmethod
    def tanh_layer(data, weight, bias, phase, dropout):
        # Fully Connected
        fc = tf.matmul(data, weight) + bias
        # Batch Norm
        fc = tf.contrib.layers.batch_norm(fc, is_training=phase)
        # TANH
        h = tf.nn.tanh(fc)
        # Dropout
        h = tf.contrib.layers.dropout(h, keep_prob=dropout, is_training=phase)
        return h

    @staticmethod
    def elu_layer(data, weight, bias, phase, dropout):
        # Fully Connected
        fc = tf.matmul(data, weight) + bias
        # Batch Norm
        fc = tf.contrib.layers.batch_norm(fc, is_training=phase)
        # ELU
        h = tf.nn.elu(fc)
        # Dropout
        h = tf.contrib.layers.dropout(h, keep_prob=dropout, is_training=phase)
        return h

    @staticmethod
    def relu_layer(data, weight, bias, phase, dropout):
        # Fully Connected
        fc = tf.matmul(data, weight) + bias
        # Batch Norm
        fc = tf.contrib.layers.batch_norm(fc, scale=True, is_training=phase)
        # RELU
        h = tf.nn.relu(fc)
        # Dropout
        h = tf.contrib.layers.dropout(h, keep_prob=dropout, is_training=phase)
        return h

    @staticmethod
    def swish_layer(data, weight, bias, phase, dropout):
        # Fully Connected
        fc = tf.matmul(data, weight) + bias
        # Batch Norm
        fc = tf.contrib.layers.batch_norm(fc, is_training=phase)
        # SWISH
        h = fc * tf.nn.sigmoid(fc)
        # Dropout
        h = tf.contrib.layers.dropout(h, keep_prob=dropout, is_training=phase)
        return h

    @staticmethod
    def dense_layer(data, weight, bias, phase, dropout):
        # Fully Connected
        fc = tf.matmul(data, weight) + bias
        # Fully Connected
        h = fc
        return h

    @staticmethod
    def sigmoid_layer(data, weight, bias, phase, dropout):
        # Fully Connected
        fc = tf.matmul(data, weight) + bias
        # Sigmoid Activation
        h = tf.nn.sigmoid(fc)
        # Dropout
        h = tf.contrib.layers.dropout(h, keep_prob=dropout, is_training=phase)
        return h

class BaseVAE(BaseTF):
    def __init__(self, args, input_dim, v_name):
        super().__init__(args)
        # Make a new graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Append name for TF Graphs
            self.v_name = v_name
            # Input dim comes from data size
            self.in_dim = input_dim
            # Check the Posterior Distribution
            self.use_zinb = (args.post_layer == 'zinb')
            self.use_gaussian = (args.post_layer == 'gauss')
            self.use_nb = (args.post_layer == 'nb')
            # Posterior Flag
            self.posterior_flag = (not((args.post_layer == 'zinb') or (args.post_layer == 'gauss')
                                       or (args.post_layer == 'nb')))
            # Fnc Dict for Posterior Layer
            self.post_fnc = {'zinb': self.zinb_loss, 'zi_gumbel': self.zi_gumbel_bce_loss,
                             'regular': self.bce_loss, 'gauss': self.gaussian_loss,
                             'nb': self.negative_binomial_loss}
            # Fnc Dict for Reg Layer
            self.reg_fnc = {'kld': self.kld_loss, 'mmd': self.mmd_loss, 'vamp': self.vamp_prior}
            # Data Placeholder
            self.X = tf.placeholder(tf.float32, shape=(None, self.in_dim),
                                    name=add_name('batch_input_data', v_name))
            # Weights Placeholder
            self.weights = tf.placeholder(tf.float32, shape=(None,),
                                          name=add_name('exp_label_weights', v_name))
            # Placeholder for Phase Indication
            self.phase = tf.placeholder(tf.bool, name=add_name('phase', v_name))
            # Keep track of epoch
            self.epoch = tf.placeholder(tf.float32, name=add_name('epoch', v_name))
            # VampPrior Pseudo-Weights -- Default 500 Components
            if self.args.reg == 'vamp':
                with tf.variable_scope(add_name('vamp_prior_', self.v_name)):
                    self.vamp_comp = int(self.in_dim/100)
                    module_logger.info('Number of VAMP Components: ' + repr(self.vamp_comp))
                    self.tf_vamp_comp = tf.Variable(500.0, dtype=tf.float32)
                    self.vamp_inputs = tf.get_variable(
                        'pseudo_inputs', shape=[self.vamp_comp, self.in_dim],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            # Beta for Warm Up
            if self.args.warm_up is not None:
                self.beta = tf.minimum(self.epoch / floor(self.args.warm_up * self.args.epochs), 1.0)
            else:
                self.beta = tf.Variable(1.0)
            # Check for gated activation function -- have to make double weights as dict
            if self.args.act_fnc == 'gated':
                # Generate Weights
                self.encoder_weights = self.gen_gated_encoder_nn_weights()
                # Generate Biases
                self.encoder_biases = self.gen_gated_encoder_nn_bias()
                # Generate Weights
                self.decoder_weights = self.gen_gated_decoder_nn_weights(self.posterior_flag)
                # Generate Biases
                self.decoder_biases = self.gen_gated_decoder_nn_bias(self.posterior_flag)
            else:
                # Generate Weights
                self.encoder_weights = self.gen_encoder_nn_weights()
                # Generate Biases
                self.encoder_biases = self.gen_encoder_nn_bias()
                # Generate Weights
                self.decoder_weights = self.gen_decoder_nn_weights(self.posterior_flag)
                # Generate Biases
                self.decoder_biases = self.gen_decoder_nn_bias(self.posterior_flag)
            # Latent Dictionary
            self.latent_dict = self.gen_latent_weights()
            # Generate ZINB Dictionary
            if self.use_zinb:
                self.zinb_dict = self.gen_zinb_vars()
            # Generate Gaussian Dictionary
            if self.use_gaussian:
                self.gauss_dict = self.gen_gaussian_vars()
            # Generate NB Dictionary
            if self.use_nb:
                self.nb_dict = self.gen_nb_vars()
            # Latent
            self.z_mu, self.z_std = self.encoder_layers()
            # Vamp Model
            if self.args.reg == 'vamp':
                self.z_vamp_mu, self.z_vamp_std = self.vamp_encoder_layers()
            # Sample
            epsilon = tf.random_normal((tf.shape(self.z_mu)[0], self.args.latent_dim), mean=0.,
                                       stddev=1.0)
            # Build Latent
            self.z = self.z_mu + tf.exp(self.z_std / 2) * epsilon
            # Reconstruct from Latent
            if self.use_zinb:
                self.zinb_log_mu, self.zinb_log_theta, self.zinb_logit_pi = self.zinb_decoder_layers()
            elif self.use_gaussian:
                self.gauss_log_mu, self.gauss_log_theta = self.gaussian_decoder_layers()
            elif self.use_nb:
                self.nb_log_mu, self.nb_log_theta = self.nb_decoder_layers()
            else:
                self.X_recon = self.decoder_layers()
            # Build the Loss Function (Recon Loss + Reg Loss)
            self.reg_loss = self.reg_loss_layer()
            self.recon_loss = self.recon_loss_layer()
            self.loss = tf.reduce_mean((self.recon_loss + self.reg_loss) * self.weights)
            # Train OP -- VAE
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Optimizer
                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=self.args.lr).minimize(self.loss)
            # Bump a Session
            self.sess = tf.Session(graph=self.graph)
            # Initializing the tensor flow variables
            self.sess.run(tf.global_variables_initializer())
            # Make a saver object
            self.saver = tf.train.Saver(max_to_keep=1)

    def restore_model(self, full_path):
        module_logger.info('============LOADING BEST MODEL============')
        module_logger.info('Restoring Best Model...')

        print(full_path)
        print(type(self.sess))


        self.saver.restore(self.sess, tf.train.latest_checkpoint(full_path))
        module_logger.info('============BEST MODEL LOADED============')

    def save_model(self, full_path):
        # Model Save
        self.saver.save(self.sess, save_path=(os.path.join(full_path, 'model.chkp')))

    def partial_fit(self, X, weights, phase, epoch):
        _, cost, reg_loss, rec_loss = self.sess.run(
            [self.optimizer, self.loss, self.reg_loss, self.recon_loss], feed_dict={
                self.X: X, self.weights: weights, self.phase: phase, self.epoch: epoch})
        return cost, np.mean(reg_loss), np.mean(rec_loss)

    def get_loss(self, X, weights, phase, epoch):
        cost, beta = self.sess.run(
            [self.loss, self.beta], feed_dict={self.X: X, self.weights: weights,
                                               self.phase: phase, self.epoch: epoch})
        return cost, beta

    def get_ll(self, X, phase, epoch):
        ll = self.sess.run(
            self.recon_loss, feed_dict={self.X: X, self.phase: False, self.epoch: epoch})
        return ll

    def transform(self, X):
        return self.sess.run(self.z_mu, feed_dict={self.X: X, self.phase: False,
                                                   self.epoch: self.args.epochs})

    def pop_layers(self):
        weights, biases, z_mu_weights, z_mu_biases = self.sess.run(
            [self.encoder_weights, self.encoder_biases, self.latent_dict.get('w_z_mean'),
             self.latent_dict.get('b_z_mean')])
        all_weights = weights + [z_mu_weights]
        all_biases = biases + [z_mu_biases]
        return all_weights, all_biases

    def recon_loss_layer(self):
        recon_error = self.post_fnc.get(self.args.post_layer)()
        return recon_error

    def reg_loss_layer(self):
        reg_error = self.reg_fnc.get(self.args.reg)()
        if self.args.warm_up is not None:
            reg_error *= self.beta
        return reg_error

    def zinb_loss(self, eps=1e-8):
        # Revert to Actual Values
        # mu protected against overflow
        mu_pos_neg_mask = tf.cast(tf.less(self.zinb_log_mu, 0.0), tf.float32)
        mu = (tf.multiply(mu_pos_neg_mask, tf.exp(-tf.abs(self.zinb_log_mu))) +
              tf.multiply((1-mu_pos_neg_mask), tf.divide(1, tf.exp(-tf.abs(self.zinb_log_mu))
                                                         + eps)))
        # theta protected against overflow
        theta_pos_neg_mask = tf.cast(tf.less(self.zinb_log_theta, 0.0), tf.float32)
        theta = (tf.multiply(theta_pos_neg_mask, tf.exp(-tf.abs(self.zinb_log_theta))) +
                 tf.multiply((1-theta_pos_neg_mask),
                             tf.divide(1.0, tf.exp(-tf.abs(self.zinb_log_theta)) + eps)))
        # Probability
        pi = tf.divide(1, (1 + tf.exp(-self.zinb_logit_pi)))
        # Check Less Than and Re-cast as float for multiplication
        tf_mask = tf.cast(tf.less(self.X, eps), tf.float32)

        # BASE FORMULATION
        # Negative Binomial X > 0 (Overflow Protected)
        pre_nb = (tf.lgamma(self.X + theta + eps) - tf.lgamma(self.X + 1 + eps) -
                  tf.lgamma(theta + eps) +
                  (theta * (tf.log(theta + eps) - tf.log(theta + mu + eps))) +
                  (self.X * (tf.log(mu + eps) - tf.log(theta + mu + eps))))
        nb_mask = tf.cast(tf.less(pre_nb, 0.0), tf.float32)
        nb = (tf.multiply(nb_mask, tf.exp(-tf.abs(pre_nb))) +
              tf.multiply((1-nb_mask), tf.divide(1.0, tf.exp(-tf.abs(pre_nb)) + eps)))
        # Negative Binomial X = 0
        nb_zero = tf.exp(theta * (tf.log(theta + eps) - tf.log(theta + mu + eps)))

        # Use 0/1 Indicator for Multiplication as sub for Dirac Delta
        zero_side = pi + ((1 - pi) * nb_zero)
        non_zero_side = (1 - pi) * nb
        ll = tf.log((tf.multiply(tf_mask, zero_side) + tf.multiply((1-tf_mask), non_zero_side))
                    + eps)
        return -tf.reduce_sum(ll, 1)

    def negative_binomial_loss(self, eps=1e-8):
        # Revert to Actual Values
        # mu protected against overflow
        mu_pos_neg_mask = tf.cast(tf.less(self.nb_log_mu, 0.0), tf.float32)
        mu = (tf.multiply(mu_pos_neg_mask, tf.exp(-tf.abs(self.nb_log_mu))) +
              tf.multiply((1 - mu_pos_neg_mask), tf.divide(1, tf.exp(-tf.abs(self.nb_log_mu))
                                                           + eps)))
        # theta protected against overflow
        theta_pos_neg_mask = tf.cast(tf.less(self.nb_log_theta, 0.0), tf.float32)
        theta = (tf.multiply(theta_pos_neg_mask, tf.exp(-tf.abs(self.nb_log_theta))) +
                 tf.multiply((1 - theta_pos_neg_mask),
                             tf.divide(1.0, tf.exp(-tf.abs(self.nb_log_theta)) + eps)))

        # BASE FORMULATION
        # Negative Binomial (Overflow Protected)
        pre_nb = (tf.lgamma(self.X + theta + eps) - tf.lgamma(self.X + 1 + eps) -
                  tf.lgamma(theta + eps) +
                  (theta * (tf.log(theta + eps) - tf.log(theta + mu + eps))) +
                  (self.X * (tf.log(mu + eps) - tf.log(theta + mu + eps))))
        nb_mask = tf.cast(tf.less(pre_nb, 0.0), tf.float32)
        nb = (tf.multiply(nb_mask, tf.exp(-tf.abs(pre_nb))) +
              tf.multiply((1 - nb_mask), tf.divide(1.0, tf.exp(-tf.abs(pre_nb)) + eps)))
        # Log Likelihood w/ EPS
        ll = tf.log(nb + eps)
        return -tf.reduce_sum(ll, 1)

    def gaussian_loss(self, eps=1e-8):
        # Revert to Actual Values
        # Mu protected against overflow
        mu_pos_neg_mask = tf.cast(tf.less(self.gauss_log_mu, 0.0), tf.float32)
        mu = (tf.multiply(mu_pos_neg_mask, tf.exp(-tf.abs(self.gauss_log_mu))) +
              tf.multiply((1 - mu_pos_neg_mask), tf.divide(1, tf.exp(-tf.abs(self.gauss_log_mu))
                                                           + eps)))
        # Theta protected against overflow (Sigma^2)
        theta_pos_neg_mask = tf.cast(tf.less(self.gauss_log_theta, 0.0), tf.float32)
        theta = (tf.multiply(theta_pos_neg_mask, tf.exp(-tf.abs(self.gauss_log_theta))) +
                 tf.multiply((1 - theta_pos_neg_mask),
                             tf.divide(1.0, tf.exp(-tf.abs(self.gauss_log_theta)) + eps)))
        # BASE FORMULATION - LOG LIKELIHOOD GAUSSIAN
        lead_term = -tf.log(tf.multiply(tf.cast(2.0, tf.float32), tf.multiply(np.pi, theta)))
        last_term = -tf.divide(tf.square(tf.subtract(self.X, mu)),
                               tf.multiply(tf.cast(2.0, tf.float32), theta))
        ll = lead_term + last_term
        return -tf.reduce_sum(ll, 1)

    def bce_loss(self):
        return self.in_dim * metrics.binary_crossentropy(self.X, self.X_recon)

    def alt_bce_loss(self, offset=1e-7):
        # bound by clipping to avoid nan
        obs_ = tf.clip_by_value(self.X_recon, offset, 1 - offset)
        return -tf.reduce_sum(self.X * tf.log(obs_) +
                              (1 - self.X) * tf.log(1 - obs_), 1)

    def vamp_prior(self):
        # Vamp Prior
        log_p_z = self.log_vamp_prior(self.z, self.z_vamp_mu, self.z_vamp_std,
                                      self.vamp_comp, self.args.latent_dim)
        # Approx. Posterior
        log_q_z = self.log_normal_diagonal(self.z, self.z_mu, self.z_std)
        # KL Divergence
        return -tf.subtract(log_p_z, log_q_z)

    def log_normal_diagonal(self, z, mean, log_var):
        """Multivariate diagonal normal distribution

        Args:
            z: random variable
            mean: mean
            log_var: lof of variance

        Returns:
            log_n_diag: natural log of prob dist
        """
        log_normal = -0.5 * (log_var + tf.divide(tf.square(tf.subtract(z, mean)), tf.exp(log_var)))
        log_normal = tf.reduce_sum(log_normal, axis=1)
        return log_normal

    def log_vamp_prior(self, z, mean, log_var, n_vamp_comp, z_dim):
        # Tile z, mean, log_var based on number of vamp components and
        # then reshape to [batch_size*vamp_comp, z_dim]
        z_expand = tf.reshape(tf.tile(z, [1, n_vamp_comp]), [-1, z_dim])
        mean_expand = tf.reshape(tf.tile(mean, [tf.shape(z)[0], 1]), [-1, z_dim])
        logvar_expand = tf.reshape(tf.tile(log_var, [tf.shape(z)[0], 1]), [-1, z_dim])
        # Calculate log normal diag
        log_normal = self.log_normal_diagonal(z_expand, mean_expand, logvar_expand)
        # Reshape to [batch_size, vamp_comp]
        log_normal_per_vamp_comp = tf.reshape(log_normal, [-1, n_vamp_comp])
        # Reduce via logsumexp (overflow protected by TF)
        log_sum_per_vamp_comp = tf.reduce_logsumexp(log_normal_per_vamp_comp, axis=1)
        log_p_z = log_sum_per_vamp_comp - tf.log(self.tf_vamp_comp)
        return log_p_z

    def kld_loss(self):
        return -0.5 * tf.reduce_sum(1 + self.z_std - tf.square(self.z_mu) - tf.exp(self.z_std),
                                    axis=1)

    def kld_loss_alt(self):
        log_p_z = tf.reduce_sum(-0.5 * tf.pow(self.z, 2), axis=1)
        log_q_z = tf.reduce_sum(
            -0.5 * (self.z_std + tf.divide(tf.square(tf.subtract(self.z, self.z_mu)),
                                           tf.exp(self.z_std))), axis=1)
        return -(log_p_z - log_q_z)

    def mmd_loss(self):
        true_samples = tf.random_normal(tf.stack([tf.shape(self.z)[0], self.args.latent_dim]))
        x_kernel = self.compute_kernel(true_samples, true_samples)
        y_kernel = self.compute_kernel(self.z, self.z)
        xy_kernel = self.compute_kernel(true_samples, self.z)
        mmd_error = (tf.reduce_sum(x_kernel, axis=1) + tf.reduce_sum(y_kernel, axis=1)
                     - (2 * tf.reduce_sum(xy_kernel, axis=1)))
        return 50*mmd_error

    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(
            -tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def zi_gumbel_bce_loss(self, tau=0.75, eps=1e-20):
        # Calculate drop and non-drop probs
        log_p = tf.log(tf.exp(-tf.square(self.X_recon)) + eps)
        log_q = tf.log((1-tf.exp(-tf.square(self.X_recon))) + eps)
        # Gumbel uniform sample
        g_p = self.gumbel_softmax()
        g_q = self.gumbel_softmax()
        # Softmax
        s_top = tf.exp(tf.divide((log_p + g_p), tau))
        s_bottom = tf.add(tf.exp(tf.divide((log_p + g_p), tau)),
                          tf.exp(tf.divide((log_q + g_q), tau)))
        # Multiply Gumbel Softmax by Xrecon
        s = tf.multiply(tf.divide(s_top, s_bottom), self.X_recon)
        # BCE w/ Gumbel Samples
        return self.in_dim * metrics.binary_crossentropy(self.X, s)

    def gumbel_softmax(self, eps=1e-8):
        smp_gumbel = -tf.log(-tf.log(tf.random_uniform(shape=tf.shape(self.X_recon)) + eps) + eps)
        return smp_gumbel

    def encoder_layers(self):
        with tf.variable_scope(add_name('variational_gaussian', self.v_name)):
            # Iterate through the encoder network
            h = None
            for idx, weights in enumerate(self.encoder_weights):
                if idx == 0:
                    h = self.layer_fnc.get(self.args.act_fnc)(self.X, self.encoder_weights[idx],
                                                              self.encoder_biases[idx], self.phase,
                                                              self.args.dropout)
                else:
                    h = self.layer_fnc.get(self.args.act_fnc)(h, self.encoder_weights[idx],
                                                              self.encoder_biases[idx], self.phase,
                                                              self.args.dropout)
            # Calculate the latent representation
            z_mu = self.layer_fnc.get('dense')(h, self.latent_dict.get('w_z_mean'),
                                               self.latent_dict.get('b_z_mean'), self.phase,
                                               self.args.dropout)
            z_std = self.layer_fnc.get('dense')(h, self.latent_dict.get('w_z_std'),
                                                self.latent_dict.get('b_z_std'), self.phase,
                                                self.args.dropout)
            return z_mu, z_std

    def vamp_encoder_layers(self):
        # Reuse the Encoder Scope
        with tf.variable_scope(add_name('variational_gaussian_vamp', self.v_name), reuse=True):
            # Iterate through encoder but with vamp_prior inputs
            h_vamp = None
            for idx, weights in enumerate(self.encoder_weights):
                if idx == 0:
                    h_vamp = self.layer_fnc.get(
                        self.args.act_fnc)(self.vamp_inputs, self.encoder_weights[idx],
                                           self.encoder_biases[idx], self.phase, self.args.dropout)
                else:
                    h_vamp = self.layer_fnc.get(
                        self.args.act_fnc)(h_vamp, self.encoder_weights[idx],
                                           self.encoder_biases[idx], self.phase, self.args.dropout)
            # Calculate the latent representation
            z_vamp_mu = self.layer_fnc.get(
                'dense')(h_vamp, self.latent_dict.get('w_z_mean'), self.latent_dict.get('b_z_mean'),
                         self.phase, self.args.dropout)
            z_vamp_std = self.layer_fnc.get(
                'dense')(h_vamp, self.latent_dict.get('w_z_std'), self.latent_dict.get('b_z_std'),
                         self.phase, self.args.dropout)
            return z_vamp_mu, z_vamp_std

    def decoder_layers(self):
        with tf.variable_scope(add_name('generative_model', self.v_name)):
            # Iterate through the decoder network
            h = None
            for idx, weights in enumerate(self.decoder_weights):
                # First Layer uses Z
                if idx == 0:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        self.z, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
                # Last layer uses a sigmoid layer
                elif idx == (len(self.decoder_weights) - 1):
                    h = self.layer_fnc.get('sigmoid')(
                        h, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
                else:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        h, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
            return h

    def zinb_decoder_layers(self):
        with tf.variable_scope(add_name('zinb_generative_model', self.v_name)):
            # Iterate through the decoder network
            h = None
            for idx, weights in enumerate(self.decoder_weights):
                # First Layer uses Z
                if idx == 0:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        self.z, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
                else:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        h, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
            # FINAL LAYER IS ZINB
            # ZINB Log Mean
            zinb_log_mu = self.layer_fnc.get('dense')(
                h, self.zinb_dict.get('w_zinb_log_mu'), self.zinb_dict.get('b_zinb_log_mu'),
                self.phase, self.args.dropout)
            # ZINB Log Inverse Dispersion
            zinb_log_theta = self.zinb_dict.get('theta_simple')
            # ZINB Pi
            zinb_logit_pi = self.layer_fnc.get('dense')(
                h, self.zinb_dict.get('w_zinb_logit_pi'), self.zinb_dict.get('b_zinb_logit_pi'),
                self.phase, self.args.dropout)
            return zinb_log_mu, zinb_log_theta, zinb_logit_pi

    def nb_decoder_layers(self):
        with tf.variable_scope(add_name('nb_generative_model', self.v_name)):
            # Iterate through the decoder network
            h = None
            for idx, weights in enumerate(self.decoder_weights):
                # First Layer uses Z
                if idx == 0:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        self.z, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
                else:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        h, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
            # FINAL LAYER IS NB
            # NB Log Mean
            nb_log_mu = self.layer_fnc.get('dense')(
                h, self.nb_dict.get('w_nb_log_mu'), self.nb_dict.get('b_nb_log_mu'),
                self.phase, self.args.dropout)
            # NB Log Inverse Dispersion
            nb_log_theta = self.nb_dict.get('theta_simple')
            return nb_log_mu, nb_log_theta

    def gaussian_decoder_layers(self):
        with tf.variable_scope(add_name('gaussian_generative_model', self.v_name)):
            # Iterate through decoder
            h = None
            for idx, weights in enumerate(self.decoder_weights):
                # First Layer uses Z
                if idx == 0:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        self.z, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
                else:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        h, self.decoder_weights[idx], self.decoder_biases[idx], self.phase,
                        self.args.dropout)
            # FINAL LAYER IS GAUSSIAN (MU - Mean, THETA - Variance)
            # Gaussian MU
            gauss_log_mu = self.layer_fnc.get('dense')(
                h, self.gauss_dict.get('w_gauss_log_mu'), self.gauss_dict.get('b_gauss_log_mu'),
                self.phase, self.args.dropout)
            # Gaussian THETA
            gauss_log_theta = self.layer_fnc.get('dense')(
                h, self.gauss_dict.get('w_gauss_log_theta'), self.gauss_dict.get('b_gauss_log_theta'),
                self.phase, self.args.dropout)
            return gauss_log_mu, gauss_log_theta

    def gen_decoder_nn_weights(self, add_out_dim=True):
        with tf.variable_scope(add_name('generative_model_weights', self.v_name)):
            module_logger.info('Generating Initial Weights For Decoder...')
            # Make an iterable list
            layer_list = [self.args.latent_dim] + self.args.decoder_dims
            if add_out_dim:
                layer_list += [self.in_dim]
            with tf.name_scope('decoder_w'):
                decoder_w_list = []
                for idx in range(len(layer_list) - 1):
                    decoder_w_list.append(
                        tf.get_variable(
                            ('d_W_' + str(idx)), shape=[layer_list[idx], layer_list[idx+1]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True)))
                    self.log_init(layer_list[idx], layer_list[idx+1], self.args.act_fnc)
                return decoder_w_list

    def gen_decoder_nn_bias(self, add_out_dim=True):
        with tf.variable_scope(add_name('generative_model_biases', self.v_name)):
            module_logger.info('Generating Initial Biases For Decoder...')
            # Make an iterable list
            layer_list = self.args.decoder_dims
            if add_out_dim:
                layer_list += [self.in_dim]
            with tf.name_scope('decoder_b'):
                decoder_b_list = []
                for idx in range(len(layer_list)):
                    decoder_b_list.append(
                        tf.get_variable(('d_b_' + str(idx)), shape=[layer_list[idx]],
                                        initializer=tf.zeros_initializer()))
                return decoder_b_list

    def gen_gated_decoder_nn_weights(self, add_out_dim=True):
        with tf.variable_scope(add_name('generative_model_gated_weights', self.v_name)):
            module_logger.info('Generating Initial Weights For Decoder...')
            # Make an iterable list
            layer_list = [self.args.latent_dim] + self.args.decoder_dims
            if add_out_dim:
                layer_list += [self.in_dim]
            with tf.name_scope('decoder_w_g_'):
                decoder_w_list = []
                for idx in range(len(layer_list) - 1):
                    w_dict = {}
                    # Generate the h weights
                    w_dict.update({'h_weight': tf.get_variable(
                        ('d_W_h_' + str(idx)), shape=[layer_list[idx], layer_list[idx+1]],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
                    # Generate the g weights
                    w_dict.update({'g_weight': tf.get_variable(
                        ('d_W_g_' + str(idx)), shape=[layer_list[idx], layer_list[idx + 1]],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
                    decoder_w_list.append(w_dict)
                    self.log_init(layer_list[idx], layer_list[idx+1], self.args.act_fnc)
                return decoder_w_list

    def gen_gated_decoder_nn_bias(self, add_out_dim=True):
        with tf.variable_scope(add_name('generative_model_gated_biases', self.v_name)):
            module_logger.info('Generating Initial Biases For Decoder...')
            # Make an iterable list
            layer_list = self.args.decoder_dims
            if add_out_dim:
                layer_list += [self.in_dim]
            with tf.name_scope('decoder_b_g_'):
                decoder_b_list = []
                for idx in range(len(layer_list)):
                    b_dict = {}
                    # Generate h bias
                    b_dict.update({'h_bias': tf.get_variable(
                        ('d_b_h_' + str(idx)), shape=[layer_list[idx]],
                        initializer=tf.zeros_initializer())})
                    # Generate g bias
                    b_dict.update({'g_bias': tf.get_variable(
                        ('d_b_g_' + str(idx)), shape=[layer_list[idx]],
                        initializer=tf.zeros_initializer())})
                    decoder_b_list.append(b_dict)
                return decoder_b_list

    def gen_encoder_nn_weights(self):
        with tf.variable_scope(add_name('variational_gaussian_weights', self.v_name)):
            module_logger.info('Generating Initial Weights For Encoder...')
            # Make an iterable list
            layer_list = [self.in_dim] + self.args.encoder_dims
            with tf.name_scope('encoder_w'):
                encoder_w_list = []
                for idx in range(len(layer_list) - 1):
                    encoder_w_list.append(
                        tf.get_variable(
                            ('e_W_' + str(idx)), shape=[layer_list[idx], layer_list[idx+1]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True)))
                    self.log_init(layer_list[idx], layer_list[idx+1], self.args.act_fnc)
                return encoder_w_list

    def gen_encoder_nn_bias(self):
        with tf.variable_scope(add_name('variational_gaussian_biases', self.v_name)):
            module_logger.info('Generating Initial Biases For Encoder...')
            # Make an iterable list
            layer_list = self.args.encoder_dims
            with tf.name_scope('encoder_b'):
                encoder_b_list = []
                for idx in range(len(layer_list)):
                    encoder_b_list.append(
                        tf.get_variable(('e_b_' + str(idx)), shape=[layer_list[idx]],
                                        initializer=tf.zeros_initializer()))
                return encoder_b_list

    def gen_gated_encoder_nn_weights(self):
        with tf.variable_scope(add_name('variational_gaussian_gated_weights', self.v_name)):
            module_logger.info('Generating Initial Weights For Encoder...')
            # Make an iterable list
            layer_list = [self.in_dim] + self.args.encoder_dims
            with tf.name_scope('encoder_w_g_'):
                encoder_w_list = []
                for idx in range(len(layer_list) - 1):
                    w_dict = {}
                    # Generate the h weight
                    w_dict.update({'h_weight': tf.get_variable(
                        ('e_W_h_' + str(idx)), shape=[layer_list[idx], layer_list[idx+1]],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
                    # Generate the g weight
                    w_dict.update({'g_weight': tf.get_variable(
                        ('e_W_g_' + str(idx)), shape=[layer_list[idx], layer_list[idx + 1]],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
                    # Append to list
                    encoder_w_list.append(w_dict)
                    self.log_init(layer_list[idx], layer_list[idx+1], self.args.act_fnc)
                return encoder_w_list

    def gen_gated_encoder_nn_bias(self):
        with tf.variable_scope(add_name('variational_gaussian_gated_biases', self.v_name)):
            module_logger.info('Generating Initial Biases For Encoder...')
            # Make an iterable list
            layer_list = self.args.encoder_dims
            with tf.name_scope('encoder_b_g_'):
                encoder_b_list = []
                for idx in range(len(layer_list)):
                    b_dict = {}
                    # Generate h bias
                    b_dict.update({'h_bias': tf.get_variable(
                        ('e_b_h_' + str(idx)), shape=[layer_list[idx]],
                        initializer=tf.zeros_initializer())})
                    # Generate g bias
                    b_dict.update({'g_bias': tf.get_variable(
                        ('e_b_g_' + str(idx)), shape=[layer_list[idx]],
                        initializer=tf.zeros_initializer())})
                    encoder_b_list.append(b_dict)
                return encoder_b_list

    def gen_latent_weights(self):
        with tf.variable_scope(add_name('variational_gaussian_latent', self.v_name)):
            module_logger.info('Generating Initial Weights and Biases For Latent Distribution...')
            # Mean -- Weights
            latent_info = {'w_z_mean': tf.get_variable(
                'e_W_z_mean', shape=[self.args.encoder_dims[-1], self.args.latent_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))}
            self.log_init(self.args.encoder_dims[-1], self.args.latent_dim, 'Gaussian MEAN')
            # Std -- Weights
            latent_info.update({'w_z_std': tf.get_variable(
                'e_W_z_std', shape=[self.args.encoder_dims[-1], self.args.latent_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
            self.log_init(self.args.encoder_dims[-1], self.args.latent_dim, 'Gaussian STD')
            # Mean -- Bias
            latent_info.update({'b_z_mean': tf.get_variable(
                'e_B_z_mean', shape=[self.args.latent_dim], initializer=tf.zeros_initializer())})
            # Std -- Bias
            latent_info.update({'b_z_std': tf.get_variable(
                'e_B_z_std', shape=[self.args.latent_dim], initializer=tf.zeros_initializer())})
            return latent_info

    def gen_library_weights(self):
        with tf.variable_scope(add_name('variational_gaussian_library', self.v_name)):
            module_logger.info('Generating Initial Weights and Biases For Latent Distribution...')
            # Mean -- Weights
            library_info = {'w_lib_mean': tf.get_variable(
                'e_W_lib_mean', shape=[self.args.encoder_dims[-1], self.args.latent_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))}
            self.log_init(self.args.encoder_dims[-1], self.args.latent_dim, 'Library MEAN')
            # Std -- Weights
            library_info.update({'w_lib_std': tf.get_variable(
                'e_W_lib_std', shape=[self.args.encoder_dims[-1], self.args.latent_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
            self.log_init(self.args.encoder_dims[-1], self.args.latent_dim, 'Library STD')
            # Mean -- Bias
            library_info.update({'b_lib_mean': tf.get_variable(
                'e_B_lib_mean', shape=[self.args.latent_dim], initializer=tf.zeros_initializer())})
            # Std -- Bias
            library_info.update({'b_lib_std': tf.get_variable(
                'e_B_lib_std', shape=[self.args.latent_dim], initializer=tf.zeros_initializer())})
            return library_info

    def gen_zinb_vars(self):
        with tf.variable_scope(add_name('generating_zinb_variables', self.v_name)):
            #  Weights
            # ZINB MU
            zinb_info = {'w_zinb_log_mu': tf.get_variable(
                'd_W_zinb_log_mu', shape=[self.args.decoder_dims[-1], self.in_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))}
            self.log_init(self.args.decoder_dims[-1], self.in_dim, 'ZINB LOG MU')
            # ZINB THETA
            zinb_info.update({'w_zinb_log_theta': tf.get_variable(
                'd_W_zinb_log_theta', shape=[self.args.decoder_dims[-1], self.in_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
            self.log_init(self.args.decoder_dims[-1], self.in_dim, 'ZINB LOG THETA')
            # ZINB PI
            zinb_info.update({'w_zinb_logit_pi': tf.get_variable(
                'd_W_zinb_pi', shape=[self.args.decoder_dims[-1], self.in_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
            self.log_init(self.args.decoder_dims[-1], self.in_dim, 'ZINB LOGIT PI')
            # Bias
            # ZINB MU
            zinb_info.update({'b_zinb_log_mu': tf.get_variable(
                'd_B_zinb_log_mu', shape=[self.in_dim], initializer=tf.zeros_initializer())})
            # ZINB Theta
            zinb_info.update({'b_zinb_log_theta': tf.get_variable(
                'd_B_zinb_log_theta', shape=[self.in_dim], initializer=tf.zeros_initializer())})
            # ZINB PI
            zinb_info.update({'b_zinb_logit_pi': tf.get_variable(
                'd_B_zinb_logit_pi', shape=[self.in_dim], initializer=tf.zeros_initializer())})
            # Simple Theta
            zinb_info.update({'theta_simple': tf.Variable(tf.random_normal([self.in_dim]),
                                                          name="zinb_theta_simple")})
            return zinb_info

    def gen_gaussian_vars(self):
        with tf.variable_scope(add_name('generating_gaussian_variables', self.v_name)):
            # Weights
            # GAUSSIAN MU
            gaussian_info = {'w_gauss_log_mu': tf.get_variable(
                'd_W_gauss_log_mu', shape=[self.args.decoder_dims[-1], self.in_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))}
            self.log_init(self.args.decoder_dims[-1], self.in_dim, 'GAUSSIAN LOG MU')
            # GAUSSIAN THETA
            gaussian_info.update({'w_gauss_log_theta': tf.get_variable(
                'd_W_gauss_log_theta', shape=[self.args.decoder_dims[-1], self.in_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
            self.log_init(self.args.decoder_dims[-1], self.in_dim, 'GAUSSIAN LOG THETA')
            # Bias
            # GAUSSIAN MU
            gaussian_info.update({'b_gauss_log_mu': tf.get_variable(
                'd_B_gauss_log_mu', shape=[self.in_dim], initializer=tf.zeros_initializer())})
            # GAUSSIAN THETA
            gaussian_info.update({'b_gauss_log_theta': tf.get_variable(
                'd_B_gauss_log_theta', shape=[self.in_dim], initializer=tf.zeros_initializer())})
            return gaussian_info

    def gen_nb_vars(self):
        with tf.variable_scope(add_name('generating_nb_variables', self.v_name)):
            #  Weights
            # NB MU
            nb_info = {'w_nb_log_mu': tf.get_variable(
                'd_W_nb_log_mu', shape=[self.args.decoder_dims[-1], self.in_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))}
            self.log_init(self.args.decoder_dims[-1], self.in_dim, 'NB LOG MU')
            # NB THETA
            nb_info.update({'w_nb_log_theta': tf.get_variable(
                'd_W_nb_log_theta', shape=[self.args.decoder_dims[-1], self.in_dim],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))})
            self.log_init(self.args.decoder_dims[-1], self.in_dim, 'NB LOG THETA')
            # Bias
            # NB MU
            nb_info.update({'b_nb_log_mu': tf.get_variable(
                'd_B_nb_log_mu', shape=[self.in_dim], initializer=tf.zeros_initializer())})
            # NB Theta
            nb_info.update({'b_nb_log_theta': tf.get_variable(
                'd_B_nb_log_theta', shape=[self.in_dim], initializer=tf.zeros_initializer())})
            # Simple Theta
            nb_info.update({'theta_simple': tf.Variable(tf.random_normal([self.in_dim]),
                                                        name="nb_theta_simple")})
            return nb_info

class NeuralNetClassifier(BaseTF):
    def __init__(self, args, num_cluster, in_dim):
        super().__init__(args)
        # Make a new graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.in_dim = in_dim
            self.num_clusters = num_cluster
            # Input Data - z_mu from VAE
            self.X = tf.placeholder(tf.float32, shape=(None, self.in_dim),
                                    name='batch_latent_data')
            # Cluster Labels - from PhenoGraph
            self.labels = tf.placeholder(tf.int64, shape=[None], name='y_labels')
            # Placeholder for Phase Indication
            self.phase = tf.placeholder(tf.bool, name='phase')
            # Set-up weights and biases
            self.weights = self.gen_nn_weights()
            self.biases = self.gen_nn_bias()
            # Output of Dense Layer
            self.dense_output = self.nn_layers()
            # Loss Function -- built in Softmax
            self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.dense_output))
            # Probabilities
            self.probs = tf.nn.softmax(logits=self.dense_output)
            # Class Calls
            self.y = tf.argmax(input=self.probs, axis=1)
            self.correct = tf.equal(self.y, self.labels)
            # Train OP -- NN
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Optimizer
                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=self.args.lr).minimize(self.loss)
            # Bump a Session
            self.sess = tf.Session(graph=self.graph)
            # Initializing the tensor flow variables
            self.sess.run(tf.global_variables_initializer())
            # Make a saver object
            self.saver = tf.train.Saver(max_to_keep=1)


    def restore_model(self, full_path):
        module_logger.info('============LOADING BEST MODEL============')
        module_logger.info('Restoring Best Model...')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(full_path))
        module_logger.info('============BEST MODEL LOADED============')

    def save_model(self, full_path):
        # Model Save
        self.saver.save(self.sess, save_path=(os.path.join(full_path, 'model.chkp')))

    def partial_fit(self, X, y, phase):
        _, cost, acc = self.sess.run(
            [self.optimizer, self.loss, self.correct], feed_dict={self.X: X, self.labels: y,
                                                                  self.phase: phase})
        return cost, acc

    def get_loss(self, X, y, phase):
        cost = self.sess.run([self.loss], feed_dict={self.X: X, self.labels: y, self.phase: phase})
        return cost

    def get_prob(self, X, y, phase):
        prob = self.sess.run([self.probs], feed_dict={self.X: X, self.labels: y, self.phase: phase})
        return prob

    def get_model_output(self, X, y, phase):
        prob, class_labels, correct_labels, loss = self.sess.run(
            [self.probs, self.y, self.correct, self.loss], feed_dict={self.X: X, self.labels: y,
                                                                      self.phase: phase})
        return prob, class_labels, correct_labels, loss

    def pop_layers(self):
        weights, biases = self.sess.run([self.weights, self.biases])
        return weights, biases

    def gen_nn_weights(self):
        with tf.variable_scope('nn_weights'):
            module_logger.info('Generating Initial Weights For NN...')
            # Make an iterable list
            layer_list = [self.in_dim] + self.args.nn_dims + [self.num_clusters]
            with tf.name_scope('nn_w'):
                nn_w_list = []
                for idx in range(len(layer_list) - 1):
                    nn_w_list.append(
                        tf.get_variable(
                            ('nn_W_' + str(idx)), shape=[layer_list[idx], layer_list[idx+1]],
                             initializer=tf.contrib.layers.xavier_initializer(uniform=True)))
                    self.log_init(layer_list[idx], layer_list[idx+1], self.args.act_fnc)
                return nn_w_list

    def gen_nn_bias(self):
        with tf.variable_scope('nn_biases'):
            module_logger.info('Generating Initial Biases For NN...')
            # Make an iterable list
            layer_list = self.args.nn_dims + [self.num_clusters]
            with tf.name_scope('nn_b'):
                nn_b_list = []
                for idx in range(len(layer_list)):
                    nn_b_list.append(
                        tf.get_variable(('nn_b_' + str(idx)), shape=[layer_list[idx]],
                                        initializer=tf.zeros_initializer()))
                return nn_b_list

    def nn_layers(self):
        with tf.variable_scope('nn_classifier'):
            h = None
            for idx, weights in enumerate(self.weights):
                if idx == 0:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        self.X, self.weights[idx], self.biases[idx], self.phase, self.args.dropout)
                elif idx == (len(self.weights) - 1):
                    h = self.layer_fnc.get('dense')(
                        h, self.weights[idx], self.biases[idx], self.phase, self.args.dropout)
                else:
                    h = self.layer_fnc.get(self.args.act_fnc)(
                        h, self.weights[idx], self.biases[idx], self.phase, self.args.dropout)
            return h

    @staticmethod
    def log_init(dim_1, dim_2, act_fnc):
        module_logger.info('Creating NN Layer...')
        module_logger.info('Layer: ' + repr(act_fnc) + ' Activation - Dim: (' +
                           repr(dim_1) + ',' + repr(dim_2) + ')')

class DatasetManager(object):
    def __init__(self, f_valid, post_layer, data_path):
        # Read the data matrix
        module_logger.info('Loading data from - ' + data_path)
        self.data_mat = feather.read_dataframe(data_path)
        self.data_mat = self.data_mat.as_matrix()
        self.scaled_data = deepcopy(self.data_mat)
        indices = np.arange(self.scaled_data.shape[0])
        self.all_weights = np.ones((self.scaled_data.shape[0],))
        (self.train_data, self.valid_data,
         self.train_idx, self.valid_idx,
         self.train_weights, self.valid_weights) = train_test_split(
            self.scaled_data, indices, self.all_weights, test_size=f_valid)

        if (post_layer == 'zinb') or (post_layer == 'gauss'):
            # Split Data
            self.train_scaled_data = self.train_data
            self.valid_scaled_data = self.valid_data
            self.all_scaled_data = self.scaled_data
        else:
            sk_scale = MinMaxScaler()
            sk_scale.fit(self.train_data)
            self.train_scaled_data = sk_scale.transform(self.train_data)
            self.valid_scaled_data = sk_scale.transform(self.valid_data)
            self.all_scaled_data = sk_scale.transform(self.scaled_data)
        # Train/Valid IDX
        self.org_indices = np.zeros((self.scaled_data.shape[0]), dtype=np.int32)
        self.org_indices[self.valid_idx,] = 1
        module_logger.info('Finished handling data via manager...')
