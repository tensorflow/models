from builtins import range
from builtins import object
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
from itertools import product

class FeedForwardNet(object):
    """Custom feed-forward network layer."""
    def __init__(self, name, in_size, out_shape, layers=1, hidden_dim=32, final_nonlinearity=None, get_uncertainty=False):
        self.name = name
        self.in_size = in_size
        self.out_shape = out_shape
        self.out_size = np.prod(out_shape)
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.final_nonlinearity = (lambda x:x) if final_nonlinearity is None else final_nonlinearity
        self.get_uncertainty = get_uncertainty

        self.weights = [None] * layers
        self.biases = [None] * layers

        self.params_list = []

        with tf.variable_scope(name):
            for layer_i in range(self.layers):
                in_size = self.hidden_dim
                out_size = self.hidden_dim
                if layer_i == 0: in_size = self.in_size
                if layer_i == self.layers - 1: out_size = self.out_size
                self.weights[layer_i] = tf.get_variable("weights%d" % layer_i, [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
                self.biases[layer_i] = tf.get_variable("bias%d" % layer_i, [1, out_size], initializer=tf.constant_initializer(0.0))
                self.params_list += [self.weights[layer_i], self.biases[layer_i]]

    def __call__(self, x, stop_params_gradient=False, is_eval=True, ensemble_idxs=None, pre_expanded=None, reduce_mode="none"):
        original_shape = tf.shape(x)
        h = tf.reshape(x, [-1, self.in_size])
        for layer_i in range(self.layers):
            nonlinearity = tf.nn.relu if layer_i + 1 < self.layers else self.final_nonlinearity
            if stop_params_gradient: h = nonlinearity(tf.matmul(h, tf.stop_gradient(self.weights[layer_i])) + tf.stop_gradient(self.biases[layer_i]))
            else:             h = nonlinearity(tf.matmul(h, self.weights[layer_i]) + self.biases[layer_i])
        if len(self.out_shape) > 0: h = tf.reshape(h, tf.concat([original_shape[:-1], tf.constant(self.out_shape)], -1))
        else:                       h = tf.reshape(h, original_shape[:-1])
        if pre_expanded is None: pre_expanded = ensemble_idxs is not None
        if reduce_mode == "none" and not pre_expanded and self.get_uncertainty:
            if len(self.out_shape) > 0: h = tf.expand_dims(h, -2)
            else:                       h = tf.expand_dims(h, -1)
        return h

    def l2_loss(self):
        return tf.add_n([tf.reduce_sum(.5 * tf.square(mu)) for mu in self.params_list])

class BayesianDropoutFeedForwardNet(FeedForwardNet):
    """Custom feed-forward network layer, with dropout as a Bayesian approximation."""
    def __init__(self, name, in_size, out_shape, layers=1, hidden_dim=32, final_nonlinearity=None, get_uncertainty=False, keep_prob=.5, eval_sample_count=2, consistent_random_seed=False):
        super(BayesianDropoutFeedForwardNet, self).__init__(name, in_size, out_shape, layers=layers, hidden_dim=hidden_dim,
                                                            final_nonlinearity=final_nonlinearity, get_uncertainty=get_uncertainty)
        self.keep_prob = keep_prob
        self.eval_sample_count = eval_sample_count
        if eval_sample_count < 2: raise Exception("eval_sample_count must be at least 2 to estimate uncertainty")
        self.dropout_seed = tf.random_uniform([layers], maxval=1e18, dtype=tf.int64) if consistent_random_seed else [None] * layers

    def __call__(self, x, stop_params_gradient=False, is_eval=True, pre_expanded=False, ensemble_idxs=None, reduce_mode="none"):
        if is_eval:
            x = tf.tile(tf.expand_dims(x,0), tf.concat([tf.constant([self.eval_sample_count]), tf.ones_like(tf.shape(x))], 0))
        original_shape = tf.shape(x)
        h = tf.reshape(x, [-1, self.in_size])
        for layer_i in range(self.layers):
            nonlinearity = tf.nn.relu if layer_i + 1 < self.layers else self.final_nonlinearity
            if layer_i > 0: h = tf.nn.dropout(h, keep_prob=self.keep_prob, seed=self.dropout_seed[layer_i])
            if stop_params_gradient: h = nonlinearity(tf.matmul(h, tf.stop_gradient(self.weights[layer_i])) + tf.stop_gradient(self.biases[layer_i]))
            else:                    h = nonlinearity(tf.matmul(h, self.weights[layer_i]) + self.biases[layer_i])
        if len(self.out_shape) > 0: h = tf.reshape(h, tf.concat([original_shape[:-1], tf.constant(self.out_shape)], -1))
        else:                       h = tf.reshape(h, original_shape[:-1])
        if is_eval:
            h, uncertainty = tf.nn.moments(h, 0)
            if self.get_uncertainty: return h, uncertainty
            else:                    return h
        else:
            return h


class EnsembleFeedForwardNet(FeedForwardNet):
    """Custom feed-forward network layer with an ensemble."""
    def __init__(self, name, in_size, out_shape, layers=1, hidden_dim=32, final_nonlinearity=None, get_uncertainty=False, ensemble_size=2, train_sample_count=2, eval_sample_count=2):
        if train_sample_count > ensemble_size: raise Exception("train_sample_count cannot be larger than ensemble size")
        if eval_sample_count > ensemble_size: raise Exception("eval_sample_count cannot be larger than ensemble size")
        self.name = name
        self.in_size = in_size
        self.out_shape = out_shape
        self.out_size = np.prod(out_shape)
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.final_nonlinearity = (lambda x:x) if final_nonlinearity is None else final_nonlinearity
        self.get_uncertainty = get_uncertainty
        self.ensemble_size = ensemble_size
        self.train_sample_count = train_sample_count
        self.eval_sample_count = eval_sample_count

        self.weights = [None] * layers
        self.biases = [None] * layers

        self.params_list = []

        with tf.variable_scope(name):
            for layer_i in range(self.layers):
                in_size = self.hidden_dim
                out_size = self.hidden_dim
                if layer_i == 0: in_size = self.in_size
                if layer_i == self.layers - 1: out_size = self.out_size
                self.weights[layer_i] = tf.get_variable("weights%d" % layer_i, [ensemble_size, in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
                self.biases[layer_i] = tf.get_variable("bias%d" % layer_i, [ensemble_size, out_size], initializer=tf.constant_initializer(0.0))
                self.params_list += [self.weights[layer_i], self.biases[layer_i]]

    def __call__(self, x, stop_params_gradient=False, is_eval=True, ensemble_idxs=None, pre_expanded=None, reduce_mode="none"):
        if pre_expanded is None: pre_expanded = ensemble_idxs is not None
        if ensemble_idxs is None:
            ensemble_idxs = tf.random_shuffle(tf.range(self.ensemble_size))
            ensemble_sample_n = self.eval_sample_count if is_eval else self.train_sample_count
            ensemble_idxs = ensemble_idxs[:ensemble_sample_n]
        else:
            ensemble_sample_n = tf.shape(ensemble_idxs)[0]

        weights = [tf.gather(w, ensemble_idxs, axis=0) for w in self.weights]
        biases = [tf.expand_dims(tf.gather(b, ensemble_idxs, axis=0),0) for b in self.biases]

        original_shape = tf.shape(x)
        if pre_expanded: h = tf.reshape(x, [-1, ensemble_sample_n, self.in_size])
        else:            h = tf.tile(tf.reshape(x, [-1, 1, self.in_size]), [1, ensemble_sample_n, 1])
        for layer_i in range(self.layers):
            nonlinearity = tf.nn.relu if layer_i + 1 < self.layers else self.final_nonlinearity
            if stop_params_gradient: h = nonlinearity(tf.einsum('bri,rij->brj', h, tf.stop_gradient(weights[layer_i])) + tf.stop_gradient(biases[layer_i]))
            else:                    h = nonlinearity(tf.einsum('bri,rij->brj', h, weights[layer_i]) + biases[layer_i])

        if pre_expanded:
            if len(self.out_shape) > 0: h = tf.reshape(h, tf.concat([original_shape[:-1], tf.constant(self.out_shape)], -1))
            else:                       h = tf.reshape(h, original_shape[:-1])
        else:
            if len(self.out_shape) > 0: h = tf.reshape(h, tf.concat([original_shape[:-1], tf.constant([ensemble_sample_n]), tf.constant(self.out_shape)], -1))
            else:                       h = tf.reshape(h, tf.concat([original_shape[:-1], tf.constant([ensemble_sample_n])], -1))

        if reduce_mode == "none":
            pass
        elif reduce_mode == "random":
            if len(self.out_shape) > 0: h = tf.reduce_sum(h * tf.reshape(tf.one_hot(tf.random_uniform([tf.shape(h)[0]], 0, ensemble_sample_n, dtype=tf.int64), ensemble_sample_n), tf.concat([tf.shape(h)[:1], tf.ones_like(tf.shape(h)[1:-2]), tf.constant([ensemble_sample_n]), tf.constant([1])], 0)), -2)
            else:                       h = tf.reduce_sum(h * tf.reshape(tf.one_hot(tf.random_uniform([tf.shape(h)[0]], 0, ensemble_sample_n, dtype=tf.int64), ensemble_sample_n), tf.concat([tf.shape(h)[:1], tf.ones_like(tf.shape(h)[1:-1]), tf.constant([ensemble_sample_n])], 0)), -1)
        elif reduce_mode == "mean":
            if len(self.out_shape) > 0: h = tf.reduce_mean(h, -2)
            else:                       h = tf.reduce_mean(h, -1)
        else: raise Exception("use a valid reduce mode: none, random, or mean")

        return h


class ReparamNormal(object):
    """Wrapper to make a feedforward network that outputs both mu and logsigma,
    for use in the reparameterization trick."""
    def __init__(self, base_net, name, in_size, out_shape, layers=2, hidden_dim=32, final_nonlinearity=None, ls_start_bias=0.0, final_net=FeedForwardNet, logsigma_min=-5., logsigma_max=2., **kwargs):
        assert layers > 1
        self.main_encoder = base_net(name+"_base", in_size, [hidden_dim], layers, hidden_dim, final_nonlinearity=tf.nn.relu, **kwargs)
        self.mu = final_net(name+"_mu", hidden_dim, out_shape, layers=1, final_nonlinearity=final_nonlinearity, **kwargs)
        self.logsigma = final_net(name+"_logsigma", hidden_dim, out_shape, layers=1, final_nonlinearity=None, **kwargs)
        self.ls_start_bias = ls_start_bias
        self.params_list = self.main_encoder.params_list + self.mu.params_list + self.logsigma.params_list
        self.logsigma_min = logsigma_min
        self.logsigma_max = logsigma_max

    def __call__(self, x):
        encoded = self.main_encoder(x)
        mu = self.mu(encoded)
        logsigma = tf.clip_by_value(self.logsigma(encoded) + self.ls_start_bias, self.logsigma_min, self.logsigma_max)
        return mu, logsigma

    def l2_loss(self):
        return self.main_encoder.l2_loss() + self.mu.l2_loss() + self.logsigma.l2_loss()
