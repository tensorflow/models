# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Author: aneelakantan (Arvind Neelakantan)
"""

import numpy as np
import tensorflow as tf


class Parameters:

  def __init__(self, u):
    self.utility = u
    self.init_seed_counter = 0
    self.word_init = {}

  def parameters(self, utility):
    params = {}
    inits = []
    embedding_dims = self.utility.FLAGS.embedding_dims
    params["unit"] = tf.Variable(
        self.RandomUniformInit([len(utility.operations_set), embedding_dims]))
    params["word"] = tf.Variable(
        self.RandomUniformInit([utility.FLAGS.vocab_size, embedding_dims]))
    params["word_match_feature_column_name"] = tf.Variable(
        self.RandomUniformInit([1]))
    params["controller"] = tf.Variable(
        self.RandomUniformInit([2 * embedding_dims, embedding_dims]))
    params["column_controller"] = tf.Variable(
        self.RandomUniformInit([2 * embedding_dims, embedding_dims]))
    params["column_controller_prev"] = tf.Variable(
        self.RandomUniformInit([embedding_dims, embedding_dims]))
    params["controller_prev"] = tf.Variable(
        self.RandomUniformInit([embedding_dims, embedding_dims]))
    global_step = tf.Variable(1, name="global_step")
    #weigths of question and history RNN (or LSTM)
    key_list = ["question_lstm"]
    for key in key_list:
      # Weights going from inputs to nodes.
      for wgts in ["ix", "fx", "cx", "ox"]:
        params[key + "_" + wgts] = tf.Variable(
            self.RandomUniformInit([embedding_dims, embedding_dims]))
      # Weights going from nodes to nodes.
      for wgts in ["im", "fm", "cm", "om"]:
        params[key + "_" + wgts] = tf.Variable(
            self.RandomUniformInit([embedding_dims, embedding_dims]))
      #Biases for the gates and cell
      for bias in ["i", "f", "c", "o"]:
        if (bias == "f"):
          print "forget gate bias"
          params[key + "_" + bias] = tf.Variable(
              tf.random_uniform([embedding_dims], 1.0, 1.1, self.utility.
                                tf_data_type[self.utility.FLAGS.data_type]))
        else:
          params[key + "_" + bias] = tf.Variable(
              self.RandomUniformInit([embedding_dims]))
    params["history_recurrent"] = tf.Variable(
        self.RandomUniformInit([3 * embedding_dims, embedding_dims]))
    params["history_recurrent_bias"] = tf.Variable(
        self.RandomUniformInit([1, embedding_dims]))
    params["break_conditional"] = tf.Variable(
        self.RandomUniformInit([2 * embedding_dims, embedding_dims]))
    init = tf.initialize_all_variables()
    return params, global_step, init

  def RandomUniformInit(self, shape):
    """Returns a RandomUniform Tensor between -param_init and param_init."""
    param_seed = self.utility.FLAGS.param_seed
    self.init_seed_counter += 1
    return tf.random_uniform(
        shape, -1.0 *
        (np.float32(self.utility.FLAGS.param_init)
        ).astype(self.utility.np_data_type[self.utility.FLAGS.data_type]),
        (np.float32(self.utility.FLAGS.param_init)
        ).astype(self.utility.np_data_type[self.utility.FLAGS.data_type]),
        self.utility.tf_data_type[self.utility.FLAGS.data_type],
        param_seed + self.init_seed_counter)
