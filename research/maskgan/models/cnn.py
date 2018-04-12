# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Simple CNN model definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def discriminator(hparams, sequence, is_training, reuse=None):
  """Define the Discriminator graph."""
  del is_training
  sequence = tf.cast(sequence, tf.int32)

  if FLAGS.dis_share_embedding:
    assert hparams.dis_rnn_size == hparams.gen_rnn_size, (
        "If you wish to share Discriminator/Generator embeddings, they must be"
        " same dimension.")
    with tf.variable_scope("gen/rnn", reuse=True):
      embedding = tf.get_variable("embedding",
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])

  dis_filter_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

  with tf.variable_scope("dis", reuse=reuse):
    if not FLAGS.dis_share_embedding:
      embedding = tf.get_variable("embedding",
                                  [FLAGS.vocab_size, hparams.dis_rnn_size])
    cnn_inputs = tf.nn.embedding_lookup(embedding, sequence)

    # Create a convolution layer for each filter size
    conv_outputs = []
    for filter_size in dis_filter_sizes:
      with tf.variable_scope("conv-%s" % filter_size):
        # Convolution Layer
        filter_shape = [
            filter_size, hparams.dis_rnn_size, hparams.dis_num_filters
        ]
        W = tf.get_variable(
            name="W", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
        b = tf.get_variable(
            name="b",
            initializer=tf.constant(0.1, shape=[hparams.dis_num_filters]))
        conv = tf.nn.conv1d(
            cnn_inputs, W, stride=1, padding="SAME", name="conv")

        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        conv_outputs.append(h)

    # Combine all the pooled features
    dis_num_filters_total = hparams.dis_num_filters * len(dis_filter_sizes)

    h_conv = tf.concat(conv_outputs, axis=2)
    h_conv_flat = tf.reshape(h_conv, [-1, dis_num_filters_total])

    # Add dropout
    with tf.variable_scope("dropout"):
      h_drop = tf.nn.dropout(h_conv_flat, FLAGS.keep_prob)

    with tf.variable_scope("fully_connected"):
      fc = tf.contrib.layers.fully_connected(
          h_drop, num_outputs=dis_num_filters_total / 2)

    # Final (unnormalized) scores and predictions
    with tf.variable_scope("output"):
      W = tf.get_variable(
          "W",
          shape=[dis_num_filters_total / 2, 1],
          initializer=tf.contrib.layers.xavier_initializer())
      b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[1]))
      predictions = tf.nn.xw_plus_b(fc, W, b, name="predictions")
      predictions = tf.reshape(
          predictions, shape=[FLAGS.batch_size, FLAGS.sequence_length])
  return predictions
