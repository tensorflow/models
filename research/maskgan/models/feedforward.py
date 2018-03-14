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

"""Simple FNN model definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
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

  with tf.variable_scope("dis", reuse=reuse):
    if not FLAGS.dis_share_embedding:
      embedding = tf.get_variable("embedding",
                                  [FLAGS.vocab_size, hparams.dis_rnn_size])

    embeddings = tf.nn.embedding_lookup(embedding, sequence)

    # Input matrices.
    W = tf.get_variable(
        "W",
        initializer=tf.truncated_normal(
            shape=[3 * hparams.dis_embedding_dim, hparams.dis_hidden_dim],
            stddev=0.1))
    b = tf.get_variable(
        "b", initializer=tf.constant(0.1, shape=[hparams.dis_hidden_dim]))

    # Output matrices.
    W_out = tf.get_variable(
        "W_out",
        initializer=tf.truncated_normal(
            shape=[hparams.dis_hidden_dim, 1], stddev=0.1))
    b_out = tf.get_variable("b_out", initializer=tf.constant(0.1, shape=[1]))

    predictions = []
    for t in xrange(FLAGS.sequence_length):
      if t > 0:
        tf.get_variable_scope().reuse_variables()

      inp = embeddings[:, t]

      if t > 0:
        past_inp = tf.unstack(embeddings[:, 0:t], axis=1)
        avg_past_inp = tf.add_n(past_inp) / len(past_inp)
      else:
        avg_past_inp = tf.zeros_like(inp)

      if t < FLAGS.sequence_length:
        future_inp = tf.unstack(embeddings[:, t:], axis=1)
        avg_future_inp = tf.add_n(future_inp) / len(future_inp)
      else:
        avg_future_inp = tf.zeros_like(inp)

      # Cumulative input.
      concat_inp = tf.concat([avg_past_inp, inp, avg_future_inp], axis=1)

      # Hidden activations.
      hidden = tf.nn.relu(tf.nn.xw_plus_b(concat_inp, W, b, name="scores"))

      # Add dropout
      with tf.variable_scope("dropout"):
        hidden = tf.nn.dropout(hidden, FLAGS.keep_prob)

      # Output.
      output = tf.nn.xw_plus_b(hidden, W_out, b_out, name="output")

      predictions.append(output)
    predictions = tf.stack(predictions, axis=1)
    return tf.squeeze(predictions, axis=2)
