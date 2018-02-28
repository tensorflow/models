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

"""Variational Dropout."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def generate_dropout_masks(keep_prob, shape, amount):
  masks = []
  for _ in range(amount):
    dropout_mask = tf.random_uniform(shape) + (keep_prob)
    dropout_mask = tf.floor(dropout_mask) / (keep_prob)
    masks.append(dropout_mask)
  return masks


def generate_variational_dropout_masks(hparams, keep_prob):
  [batch_size, num_steps, size, num_layers] = [
      FLAGS.batch_size, FLAGS.sequence_length, hparams.gen_rnn_size,
      hparams.gen_num_layers
  ]
  if len(keep_prob) == 2:
    emb_keep_prob = keep_prob[0]  # keep prob for embedding matrix
    h2h_keep_prob = emb_keep_prob  # keep prob for hidden to hidden connections
    h2i_keep_prob = keep_prob[1]  # keep prob for hidden to input connections
    out_keep_prob = h2i_keep_prob  # keep probability for output state
  else:
    emb_keep_prob = keep_prob[0]  # keep prob for embedding matrix
    h2h_keep_prob = keep_prob[1]  # keep prob for hidden to hidden connections
    h2i_keep_prob = keep_prob[2]  # keep prob for hidden to input connections
    out_keep_prob = keep_prob[3]  # keep probability for output state
  h2i_masks = []  # Masks for input to recurrent connections
  h2h_masks = []  # Masks for recurrent to recurrent connections

  # Input word dropout mask
  emb_masks = generate_dropout_masks(emb_keep_prob, [num_steps, 1], batch_size)
  output_mask = generate_dropout_masks(out_keep_prob, [batch_size, size], 1)[0]
  h2i_masks = generate_dropout_masks(h2i_keep_prob, [batch_size, size],
                                     num_layers)
  h2h_masks = generate_dropout_masks(h2h_keep_prob, [batch_size, size],
                                     num_layers)
  return h2h_masks, h2i_masks, emb_masks, output_mask
