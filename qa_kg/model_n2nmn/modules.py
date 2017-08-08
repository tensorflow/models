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

import tensorflow as tf


class Modules:

  def __init__(self, config, kb, word_vecs, num_choices, embedding_mat):
    self.config = config

    self.embedding_mat = embedding_mat

    # kb has shape [N_kb, 3]
    self.kb = kb
    self.embed_keys_e, self.embed_keys_r, self.embed_vals_e = self.embed_kb()

    # word_vecs has shape [T_decoder, N, D_txt]
    self.word_vecs = word_vecs
    self.num_choices = num_choices

  def embed_kb(self):
    keys_e, keys_r, vals_e = [], [], []
    for idx_sub, idx_rel, idx_obj in self.kb:
      keys_e.append(idx_sub)
      keys_r.append(idx_rel)
      vals_e.append(idx_obj)
    embed_keys_e = tf.nn.embedding_lookup(self.embedding_mat, keys_e)
    embed_keys_r = tf.nn.embedding_lookup(self.embedding_mat, keys_r)
    embed_vals_e = tf.nn.embedding_lookup(self.embedding_mat, vals_e)
    return embed_keys_e, embed_keys_r, embed_vals_e

  def _slice_word_vecs(self, time_idx, batch_idx):
    # this callable will be wrapped into a td.Function
    # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
    # time is highest dim in word_vecs
    joint_index = tf.stack([time_idx, batch_idx], axis=1)
    return tf.gather_nd(self.word_vecs, joint_index)

  # All the layers are wrapped with td.ScopedLayer
  def KeyFindModule(self,
                    time_idx,
                    batch_idx,
                    scope='KeyFindModule',
                    reuse=None):
    # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
    text_param = self._slice_word_vecs(time_idx, batch_idx)

    # Mapping: embed_keys_e x text_param -> att
    # Input:
    #   embed_keys_e: [N_kb, D_txt]
    #   text_param: [N, D_txt]
    # Output:
    #   att: [N, N_kb]
    #
    # Implementation:
    #   1. Elementwise multiplication between embed_key_e and text_param
    #   2. L2-normalization
    with tf.variable_scope(scope, reuse=reuse):
      m = tf.matmul(text_param, self.embed_keys_e, transpose_b=True)
      att = tf.nn.l2_normalize(m, dim=1)
    return att

  def KeyFilterModule(self,
                      input_0,
                      time_idx,
                      batch_idx,
                      scope='KeyFilterModule',
                      reuse=None):
    att_0 = input_0
    text_param = self._slice_word_vecs(time_idx, batch_idx)

    # Mapping: and(embed_keys_r x text_param, att) -> att
    # Input:
    #   embed_keys_r: [N_kb, D_txt]
    #   text_param: [N, D_txt]
    #   att_0: [N, N_kb]
    # Output:
    #   att: [N, N_kb]
    #
    # Implementation:
    #   1. Elementwise multiplication between embed_key_r and text_param
    #   2. L2-normalization
    #   3. Take the elementwise-min
    with tf.variable_scope(scope, reuse=reuse):
      m = tf.matmul(text_param, self.embed_keys_r, transpose_b=True)
      att_1 = tf.nn.l2_normalize(m, dim=1)
      att = tf.minimum(att_0, att_1)
    return att

  def ValDescribeModule(self,
                        input_0,
                        time_idx,
                        batch_idx,
                        scope='ValDescribeModule',
                        reuse=None):
    att = input_0

    # Mapping: att -> answer probs
    # Input:
    #   embed_vals_e: [N_kb, D_txt]
    #   att: [N, N_kb]
    #   embedding_mat: [self.num_choices, D_txt]
    # Output:
    #   answer_scores: [N, self.num_choices]
    #
    # Implementation:
    #   1. Attention-weighted sum over values
    #   2. Compute cosine similarity scores between the weighted sum and
    #      each candidate answer
    with tf.variable_scope(scope, reuse=reuse):
      # weighted_sum has shape [N, D_txt]
      weighted_sum = tf.matmul(att, self.embed_vals_e)
      # scores has shape [N, self.num_choices]
      scores = tf.matmul(
          weighted_sum,
          tf.nn.l2_normalize(self.embedding_mat, dim=1),
          transpose_b=True)
    return scores
