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

"""Dependency parsing module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from corpus_processing import minibatching
from model import model_helpers
from model import task_module


class DepparseModule(task_module.SemiSupervisedModule):
  def __init__(self, config, task_name, n_classes, inputs, encoder):
    super(DepparseModule, self).__init__()

    self.task_name = task_name
    self.n_classes = n_classes
    self.labels = labels = tf.placeholder(tf.float32, [None, None, None],
                                          name=task_name + '_labels')

    class PredictionModule(object):
      def __init__(self, name, dep_reprs, head_reprs, roll_direction=0):
        self.name = name
        with tf.variable_scope(name + '/predictions'):
          # apply hidden layers to the input representations
          arc_dep_hidden = model_helpers.project(
              dep_reprs, config.projection_size, 'arc_dep_hidden')
          arc_head_hidden = model_helpers.project(
              head_reprs, config.projection_size, 'arc_head_hidden')
          arc_dep_hidden = tf.nn.relu(arc_dep_hidden)
          arc_head_hidden = tf.nn.relu(arc_head_hidden)
          arc_head_hidden = tf.nn.dropout(arc_head_hidden, inputs.keep_prob)
          arc_dep_hidden = tf.nn.dropout(arc_dep_hidden, inputs.keep_prob)

          # bilinear classifier excluding the final dot product
          arc_head = tf.layers.dense(
              arc_head_hidden, config.depparse_projection_size, name='arc_head')
          W = tf.get_variable('shared_W',
                              shape=[config.projection_size, n_classes,
                                     config.depparse_projection_size])
          Wr = tf.get_variable('relation_specific_W',
                               shape=[config.projection_size,
                                      config.depparse_projection_size])
          Wr_proj = tf.tile(tf.expand_dims(Wr, axis=-2), [1, n_classes, 1])
          W += Wr_proj
          arc_dep = tf.tensordot(arc_dep_hidden, W, axes=[[-1], [0]])
          shape = tf.shape(arc_dep)
          arc_dep = tf.reshape(arc_dep,
                               [shape[0], -1, config.depparse_projection_size])

          # apply the transformer scaling trick to prevent dot products from
          # getting too large (possibly not necessary)
          scale = np.power(
              config.depparse_projection_size, 0.25).astype('float32')
          scale = tf.get_variable('scale', initializer=scale, dtype=tf.float32)
          arc_dep /= scale
          arc_head /= scale

          # compute the scores for each candidate arc
          word_scores = tf.matmul(arc_head, arc_dep, transpose_b=True)
          root_scores = tf.layers.dense(arc_head, n_classes, name='root_score')
          arc_scores = tf.concat([root_scores, word_scores], axis=-1)

          # disallow the model from making impossible predictions
          mask = inputs.mask
          mask_shape = tf.shape(mask)
          mask = tf.tile(tf.expand_dims(mask, -1), [1, 1, n_classes])
          mask = tf.reshape(mask, [-1, mask_shape[1] * n_classes])
          mask = tf.concat([tf.ones((mask_shape[0], 1)),
                            tf.zeros((mask_shape[0], n_classes - 1)), mask],
                           axis=1)
          mask = tf.tile(tf.expand_dims(mask, 1), [1, mask_shape[1], 1])
          arc_scores += (mask - 1) * 100.0

          self.logits = arc_scores
          self.loss = model_helpers.masked_ce_loss(
              self.logits, labels, inputs.mask,
              roll_direction=roll_direction)

    primary = PredictionModule(
        'primary',
        [encoder.uni_reprs, encoder.bi_reprs],
        [encoder.uni_reprs, encoder.bi_reprs])
    ps = [
        PredictionModule(
            'full',
            [encoder.uni_reprs, encoder.bi_reprs],
            [encoder.uni_reprs, encoder.bi_reprs]),
        PredictionModule('fw_fw', [encoder.uni_fw], [encoder.uni_fw]),
        PredictionModule('fw_bw', [encoder.uni_fw], [encoder.uni_bw]),
        PredictionModule('bw_fw', [encoder.uni_bw], [encoder.uni_fw]),
        PredictionModule('bw_bw', [encoder.uni_bw], [encoder.uni_bw]),
    ]

    self.unsupervised_loss = sum(p.loss for p in ps)
    self.supervised_loss = primary.loss
    self.probs = tf.nn.softmax(primary.logits)
    self.preds = tf.argmax(primary.logits, axis=-1)

  def update_feed_dict(self, feed, mb):
    if self.task_name in mb.teacher_predictions:
      feed[self.labels] = mb.teacher_predictions[self.task_name]
    elif mb.task_name != 'unlabeled':
      labels = minibatching.build_array(
          [[0] + e.labels + [0] for e in mb.examples])
      feed[self.labels] = np.eye(
          (1 + mb.words.shape[1]) * self.n_classes)[labels]

