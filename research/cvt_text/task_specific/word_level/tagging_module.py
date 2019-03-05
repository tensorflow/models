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

"""Sequence tagging module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from corpus_processing import minibatching
from model import model_helpers
from model import task_module


class TaggingModule(task_module.SemiSupervisedModule):
  def __init__(self, config, task_name, n_classes, inputs,
               encoder):
    super(TaggingModule, self).__init__()
    self.task_name = task_name
    self.n_classes = n_classes
    self.labels = labels = tf.placeholder(tf.float32, [None, None, None],
                                          name=task_name + '_labels')

    class PredictionModule(object):
      def __init__(self, name, input_reprs, roll_direction=0, activate=True):
        self.name = name
        with tf.variable_scope(name + '/predictions'):
          projected = model_helpers.project(input_reprs, config.projection_size)
          if activate:
            projected = tf.nn.relu(projected)
          self.logits = tf.layers.dense(projected, n_classes, name='predict')

        targets = labels
        targets *= (1 - inputs.label_smoothing)
        targets += inputs.label_smoothing / n_classes
        self.loss = model_helpers.masked_ce_loss(
            self.logits, targets, inputs.mask, roll_direction=roll_direction)

    primary = PredictionModule('primary',
                               ([encoder.uni_reprs, encoder.bi_reprs]))
    ps = [
        PredictionModule('full', ([encoder.uni_reprs, encoder.bi_reprs]),
                         activate=False),
        PredictionModule('forwards', [encoder.uni_fw]),
        PredictionModule('backwards', [encoder.uni_bw]),
        PredictionModule('future', [encoder.uni_fw], roll_direction=1),
        PredictionModule('past', [encoder.uni_bw], roll_direction=-1),
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
      feed[self.labels] = np.eye(self.n_classes)[labels]
