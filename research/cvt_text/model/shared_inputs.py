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

"""Placeholders for non-task-specific model inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Inputs(object):
  def __init__(self, config):
    self._config = config
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    self.label_smoothing = tf.placeholder(tf.float32, name='label_smoothing')
    self.lengths = tf.placeholder(tf.int32, shape=[None], name='lengths')
    self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
    self.words = tf.placeholder(tf.int32, shape=[None, None], name='words')
    self.chars = tf.placeholder(tf.int32, shape=[None, None, None],
                                name='chars')

  def create_feed_dict(self, mb, is_training):
    cvt = mb.task_name == 'unlabeled'
    return {
        self.keep_prob: 1.0 if not is_training else
                        (self._config.unlabeled_keep_prob if cvt else
                         self._config.labeled_keep_prob),
        self.label_smoothing: self._config.label_smoothing
                              if (is_training and not cvt) else 0.0,
        self.lengths: mb.lengths,
        self.words: mb.words,
        self.chars: mb.chars,
        self.mask: mb.mask.astype('float32')
    }
