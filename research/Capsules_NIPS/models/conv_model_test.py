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

"""Tests for conv_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from models import conv_model


class ConvModelTest(tf.test.TestCase):

  def setUp(self):
    self.hparams = tf.contrib.training.HParams(
        learning_rate=0.001,
        decay_rate=0.96,
        decay_steps=1,
        padding='SAME',
        verbose=False,
        loss_type='softmax')

  def testIntegrity(self):
    """Checks a multi_gpu call on ConvModel builds the desired graph.

    With the correct inference graph, multi_gpu is able to call inference
    multiple times without any increase in number of trainable variables or a
    duplication error. Each tower should have 4 set of (weight, bias) variable.
    """
    with tf.Graph().as_default():
      test_model = conv_model.ConvModel(self.hparams)
      toy_image = np.reshape(np.arange(32 * 32), (1, 1, 32, 32))
      input_image = tf.constant(toy_image, dtype=tf.float32)
      features = {
          'height': 32,
          'depth': 1,
          'images': input_image,
          'labels': tf.one_hot([2], 10),
          'num_targets': 1,
          'num_classes': 10,
      }
      _, tower_output = test_model.multi_gpu([features, features, features], 3)
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(trainable_vars), 8)
      _, classes = tower_output[0].logits.get_shape()
      self.assertEqual(10, classes.value)


if __name__ == '__main__':
  tf.test.main()
