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

"""Tests for capsule_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from models import capsule_model


class CapsuleModelTest(tf.test.TestCase):

  def setUp(self):
    self.hparams = tf.contrib.training.HParams(
        learning_rate=0.001,
        decay_rate=0.96,
        decay_steps=1,
        num_prime_capsules=2,
        padding='SAME',
        leaky=False,
        routing=3,
        verbose=False,
        loss_type='softmax',
        remake=False)

  def testBuildCapsule(self):
    """Checks the correct shape of capsule output and total number of variables.

    The output shape should be [batch, 10, 16]. Also each capsule layer should
    declare 2 sets of variables (weight and bias), therefore single call to
    _build_capsule declares 4 variables for a total of 2 capsule layers.
    """
    with tf.Graph().as_default():
      test_model = capsule_model.CapsuleModel(self.hparams)
      toy_input = np.reshape(np.arange(256 * 14 * 14), (1, 1, 256, 14, 14))
      input_tensor = tf.constant(toy_input, dtype=tf.float32)
      output = test_model._build_capsule(input_tensor, 10)
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(trainable_vars), 4)
      _, capsules, atoms = output.get_shape()
      self.assertListEqual([10, 16], [capsules.value, atoms.value])

  def testInference(self):
    """Checks the correct shape of capsule output and total number of variables.

    The output logit shape should be [batch, 10]. Also each layer should
    declare 2 sets of variables (weight and bias), therefore single call to
    inference declares 6 variables for a total of 3 layers.
    """
    with tf.Graph().as_default():
      test_model = capsule_model.CapsuleModel(self.hparams)
      toy_image = np.reshape(np.arange(32 * 32), (1, 1, 32, 32))
      input_image = tf.constant(toy_image, dtype=tf.float32)
      features = {
          'height': 32,
          'depth': 1,
          'num_classes': 10,
          'images': input_image
      }
      output = test_model.inference(features)
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(trainable_vars), 6)
      _, classes = output.logits.get_shape()
      self.assertEqual(10, classes.value)

  def testIntegrity(self):
    """Checks a multi_gpu call on CapsuleModel builds the desired graph.

    With the correct inference graph, multi_gpu is able to call inference
    multiple times without any increase in number of trainable variables or a
    duplication error.
    """
    with tf.Graph().as_default():
      test_model = capsule_model.CapsuleModel(self.hparams)
      toy_image = np.reshape(np.arange(32 * 32), (1, 1, 32, 32))
      input_image = tf.constant(toy_image, dtype=tf.float32)
      features = {
          'height': 32,
          'depth': 1,
          'images': input_image,
          'labels': tf.one_hot([2], 10),
          'num_classes': 10,
          'num_targets': 1,
      }
      _, tower_output = test_model.multi_gpu([features, features, features], 3)
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(trainable_vars), 6)
      _, classes = tower_output[0].logits.get_shape()
      self.assertEqual(10, classes.value)

  def testInferenceWithRemake(self):
    """Checks the correct shape of remakes and total number of variables.

    The reconstruction should have same shape as input. Each remake network
    should declare 6 sets of variables (weight and bias) and different targets
    should share the variables.
    """
    with tf.Graph().as_default():
      self.hparams.parse('remake=True,verbose=True')
      test_model = capsule_model.CapsuleModel(self.hparams)
      toy_image = np.reshape(np.arange(32 * 32), (1, 1, 32, 32))
      input_image = tf.constant(toy_image, dtype=tf.float32)
      features = {
          'height': 32,
          'depth': 1,
          'images': input_image,
          'recons_image': input_image,
          'spare_image': input_image,
          'recons_label': tf.constant([2]),
          'spare_label': tf.constant([2]),
          'num_targets': 2,
          'num_classes': 10,
      }
      output = test_model.inference(features)
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(trainable_vars), 12)
      remake_1, remake_2 = output.remakes
      self.assertEqual(32 * 32, remake_1.get_shape()[1].value)
      self.assertEqual(32 * 32, remake_2.get_shape()[1].value)


if __name__ == '__main__':
  tf.test.main()
