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
"""Tests for inception."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import slim


def get_variables(scope=None):
  return slim.variables.get_variables(scope)


def get_variables_by_name(name):
  return slim.variables.get_variables_by_name(name)


class CollectionsTest(tf.test.TestCase):

  def testVariables(self):
    batch_size = 5
    height, width = 299, 299
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope([slim.ops.conv2d],
                          batch_norm_params={'decay': 0.9997}):
        slim.inception.inception_v3(inputs)
      self.assertEqual(len(get_variables()), 388)
      self.assertEqual(len(get_variables_by_name('weights')), 98)
      self.assertEqual(len(get_variables_by_name('biases')), 2)
      self.assertEqual(len(get_variables_by_name('beta')), 96)
      self.assertEqual(len(get_variables_by_name('gamma')), 0)
      self.assertEqual(len(get_variables_by_name('moving_mean')), 96)
      self.assertEqual(len(get_variables_by_name('moving_variance')), 96)

  def testVariablesWithoutBatchNorm(self):
    batch_size = 5
    height, width = 299, 299
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope([slim.ops.conv2d],
                          batch_norm_params=None):
        slim.inception.inception_v3(inputs)
      self.assertEqual(len(get_variables()), 196)
      self.assertEqual(len(get_variables_by_name('weights')), 98)
      self.assertEqual(len(get_variables_by_name('biases')), 98)
      self.assertEqual(len(get_variables_by_name('beta')), 0)
      self.assertEqual(len(get_variables_by_name('gamma')), 0)
      self.assertEqual(len(get_variables_by_name('moving_mean')), 0)
      self.assertEqual(len(get_variables_by_name('moving_variance')), 0)

  def testVariablesByLayer(self):
    batch_size = 5
    height, width = 299, 299
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope([slim.ops.conv2d],
                          batch_norm_params={'decay': 0.9997}):
        slim.inception.inception_v3(inputs)
      self.assertEqual(len(get_variables()), 388)
      self.assertEqual(len(get_variables('conv0')), 4)
      self.assertEqual(len(get_variables('conv1')), 4)
      self.assertEqual(len(get_variables('conv2')), 4)
      self.assertEqual(len(get_variables('conv3')), 4)
      self.assertEqual(len(get_variables('conv4')), 4)
      self.assertEqual(len(get_variables('mixed_35x35x256a')), 28)
      self.assertEqual(len(get_variables('mixed_35x35x288a')), 28)
      self.assertEqual(len(get_variables('mixed_35x35x288b')), 28)
      self.assertEqual(len(get_variables('mixed_17x17x768a')), 16)
      self.assertEqual(len(get_variables('mixed_17x17x768b')), 40)
      self.assertEqual(len(get_variables('mixed_17x17x768c')), 40)
      self.assertEqual(len(get_variables('mixed_17x17x768d')), 40)
      self.assertEqual(len(get_variables('mixed_17x17x768e')), 40)
      self.assertEqual(len(get_variables('mixed_8x8x2048a')), 36)
      self.assertEqual(len(get_variables('mixed_8x8x2048b')), 36)
      self.assertEqual(len(get_variables('logits')), 2)
      self.assertEqual(len(get_variables('aux_logits')), 10)

  def testVariablesToRestore(self):
    batch_size = 5
    height, width = 299, 299
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope([slim.ops.conv2d],
                          batch_norm_params={'decay': 0.9997}):
        slim.inception.inception_v3(inputs)
      variables_to_restore = tf.get_collection(
          slim.variables.VARIABLES_TO_RESTORE)
      self.assertEqual(len(variables_to_restore), 388)
      self.assertListEqual(variables_to_restore, get_variables())

  def testVariablesToRestoreWithoutLogits(self):
    batch_size = 5
    height, width = 299, 299
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope([slim.ops.conv2d],
                          batch_norm_params={'decay': 0.9997}):
        slim.inception.inception_v3(inputs, restore_logits=False)
      variables_to_restore = tf.get_collection(
          slim.variables.VARIABLES_TO_RESTORE)
      self.assertEqual(len(variables_to_restore), 384)

  def testRegularizationLosses(self):
    batch_size = 5
    height, width = 299, 299
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        slim.inception.inception_v3(inputs)
      losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(losses), len(get_variables_by_name('weights')))

  def testTotalLossWithoutRegularization(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1001
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      dense_labels = tf.random_uniform((batch_size, num_classes))
      with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0):
        logits, end_points = slim.inception.inception_v3(
            inputs,
            num_classes=num_classes)
        # Cross entropy loss for the main softmax prediction.
        slim.losses.cross_entropy_loss(logits,
                                       dense_labels,
                                       label_smoothing=0.1,
                                       weight=1.0)
        # Cross entropy loss for the auxiliary softmax head.
        slim.losses.cross_entropy_loss(end_points['aux_logits'],
                                       dense_labels,
                                       label_smoothing=0.1,
                                       weight=0.4,
                                       scope='aux_loss')
      losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
      self.assertEqual(len(losses), 2)

  def testTotalLossWithRegularization(self):
    batch_size = 5
    height, width = 299, 299
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      dense_labels = tf.random_uniform((batch_size, num_classes))
      with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        logits, end_points = slim.inception.inception_v3(inputs, num_classes)
        # Cross entropy loss for the main softmax prediction.
        slim.losses.cross_entropy_loss(logits,
                                       dense_labels,
                                       label_smoothing=0.1,
                                       weight=1.0)
        # Cross entropy loss for the auxiliary softmax head.
        slim.losses.cross_entropy_loss(end_points['aux_logits'],
                                       dense_labels,
                                       label_smoothing=0.1,
                                       weight=0.4,
                                       scope='aux_loss')
      losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
      self.assertEqual(len(losses), 2)
      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(reg_losses), 98)


if __name__ == '__main__':
  tf.test.main()
