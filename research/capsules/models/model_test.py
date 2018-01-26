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

"""Tests for model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import mock
import tensorflow as tf

from models import model
from models.layers import variables


class ModelTest(tf.test.TestCase):

  def setUp(self):
    self.hparams = tf.contrib.training.HParams(
        learning_rate=0.001,
        decay_rate=0.96,
        decay_steps=1,
        loss_type='softmax')

  class MockModel(model.Model):
    """Implements toy inference method without variable declaration."""

    def inference(self, features):
      logits = tf.constant([[0.1, 0.2, 0.7]])
      return model.Inferred(logits, None)

  class MockModelWithVariable(model.Model):
    """Implements toy inference method with variable declaration."""

    def inference(self, features):
      logits = variables.bias_variable([1, 3])
      return model.Inferred(logits, None)

  @mock.patch.multiple(model.Model, __abstractmethods__=set())
  def testModelInitialization(self):
    """Checks the variables declared in the init method.

      Initialization step should only declare global_step as a non trainable
      variable.
    """
    with tf.Graph().as_default():
      model.Model(self.hparams)
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(trainable_vars), 0)
      global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      self.assertEqual(len(global_vars), 1)
      self.assertStartsWith(global_vars[0].name, 'global_step')

  def testSingleTower_WithoutVariable(self):
    """Checks model will raise error when there is no trainable variable."""
    with tf.Graph().as_default():
      test_model = self.MockModel(self.hparams)
      feature = {
          'labels': tf.one_hot([2], 3),
          'num_targets': 1,
      }
      with self.assertRaises(ValueError):
        test_model._single_tower(0, feature)

  def testSingleTower_MultipleCalls(self):
    """Checks the correct variable size after multiple single_tower calls.

      Multiple towers should not declare multiple variables and should share the
      trainable variables. Therefore, variable set size should stay at 1.
      Each tower should have its own operations therefore, total number of
      operations should increase for each tower by the same amount.
    """
    with tf.Graph().as_default() as graph:
      test_model = self.MockModelWithVariable(self.hparams)
      feature = {
          'labels': tf.one_hot([2], 3),
          'num_targets': 1,
      }
      test_model._single_tower(0, feature)
      first_ops = graph.get_operations()
      first_op_num = len(first_ops)
      test_model._single_tower(1, feature)
      second_ops = graph.get_operations()
      second_op_num = len(second_ops)
      test_model._single_tower(2, feature)
      third_ops = graph.get_operations()
      third_op_num = len(third_ops)
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(trainable_vars), 1)
      self.assertEqual(second_op_num - first_op_num,
                       third_op_num - second_op_num)

  # Model is an abstract class. mock.patch empties the abstractmethods set
  # so that it can be initializable for testing other functions.
  @mock.patch.multiple(model.Model, __abstractmethods__=set())
  def testAverageGradients(self):
    """Checks the correct average for multiple towers and multiple variables.

    The test model has 2 towers with 2 variables shared between them.
    var_0 is getting 1.0 + 3.0 as gradient -> average: 2.0
    var_1 is getting 2.0 + 4.0 as gradient -> average: 3.0
    """
    with tf.Graph().as_default():
      with tf.Session() as session:
        test_model = model.Model(self.hparams)
        grad_0 = tf.constant(1.0)
        grad_1 = tf.constant(2.0)
        tower_0 = [(grad_0, 'var_0'), (grad_1, 'var_1')]
        grad_2 = tf.constant(3.0)
        grad_3 = tf.constant(4.0)
        tower_1 = [(grad_2, 'var_0'), (grad_3, 'var_1')]
        tower_grads = [tower_0, tower_1]
        average_grads = test_model._average_gradients(tower_grads)
        self.assertEqual(len(average_grads), 2)
        self.assertEqual('var_0', average_grads[0][1])
        average_grad_0 = session.run(average_grads[0][0])
        self.assertEqual(2.0, average_grad_0)

        self.assertEqual('var_1', average_grads[1][1])
        average_grad_1 = session.run(average_grads[1][0])
        self.assertEqual(3.0, average_grad_1)

  def testMultiGpu(self):
    """Checks the correct attribute values after multi_gpu call.

    Since tests don't have GPU access, test only covers the correct number of
    elements in the result lists.
    """
    with tf.Graph().as_default():
      test_model = self.MockModelWithVariable(self.hparams)
      feature = {
          'labels': tf.one_hot([2], 3),
          'num_targets': 1,
      }
      test_model.multi_gpu([feature, feature, feature], 3)
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual(len(trainable_vars), 1)


if __name__ == '__main__':
  tf.test.main()
