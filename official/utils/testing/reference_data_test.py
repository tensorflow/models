# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""This module tests generic behavior of reference data tests.

This test is not intended to test every layer of interest, and models should
test the layers that affect them. This test is primarily focused on ensuring
that reference_data.BaseTest functions as intended. If there is a legitimate
change such as a change to TensorFlow which changes graph construction, tests
can be regenerated with the following command:

  $ python3 reference_data_test.py -regen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import tensorflow as tf  # pylint: disable=g-bad-import-order
from official.utils.misc import keras_utils
from official.utils.testing import reference_data


class GoldenBaseTest(reference_data.BaseTest):
  """Class to ensure that reference data testing runs properly."""

  def setUp(self):
    if keras_utils.is_v2_0():
      tf.compat.v1.disable_eager_execution()
    super(GoldenBaseTest, self).setUp()

  @property
  def test_name(self):
    return "reference_data_test"

  def _uniform_random_ops(self, test=False, wrong_name=False, wrong_shape=False,
                          bad_seed=False, bad_function=False):
    """Tests number generation and failure modes.

    This test is of a very simple graph: the generation of a 1x1 random tensor.
    However, it is also used to confirm that the tests are actually checking
    properly by failing in predefined ways.

    Args:
      test: Whether or not to run as a test case.
      wrong_name: Whether to assign the wrong name to the tensor.
      wrong_shape: Whether to create a tensor with the wrong shape.
      bad_seed: Whether or not to perturb the random seed.
      bad_function: Whether to perturb the correctness function.
    """
    name = "uniform_random"

    g = tf.Graph()
    with g.as_default():
      seed = self.name_to_seed(name)
      seed = seed + 1 if bad_seed else seed
      tf.compat.v1.set_random_seed(seed)
      tensor_name = "wrong_tensor" if wrong_name else "input_tensor"
      tensor_shape = (1, 2) if wrong_shape else (1, 1)
      input_tensor = tf.compat.v1.get_variable(
          tensor_name, dtype=tf.float32,
          initializer=tf.random.uniform(tensor_shape, maxval=1)
      )

    def correctness_function(tensor_result):
      result = float(tensor_result[0, 0])
      result = result + 0.1 if bad_function else result
      return [result]
    self._save_or_test_ops(
        name=name, graph=g, ops_to_eval=[input_tensor], test=test,
        correctness_function=correctness_function
    )

  def _dense_ops(self, test=False):
    name = "dense"

    g = tf.Graph()
    with g.as_default():
      tf.compat.v1.set_random_seed(self.name_to_seed(name))
      input_tensor = tf.compat.v1.get_variable(
          "input_tensor", dtype=tf.float32,
          initializer=tf.random.uniform((1, 2), maxval=1)
      )
      layer = tf.compat.v1.layers.dense(inputs=input_tensor, units=4)
      layer = tf.compat.v1.layers.dense(inputs=layer, units=1)

    self._save_or_test_ops(
        name=name, graph=g, ops_to_eval=[layer], test=test,
        correctness_function=self.default_correctness_function
    )

  def test_uniform_random(self):
    self._uniform_random_ops(test=True)

  def test_tensor_name_error(self):
    with self.assertRaises(AssertionError):
      self._uniform_random_ops(test=True, wrong_name=True)

  @unittest.skipIf(keras_utils.is_v2_0(), "TODO:(b/136010138) Fails on TF 2.0.")
  def test_tensor_shape_error(self):
    with self.assertRaises(AssertionError):
      self._uniform_random_ops(test=True, wrong_shape=True)

  def test_incorrectness_function(self):
    with self.assertRaises(AssertionError):
      self._uniform_random_ops(test=True, bad_function=True)

  def test_dense(self):
    self._dense_ops(test=True)

  def regenerate(self):
    self._uniform_random_ops(test=False)
    self._dense_ops(test=False)


if __name__ == "__main__":
  reference_data.main(argv=sys.argv, test_class=GoldenBaseTest)
