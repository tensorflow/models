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
"""Tests for framework.grouping_regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from morph_net.framework import grouping_regularizers
from morph_net.testing import op_regularizer_stub


def _l2_reg_with_025_threshold(regularizers_to_group):
  return grouping_regularizers.L2GroupingRegularizer(regularizers_to_group,
                                                     0.25)


class GroupingRegularizersTest(parameterized.TestCase, tf.test.TestCase):

  # TODO: Add parametrized tests.
  def setUp(self):
    self._reg_vec1 = [0.1, 0.3, 0.6, 0.2]
    self._alive_vec1 = [False, True, True, False]
    self._reg_vec2 = [0.2, 0.4, 0.5, 0.1]
    self._alive_vec2 = [False, True, False, True]
    self._reg_vec3 = [0.3, 0.2, 0.0, 0.25]
    self._alive_vec3 = [False, True, False, True]

    self._reg1 = op_regularizer_stub.OpRegularizerStub(self._reg_vec1,
                                                       self._alive_vec1)
    self._reg2 = op_regularizer_stub.OpRegularizerStub(self._reg_vec2,
                                                       self._alive_vec2)
    self._reg3 = op_regularizer_stub.OpRegularizerStub(self._reg_vec3,
                                                       self._alive_vec3)

  def testMaxGroupingRegularizer(self):
    group_reg = grouping_regularizers.MaxGroupingRegularizer(
        [self._reg1, self._reg2])
    with self.test_session():
      self.assertAllEqual(
          [x or y for x, y in zip(self._alive_vec1, self._alive_vec2)],
          group_reg.alive_vector.eval())
      self.assertAllClose(
          [max(x, y) for x, y in zip(self._reg_vec1, self._reg_vec2)],
          group_reg.regularization_vector.eval(), 1e-5)

  def testL2GroupingRegularizer(self):
    group_reg = grouping_regularizers.L2GroupingRegularizer(
        [self._reg1, self._reg2], 0.25)
    expcted_reg_vec = [
        np.sqrt((x**2 + y**2))
        for x, y in zip(self._reg_vec1, self._reg_vec2)
    ]
    with self.test_session():
      self.assertAllEqual([x > 0.25 for x in expcted_reg_vec],
                          group_reg.alive_vector.eval())
      self.assertAllClose(expcted_reg_vec,
                          group_reg.regularization_vector.eval(), 1e-5)

  @parameterized.named_parameters(
      ('Max', grouping_regularizers.MaxGroupingRegularizer),
      ('L2', _l2_reg_with_025_threshold))
  def testOrderDoesNotMatter(self, create_reg):
    group12 = create_reg([self._reg1, self._reg2])
    group13 = create_reg([self._reg1, self._reg3])
    group23 = create_reg([self._reg2, self._reg3])

    group123 = create_reg([group12, self._reg3])
    group132 = create_reg([group13, self._reg2])
    group231 = create_reg([group23, self._reg1])

    with self.test_session():
      self.assertAllEqual(group123.alive_vector.eval(),
                          group132.alive_vector.eval())
      self.assertAllEqual(group123.alive_vector.eval(),
                          group231.alive_vector.eval())

      self.assertAllClose(group123.regularization_vector.eval(),
                          group132.regularization_vector.eval())
      self.assertAllClose(group123.regularization_vector.eval(),
                          group231.regularization_vector.eval())


if __name__ == '__main__':
  tf.test.main()
