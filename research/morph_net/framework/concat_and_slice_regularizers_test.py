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
"""Tests for framework.concat_and_slice_regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from morph_net.framework import concat_and_slice_regularizers
from morph_net.testing import op_regularizer_stub


class ConcatAndSliceRegularizersTest(tf.test.TestCase):

  def setUp(self):
    self._reg_vec1 = [0.1, 0.3, 0.6, 0.2]
    self._alive_vec1 = [False, True, True, False]
    self._reg_vec2 = [0.2, 0.4, 0.5]
    self._alive_vec2 = [False, True, False]
    self._reg1 = op_regularizer_stub.OpRegularizerStub(self._reg_vec1,
                                                       self._alive_vec1)
    self._reg2 = op_regularizer_stub.OpRegularizerStub(self._reg_vec2,
                                                       self._alive_vec2)

  def testConcatRegularizer(self):
    concat_reg = concat_and_slice_regularizers.ConcatRegularizer(
        [self._reg1, self._reg2])
    with self.test_session():
      self.assertAllEqual(self._alive_vec1 + self._alive_vec2,
                          concat_reg.alive_vector.eval())
      self.assertAllClose(self._reg_vec1 + self._reg_vec2,
                          concat_reg.regularization_vector.eval(), 1e-5)

  def testSliceRegularizer(self):
    concat_reg = concat_and_slice_regularizers.SlicingReferenceRegularizer(
        lambda: self._reg1, 1, 2)
    with self.test_session():
      self.assertAllEqual(self._alive_vec1[1:3],
                          concat_reg.alive_vector.eval())
      self.assertAllClose(self._reg_vec1[1:3],
                          concat_reg.regularization_vector.eval(), 1e-5)


if __name__ == '__main__':
  tf.test.main()
