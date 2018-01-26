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

"""Tests for variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.layers import variables


class VariablesTest(tf.test.TestCase):

  def testVariableDeclaration(self):
    """Checks the value and shape of the squidge output given a rank 2 input."""
    with tf.Graph().as_default():
      with self.test_session() as sess:
        weights = variables.weight_variable((1, 2), stddev=0.1)
        bias = variables.bias_variable((1))
        sess.run(tf.global_variables_initializer())
        w_value, b_value = sess.run([weights, bias])
        self.assertNear(w_value[0][0], 0.0, 0.2)
        self.assertNear(w_value[0][1], 0.0, 0.2)
        self.assertEqual(b_value, 0.1)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.assertEqual(len(trainable_vars), 2)
        self.assertStartsWith(trainable_vars[0].name, 'weights')
        self.assertStartsWith(trainable_vars[1].name, 'biases')


if __name__ == '__main__':
  tf.test.main()
