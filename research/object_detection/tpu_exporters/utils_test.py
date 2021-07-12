# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test for Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow.compat.v1 as tf

from object_detection.tpu_exporters import utils


class UtilsTest(tf.test.TestCase):

  def testBfloat16ToFloat32(self):
    bfloat16_tensor = tf.random.uniform([2, 3], dtype=tf.bfloat16)
    float32_tensor = utils.bfloat16_to_float32(bfloat16_tensor)
    self.assertEqual(float32_tensor.dtype, tf.float32)

  def testOtherDtypesNotConverted(self):
    int32_tensor = tf.ones([2, 3], dtype=tf.int32)
    converted_tensor = utils.bfloat16_to_float32(int32_tensor)
    self.assertEqual(converted_tensor.dtype, tf.int32)

  def testBfloat16ToFloat32Nested(self):
    tensor_dict = {
        'key1': tf.random.uniform([2, 3], dtype=tf.bfloat16),
        'key2': [
            tf.random.uniform([1, 2], dtype=tf.bfloat16) for _ in range(3)
        ],
        'key3': tf.ones([2, 3], dtype=tf.int32),
    }
    tensor_dict = utils.bfloat16_to_float32_nested(tensor_dict)

    self.assertEqual(tensor_dict['key1'].dtype, tf.float32)
    for t in tensor_dict['key2']:
      self.assertEqual(t.dtype, tf.float32)
    self.assertEqual(tensor_dict['key3'].dtype, tf.int32)


if __name__ == '__main__':
  tf.test.main()
