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
"""Tests for exporting utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.export import export


class ExportUtilsTest(tf.test.TestCase):
  """Tests for the ExportUtils."""

  def test_build_tensor_serving_input_receiver_fn(self):
    receiver_fn = export.build_tensor_serving_input_receiver_fn(shape=[4, 5])
    with tf.Graph().as_default():
      receiver = receiver_fn()
      self.assertIsInstance(
          receiver, tf.estimator.export.TensorServingInputReceiver)

      self.assertIsInstance(receiver.features, tf.Tensor)
      self.assertEqual(receiver.features.shape, tf.TensorShape([1, 4, 5]))
      self.assertEqual(receiver.features.dtype, tf.float32)
      self.assertIsInstance(receiver.receiver_tensors, dict)
      # Note that Python 3 can no longer index .values() directly; cast to list.
      self.assertEqual(list(receiver.receiver_tensors.values())[0].shape,
                       tf.TensorShape([1, 4, 5]))

  def test_build_tensor_serving_input_receiver_fn_batch_dtype(self):
    receiver_fn = export.build_tensor_serving_input_receiver_fn(
        shape=[4, 5], dtype=tf.int8, batch_size=10)

    with tf.Graph().as_default():
      receiver = receiver_fn()
      self.assertIsInstance(
          receiver, tf.estimator.export.TensorServingInputReceiver)

      self.assertIsInstance(receiver.features, tf.Tensor)
      self.assertEqual(receiver.features.shape, tf.TensorShape([10, 4, 5]))
      self.assertEqual(receiver.features.dtype, tf.int8)
      self.assertIsInstance(receiver.receiver_tensors, dict)
      # Note that Python 3 can no longer index .values() directly; cast to list.
      self.assertEqual(list(receiver.receiver_tensors.values())[0].shape,
                       tf.TensorShape([10, 4, 5]))


if __name__ == "__main__":
  tf.test.main()
