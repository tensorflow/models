# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for google3.third_party.tensorflow_models.object_detection.utils.test_case."""

import numpy as np
import tensorflow as tf
from object_detection.utils import test_case


class TestCaseTest(test_case.TestCase):

  def test_simple(self):
    def graph_fn(tensora, tensorb):
      return tf.tensordot(tensora, tensorb, axes=1)

    tensora_np = np.ones(20)
    tensorb_np = tensora_np * 2
    output = self.execute(graph_fn, [tensora_np, tensorb_np])
    self.assertAllClose(output, 40.0)

  def test_two_outputs(self):
    def graph_fn(tensora, tensorb):
      return tensora + tensorb, tensora - tensorb
    tensora_np = np.ones(20)
    tensorb_np = tensora_np * 2
    output = self.execute(graph_fn, [tensora_np, tensorb_np])
    self.assertAllClose(output[0], tensora_np + tensorb_np)
    self.assertAllClose(output[1], tensora_np - tensorb_np)

  def test_function_with_tf_assert(self):
    def compute_fn(image):
      return tf.image.pad_to_bounding_box(image, 0, 0, 40, 40)

    image_np = np.random.rand(2, 20, 30, 3)
    output = self.execute(compute_fn, [image_np])
    self.assertAllEqual(output.shape, [2, 40, 40, 3])

  def test_tf2_only_test(self):
    """Set up tests only to run with TF2."""
    if self.is_tf2():
      def graph_fn(tensora, tensorb):
        return tensora + tensorb, tensora - tensorb
      tensora_np = np.ones(20)
      tensorb_np = tensora_np * 2
      output = self.execute_tf2(graph_fn, [tensora_np, tensorb_np])
      self.assertAllClose(output[0], tensora_np + tensorb_np)
      self.assertAllClose(output[1], tensora_np - tensorb_np)

  def test_tpu_only_test(self):
    """Set up tests only to run with TPU."""
    if self.has_tpu():
      def graph_fn(tensora, tensorb):
        return tensora + tensorb, tensora - tensorb
      tensora_np = np.ones(20)
      tensorb_np = tensora_np * 2
      output = self.execute_tpu(graph_fn, [tensora_np, tensorb_np])
      self.assertAllClose(output[0], tensora_np + tensorb_np)
      self.assertAllClose(output[1], tensora_np - tensorb_np)

if __name__ == '__main__':
  tf.test.main()
