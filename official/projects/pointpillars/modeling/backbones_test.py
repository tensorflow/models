# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for backbones."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.pointpillars.modeling import backbones


class BackboneTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([1, 32, 32, 3], 1, 1),
      ([2, 32, 64, 4], 1, 3),
  )
  def test_network_creation(self, input_shape, min_level, max_level):
    batch_size = input_shape[0]
    inputs = tf_keras.Input(shape=input_shape[1:], batch_size=batch_size)
    backbone = backbones.Backbone(input_shape, min_level, max_level)
    endpoints = backbone(inputs)
    _, h, w, c = input_shape
    for level in range(min_level, max_level + 1):
      self.assertAllEqual([
          batch_size,
          int(h / 2**level),
          int(w / 2**level),
          int(c * 2**(level - 1))
      ], endpoints[str(level)].shape.as_list())

  def test_serialization(self):
    kwargs = dict(
        input_specs=[1, 64, 64, 3],
        min_level=2,
        max_level=4,
        num_convs=3,
        kernel_regularizer=None,
    )
    net = backbones.Backbone(**kwargs)
    expected_config = kwargs
    self.assertEqual(net.get_config(), expected_config)

    new_net = backbones.Backbone.from_config(net.get_config())
    self.assertAllEqual(net.get_config(), new_net.get_config())
    _ = new_net.to_json()


if __name__ == '__main__':
  tf.test.main()
