# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for decoders."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.pointpillars.modeling import decoders


class DecoderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ({'1': [1, 32, 32, 3]},
       1, 1),
      ({'1': [1, 32, 32, 3],
        '2': [1, 16, 16, 6]},
       1, 2)
  )
  def test_network_creation(self, input_shape, min_level, max_level):
    """Test if network could be created and infer with expected shapes."""
    inputs = {}
    for k, v in input_shape.items():
      if k == str(min_level):
        batch_size, height, width, _ = v
      inputs[k] = tf_keras.Input(shape=v[1:], batch_size=batch_size)
    decoder = decoders.Decoder(input_shape)
    endpoints = decoder(inputs)

    self.assertLen(endpoints, 1)
    self.assertEqual(list(endpoints.keys())[0], str(min_level))

    self.assertIn(str(min_level), endpoints)
    expected_channels = input_shape[str(min_level)][-1] * 2 * (
        max_level - min_level + 1)
    self.assertAllEqual(endpoints[str(min_level)].shape.as_list(),
                        [batch_size, height, width, expected_channels])

  def test_serialization(self):
    kwargs = dict(
        input_specs={'1': [1, 64, 64, 3]},
        kernel_regularizer=None,
    )
    net = decoders.Decoder(**kwargs)
    expected_config = kwargs
    self.assertEqual(net.get_config(), expected_config)

    new_net = decoders.Decoder.from_config(net.get_config())
    self.assertAllEqual(net.get_config(), new_net.get_config())
    _ = new_net.to_json()


if __name__ == '__main__':
  tf.test.main()
