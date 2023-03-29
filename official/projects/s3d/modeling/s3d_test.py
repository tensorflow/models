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

"""Tests for S3D model."""

from absl.testing import parameterized
import tensorflow as tf

from official.projects.s3d.modeling import s3d


class S3dTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (7, 224, 224, 3),
      (7, 128, 128, 3),
      (7, 256, 256, 3),
      (7, 192, 192, 3),
      (64, 224, 224, 3),
      (32, 224, 224, 3),
      (64, 224, 224, 11),
      (32, 224, 224, 11),
  )
  def test_build(self, num_frames, height, width, first_temporal_kernel_size):
    batch_size = 5

    input_shape = [batch_size, num_frames, height, width, 3]
    input_specs = tf.keras.layers.InputSpec(shape=input_shape)
    network = s3d.S3D(
        input_specs=input_specs
    )
    inputs = tf.keras.Input(shape=input_shape[1:], batch_size=input_shape[0])
    endpoints = network(inputs)

    temporal_1a = (num_frames - 1)//2 + 1
    expected_shapes = {
        'Conv2d_1a_7x7': [5, temporal_1a, height//2, width//2, 64],
        'Conv2d_2b_1x1': [5, temporal_1a, height//4, width//4, 64],
        'Conv2d_2c_3x3': [5, temporal_1a, height//4, height//4, 192],
        'MaxPool_2a_3x3': [5, temporal_1a, height//4, height//4, 64],
        'MaxPool_3a_3x3': [5, temporal_1a, height//8, width//8, 192],
        'Mixed_3b': [5, temporal_1a, height//8, width//8, 256],
        'Mixed_3c': [5, temporal_1a, height//8, width//8, 480],
        'MaxPool_4a_3x3': [5, temporal_1a//2, height//16, width//16, 480],
        'Mixed_4b': [5, temporal_1a//2, height//16, width//16, 512],
        'Mixed_4c': [5, temporal_1a//2, height//16, width//16, 512],
        'Mixed_4d': [5, temporal_1a//2, height//16, width//16, 512],
        'Mixed_4e': [5, temporal_1a//2, height//16, width//16, 528],
        'Mixed_4f': [5, temporal_1a//2, height//16, width//16, 832],
        'MaxPool_5a_2x2': [5, temporal_1a//4, height//32, width//32, 832],
        'Mixed_5b': [5, temporal_1a//4, height//32, width//32, 832],
        'Mixed_5c': [5, temporal_1a//4, height//32, width//32, 1024],
    }

    output_shapes = dict()
    for end_point, output_tensor in endpoints.items():
      output_shapes[end_point] = output_tensor.shape.as_list()
    self.assertDictEqual(output_shapes, expected_shapes)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        input_specs=tf.keras.layers.InputSpec(shape=(5, 64, 224, 224, 3)),
        final_endpoint='Mixed_5c',
        first_temporal_kernel_size=3,
        temporal_conv_start_at='Conv2d_2c_3x3',
        gating_start_at='Conv2d_2c_3x3',
        swap_pool_and_1x1x1=True,
        gating_style='CELL',
        use_sync_bn=False,
        norm_momentum=0.999,
        norm_epsilon=0.001,
        temporal_conv_initializer=tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.01),
        temporal_conv_type='2+1d',
        kernel_initializer='truncated_normal',
        kernel_regularizer='l2',
        depth_multiplier=1.0
    )
    network = s3d.S3D(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = s3d.S3D.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

if __name__ == '__main__':
  tf.test.main()
