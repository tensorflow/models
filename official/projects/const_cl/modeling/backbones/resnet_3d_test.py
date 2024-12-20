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

"""Tests for resnet."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.const_cl.modeling.backbones import resnet_3d


class ResNet3DTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, 50, 4, 'v0', False, 0.0),
      (128, 50, 4, 'v0', False, 0.2),
      (256, 50, 4, 'v0', True, 0.2),
  )
  def test_network_creation(self, input_size, model_id, endpoint_filter_scale,
                            stem_type, se_ratio, init_stochastic_depth_rate):
    """Test creation of ResNet3D family models."""
    tf_keras.backend.set_image_data_format('channels_last')
    temporal_strides = [1, 1, 1, 1]
    temporal_kernel_sizes = [(3, 3, 3), (3, 1, 3, 1), (3, 1, 3, 1, 3, 1),
                             (1, 3, 1)]
    use_self_gating = [True, False, True, False]

    network = resnet_3d.ResNet3DY(
        model_id=model_id,
        temporal_strides=temporal_strides,
        temporal_kernel_sizes=temporal_kernel_sizes,
        use_self_gating=use_self_gating,
        stem_type=stem_type,
        se_ratio=se_ratio,
        init_stochastic_depth_rate=init_stochastic_depth_rate)
    inputs = tf_keras.Input(shape=(8, input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual([
        1, 2, input_size / 2**2, input_size / 2**2, 64 * endpoint_filter_scale
    ], endpoints['2'].shape.as_list())
    self.assertAllEqual([
        1, 2, input_size / 2**3, input_size / 2**3, 128 * endpoint_filter_scale
    ], endpoints['3'].shape.as_list())
    self.assertAllEqual([
        1, 2, input_size / 2**4, input_size / 2**4, 256 * endpoint_filter_scale
    ], endpoints['4'].shape.as_list())
    self.assertAllEqual([
        1, 2, input_size / 2**5, input_size / 2**5, 512 * endpoint_filter_scale
    ], endpoints['5'].shape.as_list())
    self.assertAllEqual([
        1, 2, input_size / 2**5, input_size / 2**5, 512 * endpoint_filter_scale
    ], endpoints['5_1'].shape.as_list())

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=50,
        temporal_strides=[1, 1, 1, 1],
        temporal_kernel_sizes=[(3, 3, 3), (3, 1, 3, 1), (3, 1, 3, 1, 3, 1),
                               (1, 3, 1)],
        stem_type='v0',
        stem_conv_temporal_kernel_size=5,
        stem_conv_temporal_stride=2,
        stem_pool_temporal_stride=2,
        se_ratio=0.0,
        use_self_gating=None,
        init_stochastic_depth_rate=0.0,
        use_sync_bn=False,
        activation='relu',
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    network = resnet_3d.ResNet3DY(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = resnet_3d.ResNet3DY.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
