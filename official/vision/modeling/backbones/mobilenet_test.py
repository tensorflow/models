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

"""Tests for MobileNet."""

import itertools
import math

# Import libraries

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.modeling.backbones import mobilenet


class MobileNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      'MobileNetV1',
      'MobileNetV2',
      'MobileNetV3Large',
      'MobileNetV3Small',
      'MobileNetV3EdgeTPU',
      'MobileNetMultiAVG',
      'MobileNetMultiMAX',
      'MobileNetMultiAVGSeg',
      'MobileNetMultiMAXSeg',
      'MobileNetV3SmallReducedFilters',
      'MobileNetV4ConvSmall',
      'MobileNetV4ConvMedium',
      'MobileNetV4ConvLarge',
      'MobileNetV4HybridMedium',
      'MobileNetV4HybridLarge',
      'MobileNetV4ConvMediumSeg',
  )
  def test_serialize_deserialize(self, model_id):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=model_id,
        filter_size_scale=1.0,
        stochastic_depth_drop_rate=None,
        flat_stochastic_depth_drop_rate=True,
        use_sync_bn=False,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        output_stride=None,
        min_depth=8,
        divisible_by=8,
        regularize_depthwise=False,
        finegrain_classification_mode=True,
    )
    network = mobilenet.MobileNet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = mobilenet.MobileNet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

  @parameterized.parameters(
      itertools.product(
          [1, 3],
          [
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
              'MobileNetMultiAVGSeg',
              'MobileNetMultiMAXSeg',
              'MobileNetV3SmallReducedFilters',
              'MobileNetV4ConvSmall',
              'MobileNetV4ConvMedium',
              'MobileNetV4ConvLarge',
              'MobileNetV4HybridMedium',
              'MobileNetV4HybridLarge',
              'MobileNetV4ConvMediumSeg',
          ],
      )
  )
  def test_input_specs(self, input_dim, model_id):
    """Test different input feature dimensions."""
    tf_keras.backend.set_image_data_format('channels_last')

    input_specs = tf_keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = mobilenet.MobileNet(model_id=model_id, input_specs=input_specs)

    inputs = tf_keras.Input(shape=(128, 128, input_dim), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(
      itertools.product(
          [
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
              'MobileNetMultiAVGSeg',
              'MobileNetV3SmallReducedFilters',
              'MobileNetV4ConvSmall',
              'MobileNetV4ConvMedium',
              'MobileNetV4ConvLarge',
              'MobileNetV4HybridMedium',
              'MobileNetV4HybridLarge',
              'MobileNetV4ConvMediumSeg',
          ],
          [32, 224],
      )
  )
  def test_mobilenet_creation(self, model_id,
                              input_size):
    """Test creation of MobileNet family models."""
    tf_keras.backend.set_image_data_format('channels_last')

    mobilenet_layers = {
        # The number of filters of layers having outputs been collected
        # for filter_size_scale = 1.0
        'MobileNetV1': [128, 256, 512, 1024],
        'MobileNetV2': [24, 32, 96, 320],
        'MobileNetV3Small': [16, 24, 48, 96],
        'MobileNetV3Large': [24, 40, 112, 160],
        'MobileNetV3EdgeTPU': [32, 48, 96, 192],
        'MobileNetMultiMAX': [32, 64, 128, 160],
        'MobileNetMultiAVG': [32, 64, 160, 192],
        'MobileNetMultiAVGSeg': [32, 64, 160, 96],
        'MobileNetMultiMAXSeg': [32, 64, 128, 96],
        'MobileNetV3SmallReducedFilters': [16, 24, 48, 48],
        'MobileNetV4ConvSmall': [32, 64, 96, 128],
        'MobileNetV4ConvMedium': [48, 80, 160, 256],
        'MobileNetV4ConvLarge': [48, 96, 192, 512],
        'MobileNetV4HybridMedium': [48, 80, 160, 256],
        'MobileNetV4HybridLarge': [48, 96, 192, 512],
        'MobileNetV4ConvMediumSeg': [48, 80, 160, 448],
    }

    network = mobilenet.MobileNet(model_id=model_id,
                                  filter_size_scale=1.0)

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    for idx, num_filter in enumerate(mobilenet_layers[model_id]):
      self.assertAllEqual(
          [1, input_size / 2 ** (idx+2), input_size / 2 ** (idx+2), num_filter],
          endpoints[str(idx+2)].shape.as_list())

  @parameterized.parameters(
      itertools.product(
          [
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
              'MobileNetMultiAVGSeg',
              'MobileNetMultiMAXSeg',
              'MobileNetV3SmallReducedFilters',
              'MobileNetV4ConvSmall',
              'MobileNetV4ConvMedium',
              'MobileNetV4ConvLarge',
              'MobileNetV4HybridMedium',
              'MobileNetV4HybridLarge',
              'MobileNetV4ConvMediumSeg',
          ],
          [32, 224],
      )
  )
  def test_mobilenet_intermediate_layers(self, model_id, input_size):
    tf_keras.backend.set_image_data_format('channels_last')
    # Tests the mobilenet intermediate depthwise layers.
    mobilenet_depthwise_layers = {
        # The number of filters of depthwise layers having outputs been
        # collected for filter_size_scale = 1.0. Only tests the mobilenet
        # model with inverted bottleneck block using depthwise which excludes
        # MobileNetV1.
        'MobileNetV1': [],
        'MobileNetV2': [144, 192, 576, 960],
        'MobileNetV3Small': [16, 88, 144, 576],
        'MobileNetV3Large': [72, 120, 672, 960],
        'MobileNetV3EdgeTPU': [None, None, 384, 1280],
        'MobileNetMultiMAX': [96, 128, 384, 640],
        'MobileNetMultiAVG': [64, 192, 640, 768],
        'MobileNetMultiAVGSeg': [64, 192, 640, 384],
        'MobileNetMultiMAXSeg': [96, 128, 384, 320],
        'MobileNetV3SmallReducedFilters': [16, 88, 144, 288],
        'MobileNetV4ConvSmall': [None, None, None, None],
        'MobileNetV4ConvMedium': [None, None, None, None],
        'MobileNetV4ConvLarge': [None, None, None, None],
        'MobileNetV4HybridMedium': [None, None, None, None],
        'MobileNetV4HybridLarge': [None, None, None, None],
        'MobileNetV4ConvMediumSeg': [None, None, None, None],
    }
    network = mobilenet.MobileNet(model_id=model_id,
                                  filter_size_scale=1.0,
                                  output_intermediate_endpoints=True)

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    for idx, num_filter in enumerate(mobilenet_depthwise_layers[model_id]):
      # Not using depthwise conv in this layer.
      if num_filter is None:
        continue

      self.assertAllEqual(
          [1, input_size / 2**(idx + 2), input_size / 2**(idx + 2), num_filter],
          endpoints[str(idx + 2) + '/depthwise'].shape.as_list())

  @parameterized.parameters(
      itertools.product(
          [
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
              'MobileNetMultiMAX',
              'MobileNetMultiAVGSeg',
              'MobileNetMultiMAXSeg',
              'MobileNetV3SmallReducedFilters',
              'MobileNetV4ConvSmall',
              'MobileNetV4ConvMedium',
              'MobileNetV4ConvLarge',
              'MobileNetV4HybridMedium',
              'MobileNetV4HybridLarge',
              'MobileNetV4ConvMediumSeg',
          ],
          [1.0, 0.75],
      )
  )
  def test_mobilenet_scaling(self, model_id,
                             filter_size_scale):
    """Test for creation of a MobileNet classifier."""
    mobilenet_params = {
        ('MobileNetV1', 1.0): 3228864,
        ('MobileNetV1', 0.75): 1832976,
        ('MobileNetV2', 1.0): 2257984,
        ('MobileNetV2', 0.75): 1382064,
        ('MobileNetV3Large', 1.0): 4226432,
        ('MobileNetV3Large', 0.75): 2731616,
        ('MobileNetV3Small', 1.0): 1529968,
        ('MobileNetV3Small', 0.75): 1026552,
        ('MobileNetV3EdgeTPU', 1.0): 2849312,
        ('MobileNetV3EdgeTPU', 0.75): 1737288,
        ('MobileNetMultiAVG', 1.0): 3704416,
        ('MobileNetMultiAVG', 0.75): 2349704,
        ('MobileNetMultiMAX', 1.0): 3174560,
        ('MobileNetMultiMAX', 0.75): 2045816,
        ('MobileNetMultiAVGSeg', 1.0): 2239840,
        ('MobileNetMultiAVGSeg', 0.75): 1395272,
        ('MobileNetMultiMAXSeg', 1.0): 1929088,
        ('MobileNetMultiMAXSeg', 0.75): 1216544,
        ('MobileNetV3SmallReducedFilters', 1.0): 694880,
        ('MobileNetV3SmallReducedFilters', 0.75): 505960,
        ('MobileNetV4ConvSmall', 1.0): 2518112,
        ('MobileNetV4ConvSmall', 0.75): 1670408,
        ('MobileNetV4ConvMedium', 1.0): 8502416,
        ('MobileNetV4ConvMedium', 0.75): 5096424,
        ('MobileNetV4ConvLarge', 1.0): 31459416,
        ('MobileNetV4ConvLarge', 0.75): 18099824,
        ('MobileNetV4HybridMedium', 1.0): 9869488,
        ('MobileNetV4HybridMedium', 0.75): 6072584,
        ('MobileNetV4HybridLarge', 1.0): 36648024,
        ('MobileNetV4HybridLarge', 0.75): 21598064,
        ('MobileNetV4ConvMediumSeg', 1.0): 3787024,
        ('MobileNetV4ConvMediumSeg', 0.75): 2302536,
    }

    input_size = 224
    network = mobilenet.MobileNet(model_id=model_id,
                                  filter_size_scale=filter_size_scale)
    self.assertEqual(network.count_params(),
                     mobilenet_params[(model_id, filter_size_scale)])

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(
      itertools.product(
          [
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
              'MobileNetMultiAVGSeg',
              'MobileNetMultiMAXSeg',
              'MobileNetV3SmallReducedFilters',
              'MobileNetV4ConvSmall',
              'MobileNetV4ConvMedium',
              'MobileNetV4ConvLarge',
              'MobileNetV4HybridMedium',
              'MobileNetV4HybridLarge',
              'MobileNetV4ConvMediumSeg',
          ],
          [8, 16, 32],
      )
  )
  def test_mobilenet_output_stride(self, model_id, output_stride):
    """Test for creation of a MobileNet with different output strides."""
    tf_keras.backend.set_image_data_format('channels_last')

    mobilenet_layers = {
        # The number of filters of the layers outputs been collected
        # for filter_size_scale = 1.0.
        'MobileNetV1': 1024,
        'MobileNetV2': 320,
        'MobileNetV3Small': 96,
        'MobileNetV3Large': 160,
        'MobileNetV3EdgeTPU': 192,
        'MobileNetMultiMAX': 160,
        'MobileNetMultiAVG': 192,
        'MobileNetMultiAVGSeg': 448,
        'MobileNetMultiMAXSeg': 448,
        'MobileNetV3SmallReducedFilters': 48,
        'MobileNetV4ConvSmall': 128,
        'MobileNetV4ConvMedium': 256,
        'MobileNetV4ConvLarge': 512,
        'MobileNetV4HybridMedium': 256,
        'MobileNetV4HybridLarge': 512,
        'MobileNetV4ConvMediumSeg': 448,
    }

    network = mobilenet.MobileNet(
        model_id=model_id, filter_size_scale=1.0, output_stride=output_stride)
    level = int(math.log2(output_stride))
    input_size = 224

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)
    num_filter = mobilenet_layers[model_id]
    self.assertAllEqual(
        [1, input_size / output_stride, input_size / output_stride, num_filter],
        endpoints[str(level)].shape.as_list())


if __name__ == '__main__':
  tf.test.main()
