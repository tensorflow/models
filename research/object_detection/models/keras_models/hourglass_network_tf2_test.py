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
"""Testing the Hourglass network."""
import unittest
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models.keras_models import hourglass_network as hourglass
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class HourglassFeatureExtractorTest(tf.test.TestCase, parameterized.TestCase):

  def test_identity_layer(self):

    layer = hourglass.IdentityLayer()
    output = layer(np.zeros((2, 32, 32, 3), dtype=np.float32))
    self.assertEqual(output.shape, (2, 32, 32, 3))

  def test_skip_conv_layer_stride_1(self):

    layer = hourglass.SkipConvolution(out_channels=8, stride=1)
    output = layer(np.zeros((2, 32, 32, 3), dtype=np.float32))
    self.assertEqual(output.shape, (2, 32, 32, 8))

  def test_skip_conv_layer_stride_2(self):

    layer = hourglass.SkipConvolution(out_channels=8, stride=2)
    output = layer(np.zeros((2, 32, 32, 3), dtype=np.float32))
    self.assertEqual(output.shape, (2, 16, 16, 8))

  @parameterized.parameters([{'kernel_size': 1},
                             {'kernel_size': 3},
                             {'kernel_size': 7}])
  def test_conv_block(self, kernel_size):

    layer = hourglass.ConvolutionalBlock(
        out_channels=8, kernel_size=kernel_size, stride=1)
    output = layer(np.zeros((2, 32, 32, 3), dtype=np.float32))
    self.assertEqual(output.shape, (2, 32, 32, 8))

    layer = hourglass.ConvolutionalBlock(
        out_channels=8, kernel_size=kernel_size, stride=2)
    output = layer(np.zeros((2, 32, 32, 3), dtype=np.float32))
    self.assertEqual(output.shape, (2, 16, 16, 8))

  def test_residual_block_stride_1(self):

    layer = hourglass.ResidualBlock(out_channels=8, stride=1)
    output = layer(np.zeros((2, 32, 32, 8), dtype=np.float32))
    self.assertEqual(output.shape, (2, 32, 32, 8))

  def test_residual_block_stride_2(self):

    layer = hourglass.ResidualBlock(out_channels=8, stride=2,
                                    skip_conv=True)
    output = layer(np.zeros((2, 32, 32, 8), dtype=np.float32))
    self.assertEqual(output.shape, (2, 16, 16, 8))

  def test_input_downsample_block(self):

    layer = hourglass.InputDownsampleBlock(
        out_channels_initial_conv=4, out_channels_residual_block=8)
    output = layer(np.zeros((2, 32, 32, 8), dtype=np.float32))
    self.assertEqual(output.shape, (2, 8, 8, 8))

  def test_encoder_decoder_block(self):

    layer = hourglass.EncoderDecoderBlock(
        num_stages=4, blocks_per_stage=[2, 3, 4, 5, 6],
        channel_dims=[4, 6, 8, 10, 12])
    output = layer(np.zeros((2, 64, 64, 4), dtype=np.float32))
    self.assertEqual(output.shape, (2, 64, 64, 4))

  def test_hourglass_feature_extractor(self):

    model = hourglass.HourglassNetwork(
        num_stages=4, blocks_per_stage=[2, 3, 4, 5, 6],
        channel_dims=[4, 6, 8, 10, 12, 14], num_hourglasses=2)
    outputs = model(np.zeros((2, 64, 64, 3), dtype=np.float32))
    self.assertEqual(outputs[0].shape, (2, 16, 16, 6))
    self.assertEqual(outputs[1].shape, (2, 16, 16, 6))


if __name__ == '__main__':
  tf.test.main()
