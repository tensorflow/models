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
"""Testing mobilenet_v2+FPN feature extractor for CenterNet."""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import center_net_mobilenet_v2_fpn_feature_extractor
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetMobileNetV2FPNFeatureExtractorTest(test_case.TestCase):

  def test_center_net_mobilenet_v2_fpn_feature_extractor(self):

    channel_means = (0., 0., 0.)
    channel_stds = (1., 1., 1.)
    bgr_ordering = False
    model = (
        center_net_mobilenet_v2_fpn_feature_extractor.mobilenet_v2_fpn(
            channel_means, channel_stds, bgr_ordering,
            use_separable_conv=False))

    def graph_fn():
      img = np.zeros((8, 224, 224, 3), dtype=np.float32)
      processed_img = model.preprocess(img)
      return model(processed_img)

    outputs = self.execute(graph_fn, [])
    self.assertEqual(outputs.shape, (8, 56, 56, 24))

    # Pull out the FPN network.
    output = model.get_layer('model_1')
    for layer in output.layers:
      # All convolution layers should be normal 2D convolutions.
      if 'conv' in layer.name:
        self.assertIsInstance(layer, tf.keras.layers.Conv2D)

  def test_center_net_mobilenet_v2_fpn_feature_extractor_sep_conv(self):

    channel_means = (0., 0., 0.)
    channel_stds = (1., 1., 1.)
    bgr_ordering = False
    model = (
        center_net_mobilenet_v2_fpn_feature_extractor.mobilenet_v2_fpn(
            channel_means, channel_stds, bgr_ordering, use_separable_conv=True))

    def graph_fn():
      img = np.zeros((8, 224, 224, 3), dtype=np.float32)
      processed_img = model.preprocess(img)
      return model(processed_img)

    outputs = self.execute(graph_fn, [])
    self.assertEqual(outputs.shape, (8, 56, 56, 24))
    # Pull out the FPN network.
    backbone = model.get_layer('model')
    first_conv = backbone.get_layer('Conv1')
    self.assertEqual(32, first_conv.filters)

    # Pull out the FPN network.
    output = model.get_layer('model_1')
    for layer in output.layers:
      # Convolution layers with kernel size not equal to (1, 1) should be
      # separable 2D convolutions.
      if 'conv' in layer.name and layer.kernel_size != (1, 1):
        self.assertIsInstance(layer, tf.keras.layers.SeparableConv2D)

  def test_center_net_mobilenet_v2_fpn_feature_extractor_depth_multiplier(self):

    channel_means = (0., 0., 0.)
    channel_stds = (1., 1., 1.)
    bgr_ordering = False
    model = (
        center_net_mobilenet_v2_fpn_feature_extractor.mobilenet_v2_fpn(
            channel_means, channel_stds, bgr_ordering, use_separable_conv=True,
            depth_multiplier=2.0))

    def graph_fn():
      img = np.zeros((8, 224, 224, 3), dtype=np.float32)
      processed_img = model.preprocess(img)
      return model(processed_img)

    outputs = self.execute(graph_fn, [])
    self.assertEqual(outputs.shape, (8, 56, 56, 24))
    # Pull out the FPN network.
    backbone = model.get_layer('model')
    first_conv = backbone.get_layer('Conv1')
    # Note that the first layer typically has 32 filters, but this model has
    # a depth multiplier of 2.
    self.assertEqual(64, first_conv.filters)


if __name__ == '__main__':
  tf.test.main()
