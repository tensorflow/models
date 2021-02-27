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
from object_detection.models.keras_models import mobilenet_v2
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetMobileNetV2FPNFeatureExtractorTest(test_case.TestCase):

  def test_center_net_mobilenet_v2_fpn_feature_extractor(self):

    net = mobilenet_v2.mobilenet_v2(True, include_top=False)

    model = center_net_mobilenet_v2_fpn_feature_extractor.CenterNetMobileNetV2FPNFeatureExtractor(
        net)

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

    net = mobilenet_v2.mobilenet_v2(True, include_top=False)

    model = center_net_mobilenet_v2_fpn_feature_extractor.CenterNetMobileNetV2FPNFeatureExtractor(
        net, fpn_separable_conv=True)

    def graph_fn():
      img = np.zeros((8, 224, 224, 3), dtype=np.float32)
      processed_img = model.preprocess(img)
      return model(processed_img)

    outputs = self.execute(graph_fn, [])
    self.assertEqual(outputs.shape, (8, 56, 56, 24))

    # Pull out the FPN network.
    output = model.get_layer('model_1')
    for layer in output.layers:
      # Convolution layers with kernel size not equal to (1, 1) should be
      # separable 2D convolutions.
      if 'conv' in layer.name and layer.kernel_size != (1, 1):
        self.assertIsInstance(layer, tf.keras.layers.SeparableConv2D)


if __name__ == '__main__':
  tf.test.main()
