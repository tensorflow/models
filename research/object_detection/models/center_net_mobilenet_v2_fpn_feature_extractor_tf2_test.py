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


if __name__ == '__main__':
  tf.test.main()
