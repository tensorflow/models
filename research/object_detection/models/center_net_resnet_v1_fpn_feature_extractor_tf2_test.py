# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Testing ResNet v1 FPN models for the CenterNet meta architecture."""
import unittest
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import center_net_resnet_v1_fpn_feature_extractor
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetResnetV1FpnFeatureExtractorTest(test_case.TestCase,
                                               parameterized.TestCase):

  @parameterized.parameters(
      {'resnet_type': 'resnet_v1_50'},
      {'resnet_type': 'resnet_v1_101'},
      {'resnet_type': 'resnet_v1_18'},
      {'resnet_type': 'resnet_v1_34'},
  )
  def test_correct_output_size(self, resnet_type):
    """Verify that shape of features returned by the backbone is correct."""

    model = center_net_resnet_v1_fpn_feature_extractor.\
                CenterNetResnetV1FpnFeatureExtractor(resnet_type)
    def graph_fn():
      img = np.zeros((8, 512, 512, 3), dtype=np.float32)
      processed_img = model.preprocess(img)
      return model(processed_img)

    self.assertEqual(self.execute(graph_fn, []).shape, (8, 128, 128, 64))


if __name__ == '__main__':
  tf.test.main()
