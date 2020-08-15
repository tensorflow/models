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
"""Testing hourglass feature extractor for CenterNet."""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import center_net_hourglass_feature_extractor as hourglass
from object_detection.models.keras_models import hourglass_network
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetHourglassFeatureExtractorTest(test_case.TestCase):

  def test_center_net_hourglass_feature_extractor(self):

    net = hourglass_network.HourglassNetwork(
        num_stages=4, blocks_per_stage=[2, 3, 4, 5, 6],
        channel_dims=[4, 6, 8, 10, 12, 14], num_hourglasses=2)

    model = hourglass.CenterNetHourglassFeatureExtractor(net)
    def graph_fn():
      return model(tf.zeros((2, 64, 64, 3), dtype=np.float32))
    outputs = self.execute(graph_fn, [])
    self.assertEqual(outputs[0].shape, (2, 16, 16, 6))
    self.assertEqual(outputs[1].shape, (2, 16, 16, 6))


if __name__ == '__main__':
  tf.test.main()
