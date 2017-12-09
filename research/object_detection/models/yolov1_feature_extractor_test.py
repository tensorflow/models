# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Base test class YOLOFeatureExtractors."""

from abc import abstractmethod

import numpy as np
import tensorflow as tf


class YOLOFeatureExtractorTest(tf.test.TestCase):

  def _create_feature_extractor(self, depth_multiplier):
    """Constructs a YOLOFeatureExtractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
    Returns:
      an yolov1_feature_extractor.YOLOv1FeatureExtractor
    """
    is_training = False
    reuse_weights = None
    return yolov1_feature_extractor.YOLOv1FeatureExtractor(
        is_training , reuse_weights)

if __name__ == '__main__':
  tf.test.main()
