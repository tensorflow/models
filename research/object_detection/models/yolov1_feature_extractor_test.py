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
import yolov1_feature_extractor# import YOLOv1FeatureExtractor

class YOLOFeatureExtractorTest(tf.test.TestCase):

  def _create_feature_extractor(self):
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
        
        
  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    test_image = np.random.rand(4, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor()
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

if __name__ == '__main__':
  tf.test.main()
