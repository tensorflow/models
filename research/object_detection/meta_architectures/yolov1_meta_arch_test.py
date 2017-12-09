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

import functools
import numpy as np
import tensorflow as tf

from tensorflow.python.training import saver as tf_saver
from object_detection.core import anchor_generator
from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.meta_architectures import yolov1_meta_arch
from object_detection.models import yolov1_feature_extractor
from object_detection.utils import test_utils

slim = tf.contrib.slim

# All unit test functions start with prefix 'test'

class YOLOMetaArchTest(tf.test.TestCase):
  def setUp(self):
    """
       Here we set up a simple mock YOLOv1 model
    """
    is_training = False
    self._num_classes = 20
    feature_extractor = yolov1_feature_extractor.YOLOv1FeatureExtractor(is_training)
    mock_matcher = test_utils.MockMatcher()
    region_similarity_calculator = sim_calc.IouSimilarity()

    def image_resizer_fn(image):
      return tf.identity(image)

    non_max_suppression_fn = functools.partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=-20.0,
      iou_thresh=1.0,
      max_size_per_class=5,
      max_total_size=5)

    score_conversion_fn = tf.identity

    localization_loss_weight = 5.0
    noobject_loss_weight = 0.5

    self._model = yolov1_meta_arch.YOLOMetaArch(
      is_training, feature_extractor, mock_matcher, self._num_classes,
      region_similarity_calculator, image_resizer_fn, non_max_suppression_fn,
      score_conversion_fn, localization_loss_weight, noobject_loss_weight)

  def test_preprocess_preserves_input_shapes(self):
    image_shapes = [(3, None, None, 3),
                    (None, 10, 10, 3),
                    (None, None, None, 3),
                    (5, 5, 5, 5)]
    for image_shape in image_shapes:
      image_placeholder = tf.placeholder(tf.float32, shape=image_shape)
      preprocessed_inputs = self._model.preprocess(image_placeholder)
      self.assertAllEqual(preprocessed_inputs.shape.as_list(), image_shape)

  def test_predict_results_have_correct_keys_and_shapes(self):
    pass

  def test_postprocess_results_are_correct(self):
    pass

  def test_loss_results_are_correct(self):
    pass

  def test_restore_map_for_detection_ckpt(self):
    pass

  def test_restore_map_for_classification_ckpt(self):
    pass

if __name__ == '__main__':
  tf.test.main()