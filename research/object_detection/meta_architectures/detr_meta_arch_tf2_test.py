# Lint as: python2, python3
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

"""Tests for object_detection.meta_architectures.faster_rcnn_meta_arch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
import functools

from object_detection.meta_architectures import detr_meta_arch
from object_detection.models import detr_resnet_keras_feature_extractor as detr_feature_extractor
from object_detection.utils import test_utils
from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner as cn_assigner
from object_detection.meta_architectures import center_net_meta_arch as cnma
from object_detection.models import center_net_resnet_feature_extractor
from object_detection.utils import test_case
from object_detection.utils import tf_version
from object_detection.utils import ops

_NUM_CLASSES = 12
_NUM_QUERIES = 10

def get_fake_logits(batch_size, num_queries, num_classes, val=1):
  if (num_classes == 0):
    num_classes = _NUM_CLASSES
  if (num_queries == 0):
    num_queries = _NUM_QUERIES
  fake_logits = np.zeros((batch_size, num_queries, num_classes + 1))
  fake_logits[:,:,1] = val
  return tf.convert_to_tensor(fake_logits, dtype=tf.float32)

def build_detr_meta_arch(num_queries=0, num_classes=0, build_resnet=False):
  """Builds the DETR meta architecture."""
  if (num_classes == 0):
    num_classes = _NUM_CLASSES
  if (num_queries == 0):
    num_queries = _NUM_QUERIES
  feature_extractor = (detr_feature_extractor.
      DETRResnet50KerasFeatureExtractor(False))
  image_resizer_fn = functools.partial(
      preprocessor.resize_to_range,
      min_dimension=128,
      max_dimension=128,
      pad_to_max_dimension=True)
  return detr_meta_arch.DETRMetaArch(
    True,
    num_classes,
    image_resizer_fn,
    feature_extractor,
    1,
    1,
    1,
    tf.nn.softmax,
    num_queries,
    32,
    False)

class DETRMetaArchShapesTest(test_case.TestCase, tf.test.TestCase,
                             parameterized.TestCase):

  def test_inference_shapes(self):
    batch_size = 2
    height = 10
    width = 12
    input_image_shape = (batch_size, height, width, 3)

    model = build_detr_meta_arch()
    def graph_fn(images):
      """Function to construct tf graph for the test."""

      preprocessed_inputs, true_image_shapes = model.preprocess(images)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      return (prediction_dict['box_encodings'],
              prediction_dict['class_predictions_with_background'])

    images = np.zeros(input_image_shape, dtype=np.float32)

    expected_output_shapes = {
        'box_encodings': (batch_size, 10, 4),
        'class_predictions_with_background': (
            batch_size, _NUM_QUERIES, _NUM_CLASSES + 1)
    }

    results = self.execute(graph_fn, [images])

    print(results[0].shape)
    print(results[1].shape)

    self.assertAllEqual(results[0].shape,
                        expected_output_shapes['box_encodings'])
    self.assertAllEqual(
        results[1].shape,
        expected_output_shapes['class_predictions_with_background'])

  def test_zero_loss(self):
    batch_size = 2
    
    predicted_boxes = tf.ones([batch_size, _NUM_QUERIES, 4], dtype=tf.float32)
    predicted_classes = get_fake_logits(batch_size, _NUM_QUERIES, _NUM_CLASSES, val=1e10)
    groundtruth_boxes = tf.reshape(
        ops.center_to_corner_coordinate(tf.ones([batch_size * _NUM_QUERIES, 4],
                                        dtype=tf.float32)),
        [batch_size, _NUM_QUERIES, 4])
    groundtruth_logits = get_fake_logits(batch_size, _NUM_QUERIES, _NUM_CLASSES)
    groundtruth_weights = [tf.ones([_NUM_QUERIES]) for i in range(batch_size)]

    model = build_detr_meta_arch()
    losses = model._loss_box_classifier(predicted_boxes,
                                        predicted_classes,
                                        groundtruth_boxes,
                                        groundtruth_logits,
                                        groundtruth_weights)

    self.assertEqual(losses['Loss/classification_loss'], 0.0)
    self.assertEqual(losses['Loss/localization_loss'], 0.0)

  def test_postprocess(self):
    # Smoke test
    batch_size = 1

    predicted_boxes = tf.constant([[[0.5, 0.5, 0.2, 0.2]]], dtype=tf.float32)
    predicted_classes = get_fake_logits(batch_size, 1, _NUM_CLASSES, val=1e10)
    preprocess_shapes = tf.constant([batch_size, 200, 200, 3])
    shapes = tf.repeat(tf.expand_dims(tf.constant([200, 200, 3]), axis=0),
                       batch_size, axis=0)

    model = build_detr_meta_arch(1, _NUM_CLASSES)
    postprocessed = model._postprocess_box_classifier(predicted_boxes,
                                                      predicted_classes,
                                                      shapes,
                                                      preprocess_shapes)

    self.assertAllClose(
        postprocessed[fields.DetectionResultFields.detection_boxes],
        tf.constant([[[0.4, 0.4, 0.6, 0.6]]]))

    self.assertAllClose(
        postprocessed[fields.DetectionResultFields.detection_classes],
        tf.constant([[0]]))

    self.assertAllClose(
        postprocessed[fields.DetectionResultFields.detection_scores],
        tf.constant([[1]]))

    self.assertAllClose(
        postprocessed[fields.DetectionResultFields.num_detections],
        tf.ones([1]))

if __name__ == '__main__':
  tf.test.main()