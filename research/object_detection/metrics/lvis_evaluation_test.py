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
"""Tests for tensorflow_models.object_detection.metrics.coco_evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow.compat.v1 as tf
from object_detection.core import standard_fields as fields
from object_detection.metrics import lvis_evaluation
from object_detection.utils import tf_version


def _get_categories_list():
  return [{
      'id': 1,
      'name': 'person',
      'frequency': 'f'
  }, {
      'id': 2,
      'name': 'dog',
      'frequency': 'c'
  }, {
      'id': 3,
      'name': 'cat',
      'frequency': 'r'
  }]


class LvisMaskEvaluationTest(tf.test.TestCase):

  def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
    """Tests that mAP is calculated correctly on GT and Detections."""
    masks1 = np.expand_dims(np.pad(
        np.ones([100, 100], dtype=np.uint8),
        ((100, 56), (100, 56)), mode='constant'), axis=0)
    masks2 = np.expand_dims(np.pad(
        np.ones([50, 50], dtype=np.uint8),
        ((50, 156), (50, 156)), mode='constant'), axis=0)
    masks3 = np.expand_dims(np.pad(
        np.ones([25, 25], dtype=np.uint8),
        ((25, 206), (25, 206)), mode='constant'), axis=0)

    lvis_evaluator = lvis_evaluation.LVISMaskEvaluator(
        _get_categories_list())
    lvis_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            fields.InputDataFields.groundtruth_classes: np.array([1]),
            fields.InputDataFields.groundtruth_instance_masks: masks1,
            fields.InputDataFields.groundtruth_verified_neg_classes:
                np.array([0, 0, 0]),
            fields.InputDataFields.groundtruth_not_exhaustive_classes:
                np.array([0, 0, 0])
        })
    lvis_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            fields.DetectionResultFields.detection_masks: masks1,
            fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            fields.DetectionResultFields.detection_classes:
            np.array([1])
        })
    lvis_evaluator.add_single_ground_truth_image_info(
        image_id='image2',
        groundtruth_dict={
            fields.InputDataFields.groundtruth_boxes:
            np.array([[50., 50., 100., 100.]]),
            fields.InputDataFields.groundtruth_classes: np.array([1]),
            fields.InputDataFields.groundtruth_instance_masks: masks2,
            fields.InputDataFields.groundtruth_verified_neg_classes:
                np.array([0, 0, 0]),
            fields.InputDataFields.groundtruth_not_exhaustive_classes:
                np.array([0, 0, 0])
        })
    lvis_evaluator.add_single_detected_image_info(
        image_id='image2',
        detections_dict={
            fields.DetectionResultFields.detection_masks: masks2,
            fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            fields.DetectionResultFields.detection_classes:
            np.array([1])
        })
    lvis_evaluator.add_single_ground_truth_image_info(
        image_id='image3',
        groundtruth_dict={
            fields.InputDataFields.groundtruth_boxes:
            np.array([[25., 25., 50., 50.]]),
            fields.InputDataFields.groundtruth_classes: np.array([1]),
            fields.InputDataFields.groundtruth_instance_masks: masks3,
            fields.InputDataFields.groundtruth_verified_neg_classes:
                np.array([0, 0, 0]),
            fields.InputDataFields.groundtruth_not_exhaustive_classes:
                np.array([0, 0, 0])
        })
    lvis_evaluator.add_single_detected_image_info(
        image_id='image3',
        detections_dict={
            fields.DetectionResultFields.detection_masks: masks3,
            fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            fields.DetectionResultFields.detection_classes:
            np.array([1])
        })
    metrics = lvis_evaluator.evaluate()
    self.assertAlmostEqual(metrics['DetectionMasks_AP'], 1.0)


@unittest.skipIf(tf_version.is_tf1(), 'Only Supported in TF2.X')
class LVISMaskEvaluationPyFuncTest(tf.test.TestCase):

  def testAddEvalDict(self):
    lvis_evaluator = lvis_evaluation.LVISMaskEvaluator(_get_categories_list())
    image_id = tf.constant('image1', dtype=tf.string)
    groundtruth_boxes = tf.constant(
        np.array([[100., 100., 200., 200.], [50., 50., 100., 100.]]),
        dtype=tf.float32)
    groundtruth_classes = tf.constant(np.array([1, 2]), dtype=tf.float32)
    groundtruth_masks = tf.constant(np.stack([
        np.pad(np.ones([100, 100], dtype=np.uint8), ((10, 10), (10, 10)),
               mode='constant'),
        np.pad(np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
               mode='constant')
    ]), dtype=tf.uint8)
    original_image_spatial_shapes = tf.constant([[120, 120], [120, 120]],
                                                dtype=tf.int32)
    groundtruth_verified_neg_classes = tf.constant(np.array([0, 0, 0]),
                                                   dtype=tf.float32)
    groundtruth_not_exhaustive_classes = tf.constant(np.array([0, 0, 0]),
                                                     dtype=tf.float32)
    detection_scores = tf.constant(np.array([.9, .8]), dtype=tf.float32)
    detection_classes = tf.constant(np.array([2, 1]), dtype=tf.float32)
    detection_masks = tf.constant(np.stack([
        np.pad(np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
               mode='constant'),
        np.pad(np.ones([100, 100], dtype=np.uint8), ((10, 10), (10, 10)),
               mode='constant'),
    ]), dtype=tf.uint8)

    input_data_fields = fields.InputDataFields
    detection_fields = fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        input_data_fields.groundtruth_verified_neg_classes:
            groundtruth_verified_neg_classes,
        input_data_fields.groundtruth_not_exhaustive_classes:
            groundtruth_not_exhaustive_classes,
        input_data_fields.original_image_spatial_shape:
            original_image_spatial_shapes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks
    }
    lvis_evaluator.add_eval_dict(eval_dict)
    self.assertLen(lvis_evaluator._groundtruth_list, 2)
    self.assertLen(lvis_evaluator._detection_masks_list, 2)


if __name__ == '__main__':
  tf.test.main()
