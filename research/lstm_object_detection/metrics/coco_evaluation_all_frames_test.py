# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for video_object_detection.metrics.coco_video_evaluation."""

import numpy as np
import tensorflow as tf
from lstm_object_detection.metrics import coco_evaluation_all_frames
from object_detection.core import standard_fields


class CocoEvaluationAllFramesTest(tf.test.TestCase):

  def testGroundtruthAndDetectionsDisagreeOnAllFrames(self):
    """Tests that mAP is calculated on several different frame results."""
    category_list = [{'id': 0, 'name': 'dog'}, {'id': 1, 'name': 'cat'}]
    video_evaluator = coco_evaluation_all_frames.CocoEvaluationAllFrames(
        category_list)
    video_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict=[{
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[50., 50., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
        }, {
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[50., 50., 100., 100.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
        }])
    video_evaluator.add_single_detected_image_info(
        image_id='image1',
        # A different groundtruth box on the frame other than the last one.
        detections_dict=[{
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
        }, {
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[50., 50., 100., 100.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
        }])

    metrics = video_evaluator.evaluate()
    self.assertNotEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

  def testGroundtruthAndDetections(self):
    """Tests that mAP is calculated correctly on GT and Detections."""
    category_list = [{'id': 0, 'name': 'dog'}, {'id': 1, 'name': 'cat'}]
    video_evaluator = coco_evaluation_all_frames.CocoEvaluationAllFrames(
        category_list)
    video_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict=[{
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
        }])
    video_evaluator.add_single_ground_truth_image_info(
        image_id='image2',
        groundtruth_dict=[{
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[50., 50., 100., 100.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
        }])
    video_evaluator.add_single_ground_truth_image_info(
        image_id='image3',
        groundtruth_dict=[{
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[50., 100., 100., 120.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
        }])
    video_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict=[{
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
        }])
    video_evaluator.add_single_detected_image_info(
        image_id='image2',
        detections_dict=[{
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[50., 50., 100., 100.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
        }])
    video_evaluator.add_single_detected_image_info(
        image_id='image3',
        detections_dict=[{
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[50., 100., 100., 120.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
        }])
    metrics = video_evaluator.evaluate()
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

  def testMissingDetectionResults(self):
    """Tests if groundtrue is missing, raises ValueError."""
    category_list = [{'id': 0, 'name': 'dog'}]
    video_evaluator = coco_evaluation_all_frames.CocoEvaluationAllFrames(
        category_list)
    video_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict=[{
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
        }])
    with self.assertRaisesRegexp(ValueError,
                                 r'Missing groundtruth for image-frame id:.*'):
      video_evaluator.add_single_detected_image_info(
          image_id='image3',
          detections_dict=[{
              standard_fields.DetectionResultFields.detection_boxes:
                  np.array([[100., 100., 200., 200.]]),
              standard_fields.DetectionResultFields.detection_scores:
                  np.array([.8]),
              standard_fields.DetectionResultFields.detection_classes:
                  np.array([1])
          }])


if __name__ == '__main__':
  tf.test.main()
