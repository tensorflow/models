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
"""Tests for google3.third_party.tensorflow_models.object_detection.core.class_agnostic_nms."""
import tensorflow as tf
from object_detection.core import post_processing
from object_detection.core import standard_fields as fields
from object_detection.utils import test_case


class ClassAgnosticNonMaxSuppressionTest(test_case.TestCase):

  def test_class_agnostic_nms_select_with_shared_boxes(self):
    boxes = tf.constant(
        [[[0, 0, 1, 1]], [[0, 0.1, 1, 1.1]], [[0, -0.1, 1, 0.9]],
         [[0, 10, 1, 11]], [[0, 10.1, 1, 11.1]], [[0, 100, 1, 101]],
         [[0, 1000, 1, 1002]], [[0, 1000, 1, 1002.1]]], tf.float32)
    scores = tf.constant([[.9, 0.01], [.75, 0.05], [.6, 0.01], [.95, 0],
                          [.5, 0.01], [.3, 0.01], [.01, .85], [.01, .5]])
    score_thresh = 0.1
    iou_thresh = .5
    max_classes_per_detection = 1
    max_output_size = 4

    exp_nms_corners = [[0, 10, 1, 11], [0, 0, 1, 1], [0, 1000, 1, 1002],
                       [0, 100, 1, 101]]
    exp_nms_scores = [.95, .9, .85, .3]
    exp_nms_classes = [0, 0, 1, 0]

    nms, _ = post_processing.class_agnostic_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh, max_classes_per_detection,
        max_output_size)

    with self.test_session() as sess:
      nms_corners_output, nms_scores_output, nms_classes_output = sess.run([
          nms.get(),
          nms.get_field(fields.BoxListFields.scores),
          nms.get_field(fields.BoxListFields.classes)
      ])

      self.assertAllClose(nms_corners_output, exp_nms_corners)
      self.assertAllClose(nms_scores_output, exp_nms_scores)
      self.assertAllClose(nms_classes_output, exp_nms_classes)

  def test_class_agnostic_nms_select_with_per_class_boxes(self):
    boxes = tf.constant(
        [[[4, 5, 9, 10], [0, 0, 1, 1]],
         [[0, 0.1, 1, 1.1], [4, 5, 9, 10]],
         [[0, -0.1, 1, 0.9], [4, 5, 9, 10]],
         [[0, 10, 1, 11], [4, 5, 9, 10]],
         [[0, 10.1, 1, 11.1], [4, 5, 9, 10]],
         [[0, 100, 1, 101], [4, 5, 9, 10]],
         [[4, 5, 9, 10], [0, 1000, 1, 1002]],
         [[4, 5, 9, 10], [0, 1000, 1, 1002.1]]], tf.float32)
    scores = tf.constant([[.01, 0.9],
                          [.75, 0.05],
                          [.6, 0.01],
                          [.95, 0],
                          [.5, 0.01],
                          [.3, 0.01],
                          [.01, .85],
                          [.01, .5]])
    score_thresh = 0.1
    iou_thresh = .5
    max_classes_per_detection = 1
    max_output_size = 4

    exp_nms_corners = [[0, 10, 1, 11],
                       [0, 0, 1, 1],
                       [0, 1000, 1, 1002],
                       [0, 100, 1, 101]]
    exp_nms_scores = [.95, .9, .85, .3]
    exp_nms_classes = [0, 1, 1, 0]

    nms, _ = post_processing.class_agnostic_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh, max_classes_per_detection,
        max_output_size)

    with self.test_session() as sess:
      nms_corners_output, nms_scores_output, nms_classes_output = sess.run([
          nms.get(),
          nms.get_field(fields.BoxListFields.scores),
          nms.get_field(fields.BoxListFields.classes)
      ])

      self.assertAllClose(nms_corners_output, exp_nms_corners)
      self.assertAllClose(nms_scores_output, exp_nms_scores)
      self.assertAllClose(nms_classes_output, exp_nms_classes)

  def test_batch_classagnostic_nms_with_batch_size_1(self):
    boxes = tf.constant(
        [[[[0, 0, 1, 1]], [[0, 0.1, 1, 1.1]], [[0, -0.1, 1, 0.9]],
          [[0, 10, 1, 11]], [[0, 10.1, 1, 11.1]], [[0, 100, 1, 101]],
          [[0, 1000, 1, 1002]], [[0, 1000, 1, 1002.1]]]], tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05], [.6, 0.01], [.95, 0],
                           [.5, 0.01], [.3, 0.01], [.01, .85], [.01, .5]]])
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4
    max_classes_per_detection = 1
    use_class_agnostic_nms = True

    exp_nms_corners = [[[0, 10, 1, 11], [0, 0, 1, 1], [0, 1000, 1, 1002],
                        [0, 100, 1, 101]]]
    exp_nms_scores = [[.95, .9, .85, .3]]
    exp_nms_classes = [[0, 0, 1, 0]]

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields,
     num_detections) = post_processing.batch_multiclass_non_max_suppression(
         boxes,
         scores,
         score_thresh,
         iou_thresh,
         max_size_per_class=max_output_size,
         max_total_size=max_output_size,
         use_class_agnostic_nms=use_class_agnostic_nms,
         max_classes_per_detection=max_classes_per_detection)

    self.assertIsNone(nmsed_masks)
    self.assertIsNone(nmsed_additional_fields)

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = sess.run(
          [nmsed_boxes, nmsed_scores, nmsed_classes, num_detections])
      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      self.assertEqual(num_detections, [4])


if __name__ == '__main__':
  tf.test.main()
