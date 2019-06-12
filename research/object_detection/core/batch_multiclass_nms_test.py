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
"""Tests for google3.third_party.tensorflow_models.object_detection.core.batch_multiclass_nms."""
import numpy as np
import tensorflow as tf
from object_detection.core import post_processing
from object_detection.utils import test_case


class BatchMulticlassNonMaxSuppressionTest(test_case.TestCase):

  def test_batch_multiclass_nms_with_batch_size_1(self):
    boxes = tf.constant([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                          [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                          [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                          [[0, 10, 1, 11], [0, 10, 1, 11]],
                          [[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                          [[0, 100, 1, 101], [0, 100, 1, 101]],
                          [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                          [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                        tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                           [.6, 0.01], [.95, 0],
                           [.5, 0.01], [.3, 0.01],
                           [.01, .85], [.01, .5]]])
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = [[[0, 10, 1, 11],
                        [0, 0, 1, 1],
                        [0, 999, 2, 1004],
                        [0, 100, 1, 101]]]
    exp_nms_scores = [[.95, .9, .85, .3]]
    exp_nms_classes = [[0, 0, 1, 0]]

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size)

    self.assertIsNone(nmsed_masks)
    self.assertIsNone(nmsed_additional_fields)

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   num_detections])
      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      self.assertEqual(num_detections, [4])

  def test_batch_multiclass_nms_with_batch_size_2(self):
    boxes = tf.constant([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                          [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                          [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                          [[0, 10, 1, 11], [0, 10, 1, 11]]],
                         [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                          [[0, 100, 1, 101], [0, 100, 1, 101]],
                          [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                          [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                        tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                           [.6, 0.01], [.95, 0]],
                          [[.5, 0.01], [.3, 0.01],
                           [.01, .85], [.01, .5]]])
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = np.array([[[0, 10, 1, 11],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[0, 999, 2, 1004],
                                 [0, 10.1, 1, 11.1],
                                 [0, 100, 1, 101],
                                 [0, 0, 0, 0]]])
    exp_nms_scores = np.array([[.95, .9, 0, 0],
                               [.85, .5, .3, 0]])
    exp_nms_classes = np.array([[0, 0, 0, 0],
                                [1, 0, 0, 0]])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size)

    self.assertIsNone(nmsed_masks)
    self.assertIsNone(nmsed_additional_fields)
    # Check static shapes
    self.assertAllEqual(nmsed_boxes.shape.as_list(),
                        exp_nms_corners.shape)
    self.assertAllEqual(nmsed_scores.shape.as_list(),
                        exp_nms_scores.shape)
    self.assertAllEqual(nmsed_classes.shape.as_list(),
                        exp_nms_classes.shape)
    self.assertEqual(num_detections.shape.as_list(), [2])

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   num_detections])
      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      self.assertAllClose(num_detections, [2, 3])

  def test_batch_multiclass_nms_with_per_batch_clip_window(self):
    boxes = tf.constant([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                          [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                          [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                          [[0, 10, 1, 11], [0, 10, 1, 11]]],
                         [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                          [[0, 100, 1, 101], [0, 100, 1, 101]],
                          [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                          [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                        tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                           [.6, 0.01], [.95, 0]],
                          [[.5, 0.01], [.3, 0.01],
                           [.01, .85], [.01, .5]]])
    clip_window = tf.constant([0., 0., 200., 200.])
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = np.array([[[0, 10, 1, 11],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[0, 10.1, 1, 11.1],
                                 [0, 100, 1, 101],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]])
    exp_nms_scores = np.array([[.95, .9, 0, 0],
                               [.5, .3, 0, 0]])
    exp_nms_classes = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0]])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size,
        clip_window=clip_window)

    self.assertIsNone(nmsed_masks)
    self.assertIsNone(nmsed_additional_fields)
    # Check static shapes
    self.assertAllEqual(nmsed_boxes.shape.as_list(),
                        exp_nms_corners.shape)
    self.assertAllEqual(nmsed_scores.shape.as_list(),
                        exp_nms_scores.shape)
    self.assertAllEqual(nmsed_classes.shape.as_list(),
                        exp_nms_classes.shape)
    self.assertEqual(num_detections.shape.as_list(), [2])

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   num_detections])
      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      self.assertAllClose(num_detections, [2, 2])

  def test_batch_multiclass_nms_with_per_image_clip_window(self):
    boxes = tf.constant([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                          [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                          [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                          [[0, 10, 1, 11], [0, 10, 1, 11]]],
                         [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                          [[0, 100, 1, 101], [0, 100, 1, 101]],
                          [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                          [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                        tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                           [.6, 0.01], [.95, 0]],
                          [[.5, 0.01], [.3, 0.01],
                           [.01, .85], [.01, .5]]])
    clip_window = tf.constant([[0., 0., 5., 5.],
                               [0., 0., 200., 200.]])
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = np.array([[[0, 0, 1, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[0, 10.1, 1, 11.1],
                                 [0, 100, 1, 101],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]])
    exp_nms_scores = np.array([[.9, 0., 0., 0.],
                               [.5, .3, 0, 0]])
    exp_nms_classes = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0]])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size,
        clip_window=clip_window)

    self.assertIsNone(nmsed_masks)
    self.assertIsNone(nmsed_additional_fields)
    # Check static shapes
    self.assertAllEqual(nmsed_boxes.shape.as_list(),
                        exp_nms_corners.shape)
    self.assertAllEqual(nmsed_scores.shape.as_list(),
                        exp_nms_scores.shape)
    self.assertAllEqual(nmsed_classes.shape.as_list(),
                        exp_nms_classes.shape)
    self.assertEqual(num_detections.shape.as_list(), [2])

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   num_detections])
      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      self.assertAllClose(num_detections, [1, 2])

  def test_batch_multiclass_nms_with_masks(self):
    boxes = tf.constant([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                          [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                          [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                          [[0, 10, 1, 11], [0, 10, 1, 11]]],
                         [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                          [[0, 100, 1, 101], [0, 100, 1, 101]],
                          [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                          [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                        tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                           [.6, 0.01], [.95, 0]],
                          [[.5, 0.01], [.3, 0.01],
                           [.01, .85], [.01, .5]]])
    masks = tf.constant([[[[[0, 1], [2, 3]], [[1, 2], [3, 4]]],
                          [[[2, 3], [4, 5]], [[3, 4], [5, 6]]],
                          [[[4, 5], [6, 7]], [[5, 6], [7, 8]]],
                          [[[6, 7], [8, 9]], [[7, 8], [9, 10]]]],
                         [[[[8, 9], [10, 11]], [[9, 10], [11, 12]]],
                          [[[10, 11], [12, 13]], [[11, 12], [13, 14]]],
                          [[[12, 13], [14, 15]], [[13, 14], [15, 16]]],
                          [[[14, 15], [16, 17]], [[15, 16], [17, 18]]]]],
                        tf.float32)
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = np.array([[[0, 10, 1, 11],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[0, 999, 2, 1004],
                                 [0, 10.1, 1, 11.1],
                                 [0, 100, 1, 101],
                                 [0, 0, 0, 0]]])
    exp_nms_scores = np.array([[.95, .9, 0, 0],
                               [.85, .5, .3, 0]])
    exp_nms_classes = np.array([[0, 0, 0, 0],
                                [1, 0, 0, 0]])
    exp_nms_masks = np.array([[[[6, 7], [8, 9]],
                               [[0, 1], [2, 3]],
                               [[0, 0], [0, 0]],
                               [[0, 0], [0, 0]]],
                              [[[13, 14], [15, 16]],
                               [[8, 9], [10, 11]],
                               [[10, 11], [12, 13]],
                               [[0, 0], [0, 0]]]])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size,
        masks=masks)

    self.assertIsNone(nmsed_additional_fields)
    # Check static shapes
    self.assertAllEqual(nmsed_boxes.shape.as_list(), exp_nms_corners.shape)
    self.assertAllEqual(nmsed_scores.shape.as_list(), exp_nms_scores.shape)
    self.assertAllEqual(nmsed_classes.shape.as_list(), exp_nms_classes.shape)
    self.assertAllEqual(nmsed_masks.shape.as_list(), exp_nms_masks.shape)
    self.assertEqual(num_detections.shape.as_list(), [2])

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   nmsed_masks, num_detections])

      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      self.assertAllClose(num_detections, [2, 3])
      self.assertAllClose(nmsed_masks, exp_nms_masks)

  def test_batch_multiclass_nms_with_additional_fields(self):
    boxes = tf.constant([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                          [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                          [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                          [[0, 10, 1, 11], [0, 10, 1, 11]]],
                         [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                          [[0, 100, 1, 101], [0, 100, 1, 101]],
                          [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                          [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                        tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                           [.6, 0.01], [.95, 0]],
                          [[.5, 0.01], [.3, 0.01],
                           [.01, .85], [.01, .5]]])
    additional_fields = {
        'keypoints': tf.constant(
            [[[[6, 7], [8, 9]],
              [[0, 1], [2, 3]],
              [[0, 0], [0, 0]],
              [[0, 0], [0, 0]]],
             [[[13, 14], [15, 16]],
              [[8, 9], [10, 11]],
              [[10, 11], [12, 13]],
              [[0, 0], [0, 0]]]],
            tf.float32)
    }
    additional_fields['size'] = tf.constant(
        [[[[6], [8]], [[0], [2]], [[0], [0]], [[0], [0]]],
         [[[13], [15]], [[8], [10]], [[10], [12]], [[0], [0]]]], tf.float32)
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = np.array([[[0, 10, 1, 11],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[0, 999, 2, 1004],
                                 [0, 10.1, 1, 11.1],
                                 [0, 100, 1, 101],
                                 [0, 0, 0, 0]]])
    exp_nms_scores = np.array([[.95, .9, 0, 0],
                               [.85, .5, .3, 0]])
    exp_nms_classes = np.array([[0, 0, 0, 0],
                                [1, 0, 0, 0]])
    exp_nms_additional_fields = {
        'keypoints': np.array([[[[0, 0], [0, 0]],
                                [[6, 7], [8, 9]],
                                [[0, 0], [0, 0]],
                                [[0, 0], [0, 0]]],
                               [[[10, 11], [12, 13]],
                                [[13, 14], [15, 16]],
                                [[8, 9], [10, 11]],
                                [[0, 0], [0, 0]]]])
    }
    exp_nms_additional_fields['size'] = np.array([[[[0], [0]], [[6], [8]],
                                                   [[0], [0]], [[0], [0]]],
                                                  [[[10], [12]], [[13], [15]],
                                                   [[8], [10]], [[0], [0]]]])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size,
        additional_fields=additional_fields)

    self.assertIsNone(nmsed_masks)
    # Check static shapes
    self.assertAllEqual(nmsed_boxes.shape.as_list(), exp_nms_corners.shape)
    self.assertAllEqual(nmsed_scores.shape.as_list(), exp_nms_scores.shape)
    self.assertAllEqual(nmsed_classes.shape.as_list(), exp_nms_classes.shape)
    self.assertEqual(len(nmsed_additional_fields),
                     len(exp_nms_additional_fields))
    for key in exp_nms_additional_fields:
      self.assertAllEqual(nmsed_additional_fields[key].shape.as_list(),
                          exp_nms_additional_fields[key].shape)
    self.assertEqual(num_detections.shape.as_list(), [2])

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_additional_fields,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   nmsed_additional_fields, num_detections])

      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      for key in exp_nms_additional_fields:
        self.assertAllClose(nmsed_additional_fields[key],
                            exp_nms_additional_fields[key])
      self.assertAllClose(num_detections, [2, 3])

  def test_batch_multiclass_nms_with_dynamic_batch_size(self):
    boxes_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2, 4))
    scores_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))
    masks_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2, 2, 2))

    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]])
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]])
    masks = np.array([[[[[0, 1], [2, 3]], [[1, 2], [3, 4]]],
                       [[[2, 3], [4, 5]], [[3, 4], [5, 6]]],
                       [[[4, 5], [6, 7]], [[5, 6], [7, 8]]],
                       [[[6, 7], [8, 9]], [[7, 8], [9, 10]]]],
                      [[[[8, 9], [10, 11]], [[9, 10], [11, 12]]],
                       [[[10, 11], [12, 13]], [[11, 12], [13, 14]]],
                       [[[12, 13], [14, 15]], [[13, 14], [15, 16]]],
                       [[[14, 15], [16, 17]], [[15, 16], [17, 18]]]]])
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = np.array([[[0, 10, 1, 11],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[0, 999, 2, 1004],
                                 [0, 10.1, 1, 11.1],
                                 [0, 100, 1, 101],
                                 [0, 0, 0, 0]]])
    exp_nms_scores = np.array([[.95, .9, 0, 0],
                               [.85, .5, .3, 0]])
    exp_nms_classes = np.array([[0, 0, 0, 0],
                                [1, 0, 0, 0]])
    exp_nms_masks = np.array([[[[6, 7], [8, 9]],
                               [[0, 1], [2, 3]],
                               [[0, 0], [0, 0]],
                               [[0, 0], [0, 0]]],
                              [[[13, 14], [15, 16]],
                               [[8, 9], [10, 11]],
                               [[10, 11], [12, 13]],
                               [[0, 0], [0, 0]]]])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes_placeholder, scores_placeholder, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size,
        masks=masks_placeholder)

    self.assertIsNone(nmsed_additional_fields)
    # Check static shapes
    self.assertAllEqual(nmsed_boxes.shape.as_list(), [None, 4, 4])
    self.assertAllEqual(nmsed_scores.shape.as_list(), [None, 4])
    self.assertAllEqual(nmsed_classes.shape.as_list(), [None, 4])
    self.assertAllEqual(nmsed_masks.shape.as_list(), [None, 4, 2, 2])
    self.assertEqual(num_detections.shape.as_list(), [None])

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   nmsed_masks, num_detections],
                                  feed_dict={boxes_placeholder: boxes,
                                             scores_placeholder: scores,
                                             masks_placeholder: masks})
      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      self.assertAllClose(num_detections, [2, 3])
      self.assertAllClose(nmsed_masks, exp_nms_masks)

  def test_batch_multiclass_nms_with_masks_and_num_valid_boxes(self):
    boxes = tf.constant([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                          [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                          [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                          [[0, 10, 1, 11], [0, 10, 1, 11]]],
                         [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                          [[0, 100, 1, 101], [0, 100, 1, 101]],
                          [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                          [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                        tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                           [.6, 0.01], [.95, 0]],
                          [[.5, 0.01], [.3, 0.01],
                           [.01, .85], [.01, .5]]])
    masks = tf.constant([[[[[0, 1], [2, 3]], [[1, 2], [3, 4]]],
                          [[[2, 3], [4, 5]], [[3, 4], [5, 6]]],
                          [[[4, 5], [6, 7]], [[5, 6], [7, 8]]],
                          [[[6, 7], [8, 9]], [[7, 8], [9, 10]]]],
                         [[[[8, 9], [10, 11]], [[9, 10], [11, 12]]],
                          [[[10, 11], [12, 13]], [[11, 12], [13, 14]]],
                          [[[12, 13], [14, 15]], [[13, 14], [15, 16]]],
                          [[[14, 15], [16, 17]], [[15, 16], [17, 18]]]]],
                        tf.float32)
    num_valid_boxes = tf.constant([1, 1], tf.int32)
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = [[[0, 0, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],
                       [[0, 10.1, 1, 11.1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]]
    exp_nms_scores = [[.9, 0, 0, 0],
                      [.5, 0, 0, 0]]
    exp_nms_classes = [[0, 0, 0, 0],
                       [0, 0, 0, 0]]
    exp_nms_masks = [[[[0, 1], [2, 3]],
                      [[0, 0], [0, 0]],
                      [[0, 0], [0, 0]],
                      [[0, 0], [0, 0]]],
                     [[[8, 9], [10, 11]],
                      [[0, 0], [0, 0]],
                      [[0, 0], [0, 0]],
                      [[0, 0], [0, 0]]]]

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size,
        num_valid_boxes=num_valid_boxes, masks=masks)

    self.assertIsNone(nmsed_additional_fields)

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   nmsed_masks, num_detections])
      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      self.assertAllClose(num_detections, [1, 1])
      self.assertAllClose(nmsed_masks, exp_nms_masks)

  def test_batch_multiclass_nms_with_additional_fields_and_num_valid_boxes(
      self):
    boxes = tf.constant([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                          [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                          [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                          [[0, 10, 1, 11], [0, 10, 1, 11]]],
                         [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                          [[0, 100, 1, 101], [0, 100, 1, 101]],
                          [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                          [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                        tf.float32)
    scores = tf.constant([[[.9, 0.01], [.75, 0.05],
                           [.6, 0.01], [.95, 0]],
                          [[.5, 0.01], [.3, 0.01],
                           [.01, .85], [.01, .5]]])
    additional_fields = {
        'keypoints': tf.constant(
            [[[[6, 7], [8, 9]],
              [[0, 1], [2, 3]],
              [[0, 0], [0, 0]],
              [[0, 0], [0, 0]]],
             [[[13, 14], [15, 16]],
              [[8, 9], [10, 11]],
              [[10, 11], [12, 13]],
              [[0, 0], [0, 0]]]],
            tf.float32)
    }

    additional_fields['size'] = tf.constant(
        [[[[7], [9]], [[1], [3]], [[0], [0]], [[0], [0]]],
         [[[14], [16]], [[9], [11]], [[11], [13]], [[0], [0]]]], tf.float32)

    num_valid_boxes = tf.constant([1, 1], tf.int32)
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = [[[0, 0, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],
                       [[0, 10.1, 1, 11.1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]]
    exp_nms_scores = [[.9, 0, 0, 0],
                      [.5, 0, 0, 0]]
    exp_nms_classes = [[0, 0, 0, 0],
                       [0, 0, 0, 0]]
    exp_nms_additional_fields = {
        'keypoints': np.array([[[[6, 7], [8, 9]],
                                [[0, 0], [0, 0]],
                                [[0, 0], [0, 0]],
                                [[0, 0], [0, 0]]],
                               [[[13, 14], [15, 16]],
                                [[0, 0], [0, 0]],
                                [[0, 0], [0, 0]],
                                [[0, 0], [0, 0]]]])
    }

    exp_nms_additional_fields['size'] = np.array([[[[7], [9]], [[0], [0]],
                                                   [[0], [0]], [[0], [0]]],
                                                  [[[14], [16]], [[0], [0]],
                                                   [[0], [0]], [[0], [0]]]])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     nmsed_additional_fields, num_detections
    ) = post_processing.batch_multiclass_non_max_suppression(
        boxes, scores, score_thresh, iou_thresh,
        max_size_per_class=max_output_size, max_total_size=max_output_size,
        num_valid_boxes=num_valid_boxes,
        additional_fields=additional_fields)

    self.assertIsNone(nmsed_masks)

    with self.test_session() as sess:
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_additional_fields,
       num_detections) = sess.run([nmsed_boxes, nmsed_scores, nmsed_classes,
                                   nmsed_additional_fields, num_detections])

      self.assertAllClose(nmsed_boxes, exp_nms_corners)
      self.assertAllClose(nmsed_scores, exp_nms_scores)
      self.assertAllClose(nmsed_classes, exp_nms_classes)
      for key in exp_nms_additional_fields:
        self.assertAllClose(nmsed_additional_fields[key],
                            exp_nms_additional_fields[key])
      self.assertAllClose(num_detections, [1, 1])

  # TODO(bhattad): Remove conditional after CMLE moves to TF 1.9

if __name__ == '__main__':
  tf.test.main()
