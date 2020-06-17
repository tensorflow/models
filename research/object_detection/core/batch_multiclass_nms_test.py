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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
from object_detection.core import post_processing
from object_detection.utils import test_case


class BatchMulticlassNonMaxSuppressionTest(test_case.TestCase,
                                           parameterized.TestCase):

  def test_batch_multiclass_nms_with_batch_size_1(self):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]],
                       [[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0],
                        [.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]], np.float32)
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 4

    exp_nms_corners = [[[0, 10, 1, 11],
                        [0, 0, 1, 1],
                        [0, 999, 2, 1004],
                        [0, 100, 1, 101]]]
    exp_nms_scores = [[.95, .9, .85, .3]]
    exp_nms_classes = [[0, 0, 1, 0]]
    def graph_fn(boxes, scores):
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       nmsed_additional_fields, num_detections
      ) = post_processing.batch_multiclass_non_max_suppression(
          boxes, scores, score_thresh, iou_thresh,
          max_size_per_class=max_output_size,
          max_total_size=max_output_size)
      self.assertIsNone(nmsed_masks)
      self.assertIsNone(nmsed_additional_fields)
      return (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections)

    (nmsed_boxes, nmsed_scores, nmsed_classes,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores])
    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertEqual(num_detections, [4])

  def test_batch_iou_with_negative_data(self):
    def graph_fn():
      boxes = tf.constant([[[0, -0.01, 0.1, 1.1], [0, 0.2, 0.2, 5.0],
                            [0, -0.01, 0.1, 1.], [-1, -1, -1, -1]]], tf.float32)
      iou = post_processing.batch_iou(boxes, boxes)
      return iou
    iou = self.execute_cpu(graph_fn, [])
    expected_iou = [[[0.99999994, 0.0917431, 0.9099099, -1.],
                     [0.0917431, 1., 0.08154944, -1.],
                     [0.9099099, 0.08154944, 1., -1.], [-1., -1., -1., -1.]]]
    self.assertAllClose(iou, expected_iou)

  @parameterized.parameters(False, True)
  def test_batch_multiclass_nms_with_batch_size_2(self, use_dynamic_map_fn):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]], np.float32)
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
    def graph_fn(boxes, scores):
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       nmsed_additional_fields, num_detections
      ) = post_processing.batch_multiclass_non_max_suppression(
          boxes, scores, score_thresh, iou_thresh,
          max_size_per_class=max_output_size,
          max_total_size=max_output_size,
          use_dynamic_map_fn=use_dynamic_map_fn)
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
      return (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections)

    (nmsed_boxes, nmsed_scores, nmsed_classes,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores])
    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertAllClose(num_detections, [2, 3])

  def test_batch_multiclass_nms_with_per_batch_clip_window(self):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]], np.float32)
    clip_window = np.array([0., 0., 200., 200.], np.float32)
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
    def graph_fn(boxes, scores, clip_window):
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
      return nmsed_boxes, nmsed_scores, nmsed_classes, num_detections

    (nmsed_boxes, nmsed_scores, nmsed_classes,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores, clip_window])
    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertAllClose(num_detections, [2, 2])

  def test_batch_multiclass_nms_with_per_image_clip_window(self):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]], np.float32)
    clip_window = np.array([[0., 0., 5., 5.],
                            [0., 0., 200., 200.]], np.float32)
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

    def graph_fn(boxes, scores, clip_window):
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
      return nmsed_boxes, nmsed_scores, nmsed_classes, num_detections

    (nmsed_boxes, nmsed_scores, nmsed_classes,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores, clip_window])
    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertAllClose(num_detections, [1, 2])

  def test_batch_multiclass_nms_with_masks(self):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]], np.float32)
    masks = np.array([[[[[0, 1], [2, 3]], [[1, 2], [3, 4]]],
                       [[[2, 3], [4, 5]], [[3, 4], [5, 6]]],
                       [[[4, 5], [6, 7]], [[5, 6], [7, 8]]],
                       [[[6, 7], [8, 9]], [[7, 8], [9, 10]]]],
                      [[[[8, 9], [10, 11]], [[9, 10], [11, 12]]],
                       [[[10, 11], [12, 13]], [[11, 12], [13, 14]]],
                       [[[12, 13], [14, 15]], [[13, 14], [15, 16]]],
                       [[[14, 15], [16, 17]], [[15, 16], [17, 18]]]]],
                     np.float32)
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

    def graph_fn(boxes, scores, masks):
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
      return (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
              num_detections)

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores, masks])
    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertAllClose(num_detections, [2, 3])
    self.assertAllClose(nmsed_masks, exp_nms_masks)

  def test_batch_multiclass_nms_with_additional_fields(self):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]], np.float32)
    keypoints = np.array(
        [[[[6, 7], [8, 9]],
          [[0, 1], [2, 3]],
          [[0, 0], [0, 0]],
          [[0, 0], [0, 0]]],
         [[[13, 14], [15, 16]],
          [[8, 9], [10, 11]],
          [[10, 11], [12, 13]],
          [[0, 0], [0, 0]]]],
        np.float32)
    size = np.array(
        [[[[6], [8]], [[0], [2]], [[0], [0]], [[0], [0]]],
         [[[13], [15]], [[8], [10]], [[10], [12]], [[0], [0]]]], np.float32)
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

    def graph_fn(boxes, scores, keypoints, size):
      additional_fields = {'keypoints': keypoints, 'size': size}
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
      return (nmsed_boxes, nmsed_scores, nmsed_classes,
              nmsed_additional_fields['keypoints'],
              nmsed_additional_fields['size'],
              num_detections)
    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_keypoints, nmsed_size,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores, keypoints,
                                                   size])

    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertAllClose(nmsed_keypoints,
                        exp_nms_additional_fields['keypoints'])
    self.assertAllClose(nmsed_size,
                        exp_nms_additional_fields['size'])
    self.assertAllClose(num_detections, [2, 3])

  def test_batch_multiclass_nms_with_masks_and_num_valid_boxes(self):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]], np.float32)
    masks = np.array([[[[[0, 1], [2, 3]], [[1, 2], [3, 4]]],
                       [[[2, 3], [4, 5]], [[3, 4], [5, 6]]],
                       [[[4, 5], [6, 7]], [[5, 6], [7, 8]]],
                       [[[6, 7], [8, 9]], [[7, 8], [9, 10]]]],
                      [[[[8, 9], [10, 11]], [[9, 10], [11, 12]]],
                       [[[10, 11], [12, 13]], [[11, 12], [13, 14]]],
                       [[[12, 13], [14, 15]], [[13, 14], [15, 16]]],
                       [[[14, 15], [16, 17]], [[15, 16], [17, 18]]]]],
                     np.float32)
    num_valid_boxes = np.array([1, 1], np.int32)
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

    def graph_fn(boxes, scores, masks, num_valid_boxes):
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       nmsed_additional_fields, num_detections
      ) = post_processing.batch_multiclass_non_max_suppression(
          boxes, scores, score_thresh, iou_thresh,
          max_size_per_class=max_output_size, max_total_size=max_output_size,
          masks=masks, num_valid_boxes=num_valid_boxes)
      self.assertIsNone(nmsed_additional_fields)
      return (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
              num_detections)

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores, masks,
                                                   num_valid_boxes])
    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertAllClose(num_detections, [1, 1])
    self.assertAllClose(nmsed_masks, exp_nms_masks)

  def test_batch_multiclass_nms_with_additional_fields_and_num_valid_boxes(
      self):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]], np.float32)
    keypoints = np.array(
        [[[[6, 7], [8, 9]],
          [[0, 1], [2, 3]],
          [[0, 0], [0, 0]],
          [[0, 0], [0, 0]]],
         [[[13, 14], [15, 16]],
          [[8, 9], [10, 11]],
          [[10, 11], [12, 13]],
          [[0, 0], [0, 0]]]],
        np.float32)
    size = np.array(
        [[[[7], [9]], [[1], [3]], [[0], [0]], [[0], [0]]],
         [[[14], [16]], [[9], [11]], [[11], [13]], [[0], [0]]]], np.float32)

    num_valid_boxes = np.array([1, 1], np.int32)
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
    def graph_fn(boxes, scores, keypoints, size, num_valid_boxes):
      additional_fields = {'keypoints': keypoints, 'size': size}
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       nmsed_additional_fields, num_detections
      ) = post_processing.batch_multiclass_non_max_suppression(
          boxes, scores, score_thresh, iou_thresh,
          max_size_per_class=max_output_size, max_total_size=max_output_size,
          num_valid_boxes=num_valid_boxes,
          additional_fields=additional_fields)
      self.assertIsNone(nmsed_masks)
      return (nmsed_boxes, nmsed_scores, nmsed_classes,
              nmsed_additional_fields['keypoints'],
              nmsed_additional_fields['size'], num_detections)

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_keypoints, nmsed_size,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores, keypoints,
                                                   size, num_valid_boxes])

    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertAllClose(nmsed_keypoints,
                        exp_nms_additional_fields['keypoints'])
    self.assertAllClose(nmsed_size,
                        exp_nms_additional_fields['size'])
    self.assertAllClose(num_detections, [1, 1])

  def test_combined_nms_with_batch_size_2(self):
    """Test use_combined_nms."""
    boxes = np.array([[[[0, 0, 0.1, 0.1], [0, 0, 0.1, 0.1]],
                       [[0, 0.01, 1, 0.11], [0, 0.6, 0.1, 0.7]],
                       [[0, -0.01, 0.1, 0.09], [0, -0.1, 0.1, 0.09]],
                       [[0, 0.11, 0.1, 0.2], [0, 0.11, 0.1, 0.2]]],
                      [[[0, 0, 0.2, 0.2], [0, 0, 0.2, 0.2]],
                       [[0, 0.02, 0.2, 0.22], [0, 0.02, 0.2, 0.22]],
                       [[0, -0.02, 0.2, 0.19], [0, -0.02, 0.2, 0.19]],
                       [[0, 0.21, 0.2, 0.3], [0, 0.21, 0.2, 0.3]]]],
                     np.float32)
    scores = np.array([[[.1, 0.9], [.75, 0.8],
                        [.6, 0.3], [0.95, 0.1]],
                       [[.1, 0.9], [.75, 0.8],
                        [.6, .3], [.95, .1]]], np.float32)
    score_thresh = 0.1
    iou_thresh = .5
    max_output_size = 3

    exp_nms_corners = np.array([[[0, 0.11, 0.1, 0.2],
                                 [0, 0, 0.1, 0.1],
                                 [0, 0.6, 0.1, 0.7]],
                                [[0, 0.21, 0.2, 0.3],
                                 [0, 0, 0.2, 0.2],
                                 [0, 0.02, 0.2, 0.22]]])
    exp_nms_scores = np.array([[.95, .9, 0.8],
                               [.95, .9, .75]])
    exp_nms_classes = np.array([[0, 1, 1],
                                [0, 1, 0]])

    def graph_fn(boxes, scores):
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       nmsed_additional_fields, num_detections
      ) = post_processing.batch_multiclass_non_max_suppression(
          boxes, scores, score_thresh, iou_thresh,
          max_size_per_class=max_output_size, max_total_size=max_output_size,
          use_static_shapes=True,
          use_combined_nms=True)
      self.assertIsNone(nmsed_masks)
      self.assertIsNone(nmsed_additional_fields)
      return (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections)

    (nmsed_boxes, nmsed_scores, nmsed_classes,
     num_detections) = self.execute_cpu(graph_fn, [boxes, scores])
    self.assertAllClose(nmsed_boxes, exp_nms_corners)
    self.assertAllClose(nmsed_scores, exp_nms_scores)
    self.assertAllClose(nmsed_classes, exp_nms_classes)
    self.assertListEqual(num_detections.tolist(), [3, 3])

  def test_batch_multiclass_nms_with_use_static_shapes(self):
    boxes = np.array([[[[0, 0, 1, 1], [0, 0, 4, 5]],
                       [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
                       [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
                       [[0, 10, 1, 11], [0, 10, 1, 11]]],
                      [[[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
                       [[0, 100, 1, 101], [0, 100, 1, 101]],
                       [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
                       [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]]],
                     np.float32)
    scores = np.array([[[.9, 0.01], [.75, 0.05],
                        [.6, 0.01], [.95, 0]],
                       [[.5, 0.01], [.3, 0.01],
                        [.01, .85], [.01, .5]]],
                      np.float32)
    clip_window = np.array([[0., 0., 5., 5.],
                            [0., 0., 200., 200.]],
                           np.float32)
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

    def graph_fn(boxes, scores, clip_window):
      (nmsed_boxes, nmsed_scores, nmsed_classes, _, _, num_detections
      ) = post_processing.batch_multiclass_non_max_suppression(
          boxes, scores, score_thresh, iou_thresh,
          max_size_per_class=max_output_size, clip_window=clip_window,
          use_static_shapes=True)
      return nmsed_boxes, nmsed_scores, nmsed_classes, num_detections

    (nmsed_boxes, nmsed_scores, nmsed_classes,
     num_detections) = self.execute(graph_fn, [boxes, scores, clip_window])
    for i in range(len(num_detections)):
      self.assertAllClose(nmsed_boxes[i, 0:num_detections[i]],
                          exp_nms_corners[i, 0:num_detections[i]])
      self.assertAllClose(nmsed_scores[i, 0:num_detections[i]],
                          exp_nms_scores[i, 0:num_detections[i]])
      self.assertAllClose(nmsed_classes[i, 0:num_detections[i]],
                          exp_nms_classes[i, 0:num_detections[i]])
    self.assertAllClose(num_detections, [1, 2])

if __name__ == '__main__':
  tf.test.main()
