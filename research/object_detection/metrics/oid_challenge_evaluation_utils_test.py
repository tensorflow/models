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
"""Tests for oid_od_challenge_evaluation_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from pycocotools import mask
import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.metrics import oid_challenge_evaluation_utils as utils


class OidUtilTest(tf.test.TestCase):

  def testMaskToNormalizedBox(self):
    mask_np = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    box = utils._to_normalized_box(mask_np)
    self.assertAllEqual(np.array([0.25, 0.25, 0.75, 0.5]), box)
    mask_np = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1]])
    box = utils._to_normalized_box(mask_np)
    self.assertAllEqual(np.array([0.25, 0.25, 1.0, 1.0]), box)
    mask_np = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    box = utils._to_normalized_box(mask_np)
    self.assertAllEqual(np.array([0.0, 0.0, 0.0, 0.0]), box)

  def testDecodeToTensors(self):
    mask1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.uint8)
    mask2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)

    encoding1 = mask.encode(np.asfortranarray(mask1))
    encoding2 = mask.encode(np.asfortranarray(mask2))

    vals = pd.Series([encoding1['counts'], encoding2['counts']])
    image_widths = pd.Series([mask1.shape[1], mask2.shape[1]])
    image_heights = pd.Series([mask1.shape[0], mask2.shape[0]])

    segm, bbox = utils._decode_raw_data_into_masks_and_boxes(
        vals, image_widths, image_heights)
    expected_segm = np.concatenate(
        [np.expand_dims(mask1, 0),
         np.expand_dims(mask2, 0)], axis=0)
    expected_bbox = np.array([[0.0, 0.5, 2.0 / 3.0, 1.0], [0, 0, 0, 0]])
    self.assertAllEqual(expected_segm, segm)
    self.assertAllEqual(expected_bbox, bbox)


class OidChallengeEvaluationUtilTest(tf.test.TestCase):

  def testBuildGroundtruthDictionaryBoxes(self):
    np_data = pd.DataFrame(
        [['fe58ec1b06db2bb7', '/m/04bcr3', 0.0, 0.3, 0.5, 0.6, 1, None],
         ['fe58ec1b06db2bb7', '/m/02gy9n', 0.1, 0.2, 0.3, 0.4, 0, None],
         ['fe58ec1b06db2bb7', '/m/04bcr3', None, None, None, None, None, 1],
         ['fe58ec1b06db2bb7', '/m/083vt', None, None, None, None, None, 0],
         ['fe58ec1b06db2bb7', '/m/02gy9n', None, None, None, None, None, 1]],
        columns=[
            'ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf',
            'ConfidenceImageLabel'
        ])
    class_label_map = {'/m/04bcr3': 1, '/m/083vt': 2, '/m/02gy9n': 3}
    groundtruth_dictionary = utils.build_groundtruth_dictionary(
        np_data, class_label_map)

    self.assertIn(standard_fields.InputDataFields.groundtruth_boxes,
                  groundtruth_dictionary)
    self.assertIn(standard_fields.InputDataFields.groundtruth_classes,
                  groundtruth_dictionary)
    self.assertIn(standard_fields.InputDataFields.groundtruth_group_of,
                  groundtruth_dictionary)
    self.assertIn(standard_fields.InputDataFields.groundtruth_image_classes,
                  groundtruth_dictionary)

    self.assertAllEqual(
        np.array([1, 3]), groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_classes])
    self.assertAllEqual(
        np.array([1, 0]), groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_group_of])

    expected_boxes_data = np.array([[0.5, 0.0, 0.6, 0.3], [0.3, 0.1, 0.4, 0.2]])

    self.assertNDArrayNear(
        expected_boxes_data, groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_boxes], 1e-5)
    self.assertAllEqual(
        np.array([1, 2, 3]), groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_image_classes])

  def testBuildPredictionDictionaryBoxes(self):
    np_data = pd.DataFrame(
        [['fe58ec1b06db2bb7', '/m/04bcr3', 0.0, 0.3, 0.5, 0.6, 0.1],
         ['fe58ec1b06db2bb7', '/m/02gy9n', 0.1, 0.2, 0.3, 0.4, 0.2],
         ['fe58ec1b06db2bb7', '/m/04bcr3', 0.0, 0.1, 0.2, 0.3, 0.3]],
        columns=[
            'ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'Score'
        ])
    class_label_map = {'/m/04bcr3': 1, '/m/083vt': 2, '/m/02gy9n': 3}
    prediction_dictionary = utils.build_predictions_dictionary(
        np_data, class_label_map)

    self.assertIn(standard_fields.DetectionResultFields.detection_boxes,
                  prediction_dictionary)
    self.assertIn(standard_fields.DetectionResultFields.detection_classes,
                  prediction_dictionary)
    self.assertIn(standard_fields.DetectionResultFields.detection_scores,
                  prediction_dictionary)

    self.assertAllEqual(
        np.array([1, 3, 1]), prediction_dictionary[
            standard_fields.DetectionResultFields.detection_classes])
    expected_boxes_data = np.array([[0.5, 0.0, 0.6, 0.3], [0.3, 0.1, 0.4, 0.2],
                                    [0.2, 0.0, 0.3, 0.1]])
    self.assertNDArrayNear(
        expected_boxes_data, prediction_dictionary[
            standard_fields.DetectionResultFields.detection_boxes], 1e-5)
    self.assertNDArrayNear(
        np.array([0.1, 0.2, 0.3]), prediction_dictionary[
            standard_fields.DetectionResultFields.detection_scores], 1e-5)

  def testBuildGroundtruthDictionaryMasks(self):
    mask1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                     dtype=np.uint8)
    mask2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                     dtype=np.uint8)

    encoding1 = mask.encode(np.asfortranarray(mask1))
    encoding2 = mask.encode(np.asfortranarray(mask2))

    np_data = pd.DataFrame(
        [[
            'fe58ec1b06db2bb7', mask1.shape[1], mask1.shape[0], '/m/04bcr3',
            0.0, 0.3, 0.5, 0.6, 0, None, encoding1['counts']
        ],
         [
             'fe58ec1b06db2bb7', None, None, '/m/02gy9n', 0.1, 0.2, 0.3, 0.4, 1,
             None, None
         ],
         [
             'fe58ec1b06db2bb7', mask2.shape[1], mask2.shape[0], '/m/02gy9n',
             0.5, 0.6, 0.8, 0.9, 0, None, encoding2['counts']
         ],
         [
             'fe58ec1b06db2bb7', None, None, '/m/04bcr3', None, None, None,
             None, None, 1, None
         ],
         [
             'fe58ec1b06db2bb7', None, None, '/m/083vt', None, None, None, None,
             None, 0, None
         ],
         [
             'fe58ec1b06db2bb7', None, None, '/m/02gy9n', None, None, None,
             None, None, 1, None
         ]],
        columns=[
            'ImageID', 'ImageWidth', 'ImageHeight', 'LabelName', 'XMin', 'XMax',
            'YMin', 'YMax', 'IsGroupOf', 'ConfidenceImageLabel', 'Mask'
        ])
    class_label_map = {'/m/04bcr3': 1, '/m/083vt': 2, '/m/02gy9n': 3}
    groundtruth_dictionary = utils.build_groundtruth_dictionary(
        np_data, class_label_map)
    self.assertIn(standard_fields.InputDataFields.groundtruth_boxes,
                  groundtruth_dictionary)
    self.assertIn(standard_fields.InputDataFields.groundtruth_classes,
                  groundtruth_dictionary)
    self.assertIn(standard_fields.InputDataFields.groundtruth_group_of,
                  groundtruth_dictionary)
    self.assertIn(standard_fields.InputDataFields.groundtruth_image_classes,
                  groundtruth_dictionary)
    self.assertIn(standard_fields.InputDataFields.groundtruth_instance_masks,
                  groundtruth_dictionary)
    self.assertAllEqual(
        np.array([1, 3, 3]), groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_classes])
    self.assertAllEqual(
        np.array([0, 1, 0]), groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_group_of])

    expected_boxes_data = np.array([[0.5, 0.0, 0.6, 0.3], [0.3, 0.1, 0.4, 0.2],
                                    [0.8, 0.5, 0.9, 0.6]])

    self.assertNDArrayNear(
        expected_boxes_data, groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_boxes], 1e-5)
    self.assertAllEqual(
        np.array([1, 2, 3]), groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_image_classes])

    expected_segm = np.concatenate([
        np.expand_dims(mask1, 0),
        np.zeros((1, 4, 4), dtype=np.uint8),
        np.expand_dims(mask2, 0)
    ],
                                   axis=0)
    self.assertAllEqual(
        expected_segm, groundtruth_dictionary[
            standard_fields.InputDataFields.groundtruth_instance_masks])

  def testBuildPredictionDictionaryMasks(self):
    mask1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                     dtype=np.uint8)
    mask2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                     dtype=np.uint8)

    encoding1 = mask.encode(np.asfortranarray(mask1))
    encoding2 = mask.encode(np.asfortranarray(mask2))

    np_data = pd.DataFrame(
        [[
            'fe58ec1b06db2bb7', mask1.shape[1], mask1.shape[0], '/m/04bcr3',
            encoding1['counts'], 0.8
        ],
         [
             'fe58ec1b06db2bb7', mask2.shape[1], mask2.shape[0], '/m/02gy9n',
             encoding2['counts'], 0.6
         ]],
        columns=[
            'ImageID', 'ImageWidth', 'ImageHeight', 'LabelName', 'Mask', 'Score'
        ])
    class_label_map = {'/m/04bcr3': 1, '/m/02gy9n': 3}
    prediction_dictionary = utils.build_predictions_dictionary(
        np_data, class_label_map)

    self.assertIn(standard_fields.DetectionResultFields.detection_boxes,
                  prediction_dictionary)
    self.assertIn(standard_fields.DetectionResultFields.detection_classes,
                  prediction_dictionary)
    self.assertIn(standard_fields.DetectionResultFields.detection_scores,
                  prediction_dictionary)
    self.assertIn(standard_fields.DetectionResultFields.detection_masks,
                  prediction_dictionary)

    self.assertAllEqual(
        np.array([1, 3]), prediction_dictionary[
            standard_fields.DetectionResultFields.detection_classes])

    expected_boxes_data = np.array([[0.0, 0.5, 0.5, 1.0], [0, 0, 0, 0]])
    self.assertNDArrayNear(
        expected_boxes_data, prediction_dictionary[
            standard_fields.DetectionResultFields.detection_boxes], 1e-5)
    self.assertNDArrayNear(
        np.array([0.8, 0.6]), prediction_dictionary[
            standard_fields.DetectionResultFields.detection_scores], 1e-5)
    expected_segm = np.concatenate(
        [np.expand_dims(mask1, 0),
         np.expand_dims(mask2, 0)], axis=0)
    self.assertAllEqual(
        expected_segm, prediction_dictionary[
            standard_fields.DetectionResultFields.detection_masks])


if __name__ == '__main__':
  tf.test.main()
