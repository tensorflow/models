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
import tensorflow as tf
from object_detection.core import standard_fields
from object_detection.metrics import oid_od_challenge_evaluation_utils as utils


class OidOdChallengeEvaluationUtilTest(tf.test.TestCase):

  def testBuildGroundtruthDictionary(self):
    np_data = pd.DataFrame(
        [['fe58ec1b06db2bb7', '/m/04bcr3', 0.0, 0.3, 0.5, 0.6, 1, None], [
            'fe58ec1b06db2bb7', '/m/02gy9n', 0.1, 0.2, 0.3, 0.4, 0, None
        ], ['fe58ec1b06db2bb7', '/m/04bcr3', None, None, None, None, None, 1], [
            'fe58ec1b06db2bb7', '/m/083vt', None, None, None, None, None, 0
        ], ['fe58ec1b06db2bb7', '/m/02gy9n', None, None, None, None, None, 1]],
        columns=[
            'ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf',
            'ConfidenceImageLabel'
        ])
    class_label_map = {'/m/04bcr3': 1, '/m/083vt': 2, '/m/02gy9n': 3}
    groundtruth_dictionary = utils.build_groundtruth_boxes_dictionary(
        np_data, class_label_map)

    self.assertTrue(standard_fields.InputDataFields.groundtruth_boxes in
                    groundtruth_dictionary)
    self.assertTrue(standard_fields.InputDataFields.groundtruth_classes in
                    groundtruth_dictionary)
    self.assertTrue(standard_fields.InputDataFields.groundtruth_group_of in
                    groundtruth_dictionary)
    self.assertTrue(standard_fields.InputDataFields.groundtruth_image_classes in
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

  def testBuildPredictionDictionary(self):
    np_data = pd.DataFrame(
        [['fe58ec1b06db2bb7', '/m/04bcr3', 0.0, 0.3, 0.5, 0.6, 0.1], [
            'fe58ec1b06db2bb7', '/m/02gy9n', 0.1, 0.2, 0.3, 0.4, 0.2
        ], ['fe58ec1b06db2bb7', '/m/04bcr3', 0.0, 0.1, 0.2, 0.3, 0.3]],
        columns=[
            'ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'Score'
        ])
    class_label_map = {'/m/04bcr3': 1, '/m/083vt': 2, '/m/02gy9n': 3}
    prediction_dictionary = utils.build_predictions_dictionary(
        np_data, class_label_map)

    self.assertTrue(standard_fields.DetectionResultFields.detection_boxes in
                    prediction_dictionary)
    self.assertTrue(standard_fields.DetectionResultFields.detection_classes in
                    prediction_dictionary)
    self.assertTrue(standard_fields.DetectionResultFields.detection_scores in
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


if __name__ == '__main__':
  tf.test.main()
