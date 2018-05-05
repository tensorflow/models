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
"""Tests for object_detection.data_decoders.tf_example_parser."""

import numpy as np
import numpy.testing as np_testing
import tensorflow as tf

from object_detection.core import standard_fields as fields
from object_detection.metrics import tf_example_parser


class TfExampleDecoderTest(tf.test.TestCase):

  def _Int64Feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _FloatFeature(self, value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def _BytesFeature(self, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def testParseDetectionsAndGT(self):
    source_id = 'abc.jpg'
    # y_min, x_min, y_max, x_max
    object_bb = np.array([[0.0, 0.5, 0.3], [0.0, 0.1, 0.6], [1.0, 0.6, 0.8],
                          [1.0, 0.6, 0.7]]).transpose()
    detection_bb = np.array([[0.1, 0.2], [0.0, 0.8], [1.0, 0.6],
                             [1.0, 0.85]]).transpose()

    object_class_label = [1, 1, 2]
    object_difficult = [1, 0, 0]
    object_group_of = [0, 0, 1]
    detection_class_label = [2, 1]
    detection_score = [0.5, 0.3]
    features = {
        fields.TfExampleFields.source_id:
            self._BytesFeature(source_id),
        fields.TfExampleFields.object_bbox_ymin:
            self._FloatFeature(object_bb[:, 0].tolist()),
        fields.TfExampleFields.object_bbox_xmin:
            self._FloatFeature(object_bb[:, 1].tolist()),
        fields.TfExampleFields.object_bbox_ymax:
            self._FloatFeature(object_bb[:, 2].tolist()),
        fields.TfExampleFields.object_bbox_xmax:
            self._FloatFeature(object_bb[:, 3].tolist()),
        fields.TfExampleFields.detection_bbox_ymin:
            self._FloatFeature(detection_bb[:, 0].tolist()),
        fields.TfExampleFields.detection_bbox_xmin:
            self._FloatFeature(detection_bb[:, 1].tolist()),
        fields.TfExampleFields.detection_bbox_ymax:
            self._FloatFeature(detection_bb[:, 2].tolist()),
        fields.TfExampleFields.detection_bbox_xmax:
            self._FloatFeature(detection_bb[:, 3].tolist()),
        fields.TfExampleFields.detection_class_label:
            self._Int64Feature(detection_class_label),
        fields.TfExampleFields.detection_score:
            self._FloatFeature(detection_score),
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    parser = tf_example_parser.TfExampleDetectionAndGTParser()

    results_dict = parser.parse(example)
    self.assertIsNone(results_dict)

    features[fields.TfExampleFields.object_class_label] = (
        self._Int64Feature(object_class_label))
    features[fields.TfExampleFields.object_difficult] = (
        self._Int64Feature(object_difficult))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    results_dict = parser.parse(example)

    self.assertIsNotNone(results_dict)
    self.assertEqual(source_id, results_dict[fields.DetectionResultFields.key])
    np_testing.assert_almost_equal(
        object_bb, results_dict[fields.InputDataFields.groundtruth_boxes])
    np_testing.assert_almost_equal(
        detection_bb,
        results_dict[fields.DetectionResultFields.detection_boxes])
    np_testing.assert_almost_equal(
        detection_score,
        results_dict[fields.DetectionResultFields.detection_scores])
    np_testing.assert_almost_equal(
        detection_class_label,
        results_dict[fields.DetectionResultFields.detection_classes])
    np_testing.assert_almost_equal(
        object_difficult,
        results_dict[fields.InputDataFields.groundtruth_difficult])
    np_testing.assert_almost_equal(
        object_class_label,
        results_dict[fields.InputDataFields.groundtruth_classes])

    parser = tf_example_parser.TfExampleDetectionAndGTParser()

    features[fields.TfExampleFields.object_group_of] = (
        self._Int64Feature(object_group_of))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    results_dict = parser.parse(example)
    self.assertIsNotNone(results_dict)
    np_testing.assert_almost_equal(
        object_group_of,
        results_dict[fields.InputDataFields.groundtruth_group_of])

  def testParseString(self):
    string_val = 'abc'
    features = {'string': self._BytesFeature(string_val)}
    example = tf.train.Example(features=tf.train.Features(feature=features))

    parser = tf_example_parser.StringParser('string')
    result = parser.parse(example)
    self.assertIsNotNone(result)
    self.assertEqual(result, string_val)

    parser = tf_example_parser.StringParser('another_string')
    result = parser.parse(example)
    self.assertIsNone(result)

  def testParseFloat(self):
    float_array_val = [1.5, 1.4, 2.0]
    features = {'floats': self._FloatFeature(float_array_val)}
    example = tf.train.Example(features=tf.train.Features(feature=features))

    parser = tf_example_parser.FloatParser('floats')
    result = parser.parse(example)
    self.assertIsNotNone(result)
    np_testing.assert_almost_equal(result, float_array_val)

    parser = tf_example_parser.StringParser('another_floats')
    result = parser.parse(example)
    self.assertIsNone(result)

  def testInt64Parser(self):
    int_val = [1, 2, 3]
    features = {'ints': self._Int64Feature(int_val)}
    example = tf.train.Example(features=tf.train.Features(feature=features))

    parser = tf_example_parser.Int64Parser('ints')
    result = parser.parse(example)
    self.assertIsNotNone(result)
    np_testing.assert_almost_equal(result, int_val)

    parser = tf_example_parser.Int64Parser('another_ints')
    result = parser.parse(example)
    self.assertIsNone(result)

  def testBoundingBoxParser(self):
    bounding_boxes = np.array([[0.0, 0.5, 0.3], [0.0, 0.1, 0.6],
                               [1.0, 0.6, 0.8], [1.0, 0.6, 0.7]]).transpose()
    features = {
        'ymin': self._FloatFeature(bounding_boxes[:, 0]),
        'xmin': self._FloatFeature(bounding_boxes[:, 1]),
        'ymax': self._FloatFeature(bounding_boxes[:, 2]),
        'xmax': self._FloatFeature(bounding_boxes[:, 3])
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))

    parser = tf_example_parser.BoundingBoxParser('xmin', 'ymin', 'xmax', 'ymax')
    result = parser.parse(example)
    self.assertIsNotNone(result)
    np_testing.assert_almost_equal(result, bounding_boxes)

    parser = tf_example_parser.BoundingBoxParser('xmin', 'ymin', 'xmax',
                                                 'another_ymax')
    result = parser.parse(example)
    self.assertIsNone(result)


if __name__ == '__main__':
  tf.test.main()
