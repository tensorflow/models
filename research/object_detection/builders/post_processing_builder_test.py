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

"""Tests for post_processing_builder."""

import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection.builders import post_processing_builder
from object_detection.protos import post_processing_pb2
from object_detection.utils import test_case


class PostProcessingBuilderTest(test_case.TestCase):

  def test_build_non_max_suppressor_with_correct_parameters(self):
    post_processing_text_proto = """
      batch_non_max_suppression {
        score_threshold: 0.7
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 300
        soft_nms_sigma: 0.4
      }
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    non_max_suppressor, _ = post_processing_builder.build(
        post_processing_config)
    self.assertEqual(non_max_suppressor.keywords['max_size_per_class'], 100)
    self.assertEqual(non_max_suppressor.keywords['max_total_size'], 300)
    self.assertAlmostEqual(non_max_suppressor.keywords['score_thresh'], 0.7)
    self.assertAlmostEqual(non_max_suppressor.keywords['iou_thresh'], 0.6)
    self.assertAlmostEqual(non_max_suppressor.keywords['soft_nms_sigma'], 0.4)

  def test_build_non_max_suppressor_with_correct_parameters_classagnostic_nms(
      self):
    post_processing_text_proto = """
      batch_non_max_suppression {
        score_threshold: 0.7
        iou_threshold: 0.6
        max_detections_per_class: 10
        max_total_detections: 300
        use_class_agnostic_nms: True
        max_classes_per_detection: 1
      }
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    non_max_suppressor, _ = post_processing_builder.build(
        post_processing_config)
    self.assertEqual(non_max_suppressor.keywords['max_size_per_class'], 10)
    self.assertEqual(non_max_suppressor.keywords['max_total_size'], 300)
    self.assertEqual(non_max_suppressor.keywords['max_classes_per_detection'],
                     1)
    self.assertEqual(non_max_suppressor.keywords['use_class_agnostic_nms'],
                     True)
    self.assertAlmostEqual(non_max_suppressor.keywords['score_thresh'], 0.7)
    self.assertAlmostEqual(non_max_suppressor.keywords['iou_thresh'], 0.6)

  def test_build_identity_score_converter(self):
    post_processing_text_proto = """
      score_converter: IDENTITY
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, score_converter = post_processing_builder.build(
        post_processing_config)
    self.assertEqual(score_converter.__name__, 'identity_with_logit_scale')
    def graph_fn():
      inputs = tf.constant([1, 1], tf.float32)
      outputs = score_converter(inputs)
      return outputs
    converted_scores = self.execute_cpu(graph_fn, [])
    self.assertAllClose(converted_scores, [1, 1])

  def test_build_identity_score_converter_with_logit_scale(self):
    post_processing_text_proto = """
      score_converter: IDENTITY
      logit_scale: 2.0
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, score_converter = post_processing_builder.build(post_processing_config)
    self.assertEqual(score_converter.__name__, 'identity_with_logit_scale')

    def graph_fn():
      inputs = tf.constant([1, 1], tf.float32)
      outputs = score_converter(inputs)
      return outputs
    converted_scores = self.execute_cpu(graph_fn, [])
    self.assertAllClose(converted_scores, [.5, .5])

  def test_build_sigmoid_score_converter(self):
    post_processing_text_proto = """
      score_converter: SIGMOID
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, score_converter = post_processing_builder.build(post_processing_config)
    self.assertEqual(score_converter.__name__, 'sigmoid_with_logit_scale')

  def test_build_softmax_score_converter(self):
    post_processing_text_proto = """
      score_converter: SOFTMAX
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, score_converter = post_processing_builder.build(post_processing_config)
    self.assertEqual(score_converter.__name__, 'softmax_with_logit_scale')

  def test_build_softmax_score_converter_with_temperature(self):
    post_processing_text_proto = """
      score_converter: SOFTMAX
      logit_scale: 2.0
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, score_converter = post_processing_builder.build(post_processing_config)
    self.assertEqual(score_converter.__name__, 'softmax_with_logit_scale')

  def test_build_calibrator_with_nonempty_config(self):
    """Test that identity function used when no calibration_config specified."""
    # Calibration config maps all scores to 0.5.
    post_processing_text_proto = """
      score_converter: SOFTMAX
      calibration_config {
        function_approximation {
          x_y_pairs {
              x_y_pair {
                x: 0.0
                y: 0.5
              }
              x_y_pair {
                x: 1.0
                y: 0.5
              }}}}"""
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, calibrated_score_conversion_fn = post_processing_builder.build(
        post_processing_config)
    self.assertEqual(calibrated_score_conversion_fn.__name__,
                     'calibrate_with_function_approximation')

    def graph_fn():
      input_scores = tf.constant([1, 1], tf.float32)
      outputs = calibrated_score_conversion_fn(input_scores)
      return outputs
    calibrated_scores = self.execute_cpu(graph_fn, [])
    self.assertAllClose(calibrated_scores, [0.5, 0.5])

  def test_build_temperature_scaling_calibrator(self):
    post_processing_text_proto = """
      score_converter: SOFTMAX
      calibration_config {
        temperature_scaling_calibration {
          scaler: 2.0
          }}"""
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, calibrated_score_conversion_fn = post_processing_builder.build(
        post_processing_config)
    self.assertEqual(calibrated_score_conversion_fn.__name__,
                     'calibrate_with_temperature_scaling_calibration')

    def graph_fn():
      input_scores = tf.constant([1, 1], tf.float32)
      outputs = calibrated_score_conversion_fn(input_scores)
      return outputs
    calibrated_scores = self.execute_cpu(graph_fn, [])
    self.assertAllClose(calibrated_scores, [0.5, 0.5])

if __name__ == '__main__':
  tf.test.main()
