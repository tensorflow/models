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

"""Tests for calibration_builder."""

import numpy as np
from scipy import interpolate
import tensorflow as tf
from object_detection.builders import calibration_builder
from object_detection.protos import calibration_pb2


class CalibrationBuilderTest(tf.test.TestCase):

  def test_tf_linear_interp1d_map(self):
    """Tests TF linear interpolation mapping to a single number."""
    with self.test_session() as sess:
      tf_x = tf.constant([0., 0.5, 1.])
      tf_y = tf.constant([0.5, 0.5, 0.5])
      new_x = tf.constant([0., 0.25, 0.5, 0.75, 1.])
      tf_map_outputs = calibration_builder._tf_linear_interp1d(
          new_x, tf_x, tf_y)
      tf_map_outputs_np = sess.run([tf_map_outputs])
    self.assertAllClose(tf_map_outputs_np, [[0.5, 0.5, 0.5, 0.5, 0.5]])

  def test_tf_linear_interp1d_interpolate(self):
    """Tests TF 1d linear interpolation not mapping to a single number."""
    with self.test_session() as sess:
      tf_x = tf.constant([0., 0.5, 1.])
      tf_y = tf.constant([0.6, 0.7, 1.0])
      new_x = tf.constant([0., 0.25, 0.5, 0.75, 1.])
      tf_interpolate_outputs = calibration_builder._tf_linear_interp1d(
          new_x, tf_x, tf_y)
      tf_interpolate_outputs_np = sess.run([tf_interpolate_outputs])
    self.assertAllClose(tf_interpolate_outputs_np, [[0.6, 0.65, 0.7, 0.85, 1.]])

  @staticmethod
  def _get_scipy_interp1d(new_x, x, y):
    """Helper performing 1d linear interpolation using SciPy."""
    interpolation1d_fn = interpolate.interp1d(x, y)
    return interpolation1d_fn(new_x)

  def _get_tf_interp1d(self, new_x, x, y):
    """Helper performing 1d linear interpolation using Tensorflow."""
    with self.test_session() as sess:
      tf_interp_outputs = calibration_builder._tf_linear_interp1d(
          tf.convert_to_tensor(new_x, dtype=tf.float32),
          tf.convert_to_tensor(x, dtype=tf.float32),
          tf.convert_to_tensor(y, dtype=tf.float32))
      np_tf_interp_outputs = sess.run(tf_interp_outputs)
    return np_tf_interp_outputs

  def test_tf_linear_interp1d_against_scipy_map(self):
    """Tests parity of TF linear interpolation with SciPy for simple mapping."""
    length = 10
    np_x = np.linspace(0, 1, length)

    # Mapping all numbers to 0.5
    np_y_map = np.repeat(0.5, length)

    # Scipy and TF interpolations
    test_data_np = np.linspace(0, 1, length * 10)
    scipy_map_outputs = self._get_scipy_interp1d(test_data_np, np_x, np_y_map)
    np_tf_map_outputs = self._get_tf_interp1d(test_data_np, np_x, np_y_map)
    self.assertAllClose(scipy_map_outputs, np_tf_map_outputs)

  def test_tf_linear_interp1d_against_scipy_interpolate(self):
    """Tests parity of TF linear interpolation with SciPy."""
    length = 10
    np_x = np.linspace(0, 1, length)

    # Requires interpolation over 0.5 to 1 domain
    np_y_interp = np.linspace(0.5, 1, length)

    # Scipy interpolation for comparison
    test_data_np = np.linspace(0, 1, length * 10)
    scipy_interp_outputs = self._get_scipy_interp1d(test_data_np, np_x,
                                                    np_y_interp)
    np_tf_interp_outputs = self._get_tf_interp1d(test_data_np, np_x,
                                                 np_y_interp)
    self.assertAllClose(scipy_interp_outputs, np_tf_interp_outputs)

  @staticmethod
  def _add_function_approximation_to_calibration_proto(calibration_proto,
                                                       x_array,
                                                       y_array,
                                                       class_label):
    """Adds a function approximation to calibration proto for a class label."""
    # Per-class calibration.
    if class_label:
      label_function_approximation = (calibration_proto
                                      .label_function_approximations
                                      .label_xy_pairs_map[class_label])
    # Class-agnostic calibration.
    else:
      label_function_approximation = (calibration_proto
                                      .function_approximation
                                      .x_y_pairs)
    for x, y in zip(x_array, y_array):
      x_y_pair_message = label_function_approximation.x_y_pair.add()
      x_y_pair_message.x = x
      x_y_pair_message.y = y

  def test_class_agnostic_function_approximation(self):
    """Ensures that calibration appropriate values, regardless of class."""
    # Generate fake calibration proto. For this interpolation, any input on
    # [0.0, 0.5] should be divided by 2 and any input on (0.5, 1.0] should have
    # 0.25 subtracted from it.
    class_agnostic_x = np.asarray([0.0, 0.5, 1.0])
    class_agnostic_y = np.asarray([0.0, 0.25, 0.75])
    calibration_config = calibration_pb2.CalibrationConfig()
    self._add_function_approximation_to_calibration_proto(calibration_config,
                                                          class_agnostic_x,
                                                          class_agnostic_y,
                                                          class_label=None)

    od_graph = tf.Graph()
    with self.test_session(graph=od_graph) as sess:
      calibration_fn = calibration_builder.build(calibration_config)
      # batch_size = 2, num_classes = 2, num_anchors = 2.
      class_predictions_with_background = tf.constant(
          [[[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.0]],
           [[0.6, 0.7, 0.8],
            [0.9, 1.0, 1.0]]], dtype=tf.float32)

      # Everything should map to 0.5 if classes are ignored.
      calibrated_scores = calibration_fn(class_predictions_with_background)
      calibrated_scores_np = sess.run(calibrated_scores)
    self.assertAllClose(calibrated_scores_np, [[[0.05, 0.1, 0.15],
                                                [0.2, 0.25, 0.0]],
                                               [[0.35, 0.45, 0.55],
                                                [0.65, 0.75, 0.75]]])

if __name__ == '__main__':
  tf.test.main()
