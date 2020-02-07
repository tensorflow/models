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

"""Tensorflow ops to calibrate class predictions and background class."""

import tensorflow as tf
from object_detection.utils import shape_utils


def _find_interval_containing_new_value(x, new_value):
  """Find the index of x (ascending-ordered) after which new_value occurs."""
  new_value_shape = shape_utils.combined_static_and_dynamic_shape(new_value)[0]
  x_shape = shape_utils.combined_static_and_dynamic_shape(x)[0]
  compare = tf.cast(tf.reshape(new_value, shape=(new_value_shape, 1)) >=
                    tf.reshape(x, shape=(1, x_shape)),
                    dtype=tf.int32)
  diff = compare[:, 1:] - compare[:, :-1]
  interval_idx = tf.argmin(diff, axis=1)
  return interval_idx


def _tf_linear_interp1d(x_to_interpolate, fn_x, fn_y):
  """Tensorflow implementation of 1d linear interpolation.

  Args:
    x_to_interpolate: tf.float32 Tensor of shape (num_examples,) over which 1d
      linear interpolation is performed.
    fn_x: Monotonically-increasing, non-repeating tf.float32 Tensor of shape
      (length,) used as the domain to approximate a function.
    fn_y: tf.float32 Tensor of shape (length,) used as the range to approximate
      a function.

  Returns:
    tf.float32 Tensor of shape (num_examples,)
  """
  x_pad = tf.concat([fn_x[:1] - 1, fn_x, fn_x[-1:] + 1], axis=0)
  y_pad = tf.concat([fn_y[:1], fn_y, fn_y[-1:]], axis=0)
  interval_idx = _find_interval_containing_new_value(x_pad, x_to_interpolate)

  # Interpolate
  alpha = (
      (x_to_interpolate - tf.gather(x_pad, interval_idx)) /
      (tf.gather(x_pad, interval_idx + 1) - tf.gather(x_pad, interval_idx)))
  interpolation = ((1 - alpha) * tf.gather(y_pad, interval_idx) +
                   alpha * tf.gather(y_pad, interval_idx + 1))

  return interpolation


def _function_approximation_proto_to_tf_tensors(x_y_pairs_message):
  """Extracts (x,y) pairs from a XYPairs message.

  Args:
    x_y_pairs_message: calibration_pb2..XYPairs proto
  Returns:
    tf_x: tf.float32 tensor of shape (number_xy_pairs,) for function domain.
    tf_y: tf.float32 tensor of shape (number_xy_pairs,) for function range.
  """
  tf_x = tf.convert_to_tensor([x_y_pair.x
                               for x_y_pair
                               in x_y_pairs_message.x_y_pair],
                              dtype=tf.float32)
  tf_y = tf.convert_to_tensor([x_y_pair.y
                               for x_y_pair
                               in x_y_pairs_message.x_y_pair],
                              dtype=tf.float32)
  return tf_x, tf_y


def _get_class_id_function_dict(calibration_config):
  """Create a dictionary mapping class id to function approximations.

  Args:
    calibration_config: calibration_pb2 proto containing
      id_function_approximations.
  Returns:
    Dictionary mapping a class id to a tuple of TF tensors to be used for
    function approximation.
  """
  class_id_function_dict = {}
  class_id_xy_pairs_map = (
      calibration_config.class_id_function_approximations.class_id_xy_pairs_map)
  for class_id in class_id_xy_pairs_map:
    class_id_function_dict[class_id] = (
        _function_approximation_proto_to_tf_tensors(
            class_id_xy_pairs_map[class_id]))

  return class_id_function_dict


def build(calibration_config):
  """Returns a function that calibrates Tensorflow model scores.

  All returned functions are expected to apply positive monotonic
  transformations to inputs (i.e. score ordering is strictly preserved or
  adjacent scores are mapped to the same score, but an input of lower value
  should never be exceed an input of higher value after transformation).  For
  class-agnostic calibration, positive monotonicity should hold across all
  scores. In class-specific cases, positive monotonicity should hold within each
  class.

  Args:
    calibration_config: calibration_pb2.CalibrationConfig proto.
  Returns:
    Function that that accepts class_predictions_with_background and calibrates
    the output based on calibration_config's parameters.
  Raises:
    ValueError: No calibration builder defined for "Oneof" in
      calibration_config.
  """

  # Linear Interpolation (usually used as a result of calibration via
  # isotonic regression).
  if calibration_config.WhichOneof('calibrator') == 'function_approximation':

    def calibration_fn(class_predictions_with_background):
      """Calibrate predictions via 1-d linear interpolation.

      Predictions scores are linearly interpolated based on a class-agnostic
      function approximation. Note that the 0-indexed background class is also
      transformed.

      Args:
        class_predictions_with_background: tf.float32 tensor of shape
          [batch_size, num_anchors, num_classes + 1] containing scores on the
          interval [0,1]. This is usually produced by a sigmoid or softmax layer
          and the result of calling the `predict` method of a detection model.

      Returns:
        tf.float32 tensor of the same shape as the input with values on the
        interval [0, 1].
      """
      # Flattening Tensors and then reshaping at the end.
      flat_class_predictions_with_background = tf.reshape(
          class_predictions_with_background, shape=[-1])
      fn_x, fn_y = _function_approximation_proto_to_tf_tensors(
          calibration_config.function_approximation.x_y_pairs)
      updated_scores = _tf_linear_interp1d(
          flat_class_predictions_with_background, fn_x, fn_y)

      # Un-flatten the scores
      original_detections_shape = shape_utils.combined_static_and_dynamic_shape(
          class_predictions_with_background)
      calibrated_class_predictions_with_background = tf.reshape(
          updated_scores,
          shape=original_detections_shape,
          name='calibrate_scores')
      return calibrated_class_predictions_with_background

  elif (calibration_config.WhichOneof('calibrator') ==
        'class_id_function_approximations'):

    def calibration_fn(class_predictions_with_background):
      """Calibrate predictions per class via 1-d linear interpolation.

      Prediction scores are linearly interpolated with class-specific function
      approximations. Note that after calibration, an anchor's class scores will
      not necessarily sum to 1, and score ordering may change, depending on each
      class' calibration parameters.

      Args:
        class_predictions_with_background: tf.float32 tensor of shape
          [batch_size, num_anchors, num_classes + 1] containing scores on the
          interval [0,1]. This is usually produced by a sigmoid or softmax layer
          and the result of calling the `predict` method of a detection model.

      Returns:
        tf.float32 tensor of the same shape as the input with values on the
        interval [0, 1].

      Raises:
        KeyError: Calibration parameters are not present for a class.
      """
      class_id_function_dict = _get_class_id_function_dict(calibration_config)

      # Tensors are split by class and then recombined at the end to recover
      # the input's original shape. If a class id does not have calibration
      # parameters, it is left unchanged.
      class_tensors = tf.unstack(class_predictions_with_background, axis=-1)
      calibrated_class_tensors = []
      for class_id, class_tensor in enumerate(class_tensors):
        flat_class_tensor = tf.reshape(class_tensor, shape=[-1])
        if class_id in class_id_function_dict:
          output_tensor = _tf_linear_interp1d(
              x_to_interpolate=flat_class_tensor,
              fn_x=class_id_function_dict[class_id][0],
              fn_y=class_id_function_dict[class_id][1])
        else:
          tf.logging.info(
              'Calibration parameters for class id `%d` not not found',
              class_id)
          output_tensor = flat_class_tensor
        calibrated_class_tensors.append(output_tensor)

      combined_calibrated_tensor = tf.stack(calibrated_class_tensors, axis=1)
      input_shape = shape_utils.combined_static_and_dynamic_shape(
          class_predictions_with_background)
      calibrated_class_predictions_with_background = tf.reshape(
          combined_calibrated_tensor,
          shape=input_shape,
          name='calibrate_scores')
      return calibrated_class_predictions_with_background

  elif (calibration_config.WhichOneof('calibrator') ==
        'temperature_scaling_calibration'):

    def calibration_fn(class_predictions_with_background):
      """Calibrate predictions via temperature scaling.

      Predictions logits scores are scaled by the temperature scaler. Note that
      the 0-indexed background class is also transformed.

      Args:
        class_predictions_with_background: tf.float32 tensor of shape
          [batch_size, num_anchors, num_classes + 1] containing logits scores.
          This is usually produced before a sigmoid or softmax layer.

      Returns:
        tf.float32 tensor of the same shape as the input.

      Raises:
        ValueError: If temperature scaler is of incorrect value.
      """
      scaler = calibration_config.temperature_scaling_calibration.scaler
      if scaler <= 0:
        raise ValueError('The scaler in temperature scaling must be positive.')
      calibrated_class_predictions_with_background = tf.math.divide(
          class_predictions_with_background,
          scaler,
          name='calibrate_score')
      return calibrated_class_predictions_with_background

  # TODO(zbeaver): Add sigmoid calibration.
  else:
    raise ValueError('No calibration builder defined for "Oneof" in '
                     'calibration_config.')

  return calibration_fn
