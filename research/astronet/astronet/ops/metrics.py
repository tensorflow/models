# Copyright 2018 The TensorFlow Authors.
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

"""Functions for computing evaluation metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _metric_variable(name, shape, dtype):
  """Creates a Variable in LOCAL_VARIABLES and METRIC_VARIABLES collections."""
  return tf.get_variable(
      name,
      initializer=tf.zeros(shape, dtype),
      trainable=False,
      collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES])


def _build_metrics(labels, predictions, weights, batch_losses):
  """Builds TensorFlow operations to compute model evaluation metrics.

  Args:
    labels: Tensor with shape [batch_size].
    predictions: Tensor with shape [batch_size, output_dim].
    weights: Tensor with shape [batch_size].
    batch_losses: Tensor with shape [batch_size].

  Returns:
    A dictionary {metric_name: (metric_value, update_op).
  """
  # Compute the predicted labels.
  assert len(predictions.shape) == 2
  binary_classification = (predictions.shape[1] == 1)
  if binary_classification:
    predictions = tf.squeeze(predictions, axis=[1])
    predicted_labels = tf.to_int32(
        tf.greater(predictions, 0.5), name="predicted_labels")
  else:
    predicted_labels = tf.argmax(
        predictions, 1, name="predicted_labels", output_type=tf.int32)

  metrics = {}
  with tf.variable_scope("metrics"):
    # Total number of examples.
    num_examples = _metric_variable("num_examples", [], tf.float32)
    update_num_examples = tf.assign_add(num_examples, tf.reduce_sum(weights))
    metrics["num_examples"] = (num_examples.read_value(), update_num_examples)

    # Accuracy metrics.
    num_correct = _metric_variable("num_correct", [], tf.float32)
    is_correct = weights * tf.to_float(tf.equal(labels, predicted_labels))
    update_num_correct = tf.assign_add(num_correct, tf.reduce_sum(is_correct))
    metrics["accuracy/num_correct"] = (num_correct.read_value(),
                                       update_num_correct)
    accuracy = tf.div(num_correct, num_examples, name="accuracy")
    metrics["accuracy/accuracy"] = (accuracy, tf.no_op())

    # Weighted cross-entropy loss.
    metrics["losses/weighted_cross_entropy"] = tf.metrics.mean(
        batch_losses, weights=weights, name="cross_entropy_loss")

    # Possibly create additional metrics for binary classification.
    if binary_classification:
      labels = tf.cast(labels, dtype=tf.bool)
      predicted_labels = tf.cast(predicted_labels, dtype=tf.bool)

      # AUC.
      metrics["auc"] = tf.metrics.auc(
          labels, predictions, weights=weights, num_thresholds=1000)

      def _count_condition(name, labels_value, predicted_value):
        """Creates a counter for given values of predictions and labels."""
        count = _metric_variable(name, [], tf.float32)
        is_equal = tf.to_float(
            tf.logical_and(
                tf.equal(labels, labels_value),
                tf.equal(predicted_labels, predicted_value)))
        update_op = tf.assign_add(count, tf.reduce_sum(weights * is_equal))
        return count.read_value(), update_op

      # Confusion matrix metrics.
      metrics["confusion_matrix/true_positives"] = _count_condition(
          "true_positives", labels_value=True, predicted_value=True)
      metrics["confusion_matrix/false_positives"] = _count_condition(
          "false_positives", labels_value=False, predicted_value=True)
      metrics["confusion_matrix/true_negatives"] = _count_condition(
          "true_negatives", labels_value=False, predicted_value=False)
      metrics["confusion_matrix/false_negatives"] = _count_condition(
          "false_negatives", labels_value=True, predicted_value=False)

  return metrics


def create_metric_fn(model):
  """Creates a tuple (metric_fn, metric_fn_inputs).

  This function is primarily used for creating a TPUEstimator.

  The result of calling metric_fn(**metric_fn_inputs) is a dictionary
  {metric_name: (metric_value, update_op)}.

  Args:
    model: Instance of AstroModel.

  Returns:
    A tuple (metric_fn, metric_fn_inputs).
  """
  weights = model.weights
  if weights is None:
    weights = tf.ones_like(model.labels, dtype=tf.float32)
  metric_fn_inputs = {
      "labels": model.labels,
      "predictions": model.predictions,
      "weights": weights,
      "batch_losses": model.batch_losses,
  }

  def metric_fn(labels, predictions, weights, batch_losses):
    return _build_metrics(labels, predictions, weights, batch_losses)

  return metric_fn, metric_fn_inputs


def create_metrics(model):
  """Creates a dictionary {metric_name: (metric_value, update_op)}.

  This function is primarily used for creating an Estimator.

  Args:
    model: Instance of AstroModel.

  Returns:
    A dictionary {metric_name: (metric_value, update_op).
  """
  metric_fn, metric_fn_inputs = create_metric_fn(model)
  return metric_fn(**metric_fn_inputs)
