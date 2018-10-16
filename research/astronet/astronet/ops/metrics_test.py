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

"""Tests for metrics.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet.ops import metrics


def _unpack_metric_map(names_to_tuples):
  """Unpacks {metric_name: (metric_value, update_op)} into separate dicts."""
  metric_names = names_to_tuples.keys()
  value_ops, update_ops = zip(*names_to_tuples.values())
  return dict(zip(metric_names, value_ops)), dict(zip(metric_names, update_ops))


class _MockHparams(object):
  """Mock Hparams class to support accessing with dot notation."""

  pass


class _MockModel(object):
  """Mock model for testing."""

  def __init__(self, labels, predictions, weights, batch_losses, output_dim):
    self.labels = tf.constant(labels, dtype=tf.int32)
    self.predictions = tf.constant(predictions, dtype=tf.float32)
    self.weights = None if weights is None else tf.constant(
        weights, dtype=tf.float32)
    self.batch_losses = tf.constant(batch_losses, dtype=tf.float32)
    self.hparams = _MockHparams()
    self.hparams.output_dim = output_dim


class MetricsTest(tf.test.TestCase):

  def testMultiClassificationWithoutWeights(self):
    labels = [0, 1, 2, 3]
    predictions = [
        [0.7, 0.2, 0.1, 0.0],  # Predicted label = 0
        [0.2, 0.4, 0.2, 0.2],  # Predicted label = 1
        [0.0, 0.0, 0.0, 1.0],  # Predicted label = 3
        [0.1, 0.1, 0.7, 0.1],  # Predicted label = 2
    ]
    weights = None
    batch_losses = [0, 0, 4, 2]

    model = _MockModel(labels, predictions, weights, batch_losses, output_dim=4)
    metric_map = metrics.create_metrics(model)
    value_ops, update_ops = _unpack_metric_map(metric_map)
    initializer = tf.local_variables_initializer()

    with self.test_session() as sess:
      sess.run(initializer)

      sess.run(update_ops)
      self.assertAllClose({
          "num_examples": 4,
          "accuracy/num_correct": 2,
          "accuracy/accuracy": 0.5,
          "losses/weighted_cross_entropy": 1.5,
          "confusion_matrix/label_0_pred_0": 1,
          "confusion_matrix/label_0_pred_1": 0,
          "confusion_matrix/label_0_pred_2": 0,
          "confusion_matrix/label_0_pred_3": 0,
          "confusion_matrix/label_1_pred_0": 0,
          "confusion_matrix/label_1_pred_1": 1,
          "confusion_matrix/label_1_pred_2": 0,
          "confusion_matrix/label_1_pred_3": 0,
          "confusion_matrix/label_2_pred_0": 0,
          "confusion_matrix/label_2_pred_1": 0,
          "confusion_matrix/label_2_pred_2": 0,
          "confusion_matrix/label_2_pred_3": 1,
          "confusion_matrix/label_3_pred_0": 0,
          "confusion_matrix/label_3_pred_1": 0,
          "confusion_matrix/label_3_pred_2": 1,
          "confusion_matrix/label_3_pred_3": 0
      }, sess.run(value_ops))

      sess.run(update_ops)
      self.assertAllClose({
          "num_examples": 8,
          "accuracy/num_correct": 4,
          "accuracy/accuracy": 0.5,
          "losses/weighted_cross_entropy": 1.5,
          "confusion_matrix/label_0_pred_0": 2,
          "confusion_matrix/label_0_pred_1": 0,
          "confusion_matrix/label_0_pred_2": 0,
          "confusion_matrix/label_0_pred_3": 0,
          "confusion_matrix/label_1_pred_0": 0,
          "confusion_matrix/label_1_pred_1": 2,
          "confusion_matrix/label_1_pred_2": 0,
          "confusion_matrix/label_1_pred_3": 0,
          "confusion_matrix/label_2_pred_0": 0,
          "confusion_matrix/label_2_pred_1": 0,
          "confusion_matrix/label_2_pred_2": 0,
          "confusion_matrix/label_2_pred_3": 2,
          "confusion_matrix/label_3_pred_0": 0,
          "confusion_matrix/label_3_pred_1": 0,
          "confusion_matrix/label_3_pred_2": 2,
          "confusion_matrix/label_3_pred_3": 0
      }, sess.run(value_ops))

  def testMultiClassificationWithWeights(self):
    labels = [0, 1, 2, 3]
    predictions = [
        [0.7, 0.2, 0.1, 0.0],  # Predicted label = 0
        [0.2, 0.4, 0.2, 0.2],  # Predicted label = 1
        [0.0, 0.0, 0.0, 1.0],  # Predicted label = 3
        [0.1, 0.1, 0.7, 0.1],  # Predicted label = 2
    ]
    weights = [0, 1, 0, 1]
    batch_losses = [0, 0, 4, 2]

    model = _MockModel(labels, predictions, weights, batch_losses, output_dim=4)
    metric_map = metrics.create_metrics(model)
    value_ops, update_ops = _unpack_metric_map(metric_map)
    initializer = tf.local_variables_initializer()

    with self.test_session() as sess:
      sess.run(initializer)

      sess.run(update_ops)
      self.assertAllClose({
          "num_examples": 2,
          "accuracy/num_correct": 1,
          "accuracy/accuracy": 0.5,
          "losses/weighted_cross_entropy": 1,
          "confusion_matrix/label_0_pred_0": 0,
          "confusion_matrix/label_0_pred_1": 0,
          "confusion_matrix/label_0_pred_2": 0,
          "confusion_matrix/label_0_pred_3": 0,
          "confusion_matrix/label_1_pred_0": 0,
          "confusion_matrix/label_1_pred_1": 1,
          "confusion_matrix/label_1_pred_2": 0,
          "confusion_matrix/label_1_pred_3": 0,
          "confusion_matrix/label_2_pred_0": 0,
          "confusion_matrix/label_2_pred_1": 0,
          "confusion_matrix/label_2_pred_2": 0,
          "confusion_matrix/label_2_pred_3": 0,
          "confusion_matrix/label_3_pred_0": 0,
          "confusion_matrix/label_3_pred_1": 0,
          "confusion_matrix/label_3_pred_2": 1,
          "confusion_matrix/label_3_pred_3": 0
      }, sess.run(value_ops))

      sess.run(update_ops)
      self.assertAllClose({
          "num_examples": 4,
          "accuracy/num_correct": 2,
          "accuracy/accuracy": 0.5,
          "losses/weighted_cross_entropy": 1,
          "confusion_matrix/label_0_pred_0": 0,
          "confusion_matrix/label_0_pred_1": 0,
          "confusion_matrix/label_0_pred_2": 0,
          "confusion_matrix/label_0_pred_3": 0,
          "confusion_matrix/label_1_pred_0": 0,
          "confusion_matrix/label_1_pred_1": 2,
          "confusion_matrix/label_1_pred_2": 0,
          "confusion_matrix/label_1_pred_3": 0,
          "confusion_matrix/label_2_pred_0": 0,
          "confusion_matrix/label_2_pred_1": 0,
          "confusion_matrix/label_2_pred_2": 0,
          "confusion_matrix/label_2_pred_3": 0,
          "confusion_matrix/label_3_pred_0": 0,
          "confusion_matrix/label_3_pred_1": 0,
          "confusion_matrix/label_3_pred_2": 2,
          "confusion_matrix/label_3_pred_3": 0
      }, sess.run(value_ops))

  def testBinaryClassificationWithoutWeights(self):
    labels = [0, 1, 1, 0]
    predictions = [
        [0.4],  # Predicted label = 0
        [0.6],  # Predicted label = 1
        [0.0],  # Predicted label = 0
        [1.0],  # Predicted label = 1
    ]
    weights = None
    batch_losses = [0, 0, 4, 2]

    model = _MockModel(labels, predictions, weights, batch_losses, output_dim=1)
    metric_map = metrics.create_metrics(model)
    value_ops, update_ops = _unpack_metric_map(metric_map)
    initializer = tf.local_variables_initializer()

    with self.test_session() as sess:
      sess.run(initializer)

      sess.run(update_ops)
      self.assertAllClose({
          "num_examples": 4,
          "accuracy/num_correct": 2,
          "accuracy/accuracy": 0.5,
          "losses/weighted_cross_entropy": 1.5,
          "auc": 0.25,
          "confusion_matrix/label_0_pred_0": 1,
          "confusion_matrix/label_0_pred_1": 1,
          "confusion_matrix/label_1_pred_0": 1,
          "confusion_matrix/label_1_pred_1": 1,
      }, sess.run(value_ops))

      sess.run(update_ops)
      self.assertAllClose({
          "num_examples": 8,
          "accuracy/num_correct": 4,
          "accuracy/accuracy": 0.5,
          "losses/weighted_cross_entropy": 1.5,
          "auc": 0.25,
          "confusion_matrix/label_0_pred_0": 2,
          "confusion_matrix/label_0_pred_1": 2,
          "confusion_matrix/label_1_pred_0": 2,
          "confusion_matrix/label_1_pred_1": 2,
      }, sess.run(value_ops))

  def testBinaryClassificationWithWeights(self):
    labels = [0, 1, 1, 0]
    predictions = [
        [0.4],  # Predicted label = 0
        [0.6],  # Predicted label = 1
        [0.0],  # Predicted label = 0
        [1.0],  # Predicted label = 1
    ]
    weights = [0, 1, 0, 1]
    batch_losses = [0, 0, 4, 2]

    model = _MockModel(labels, predictions, weights, batch_losses, output_dim=1)
    metric_map = metrics.create_metrics(model)
    value_ops, update_ops = _unpack_metric_map(metric_map)
    initializer = tf.local_variables_initializer()

    with self.test_session() as sess:
      sess.run(initializer)

      sess.run(update_ops)
      self.assertAllClose({
          "num_examples": 2,
          "accuracy/num_correct": 1,
          "accuracy/accuracy": 0.5,
          "losses/weighted_cross_entropy": 1,
          "auc": 0,
          "confusion_matrix/label_0_pred_0": 0,
          "confusion_matrix/label_0_pred_1": 1,
          "confusion_matrix/label_1_pred_0": 0,
          "confusion_matrix/label_1_pred_1": 1,
      }, sess.run(value_ops))

      sess.run(update_ops)
      self.assertAllClose({
          "num_examples": 4,
          "accuracy/num_correct": 2,
          "accuracy/accuracy": 0.5,
          "losses/weighted_cross_entropy": 1,
          "auc": 0,
          "confusion_matrix/label_0_pred_0": 0,
          "confusion_matrix/label_0_pred_1": 2,
          "confusion_matrix/label_1_pred_0": 0,
          "confusion_matrix/label_1_pred_1": 2,
      }, sess.run(value_ops))


if __name__ == "__main__":
  tf.test.main()
