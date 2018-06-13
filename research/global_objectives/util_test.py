# Copyright 2018 The TensorFlow Global Objectives Authors. All Rights Reserved.
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
"""Tests for global objectives util functions."""

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from global_objectives import util


def weighted_sigmoid_cross_entropy(targets, logits, weight):
  return (weight * targets * np.log(1.0 + np.exp(-logits)) + (
      (1.0 - targets) * np.log(1.0 + 1.0 / np.exp(-logits))))


def hinge_loss(labels, logits):
  # Mostly copied from tensorflow.python.ops.losses but with loss per datapoint.
  labels = tf.to_float(labels)
  all_ones = tf.ones_like(labels)
  labels = tf.subtract(2 * labels, all_ones)
  return tf.nn.relu(tf.subtract(all_ones, tf.multiply(labels, logits)))


class WeightedSigmoidCrossEntropyTest(parameterized.TestCase, tf.test.TestCase):

  def testTrivialCompatibilityWithSigmoidCrossEntropy(self):
    """Tests compatibility with unweighted function with weight 1.0."""
    x_shape = [300, 10]
    targets = np.random.random_sample(x_shape).astype(np.float32)
    logits = np.random.randn(*x_shape).astype(np.float32)
    weighted_loss = util.weighted_sigmoid_cross_entropy_with_logits(
        targets,
        logits)
    expected_loss = (
        tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
            logits, targets))
    with self.test_session():
      self.assertAllClose(expected_loss.eval(),
                          weighted_loss.eval(),
                          atol=0.000001)

  def testNonTrivialCompatibilityWithSigmoidCrossEntropy(self):
    """Tests use of an arbitrary weight (4.12)."""
    x_shape = [300, 10]
    targets = np.random.random_sample(x_shape).astype(np.float32)
    logits = np.random.randn(*x_shape).astype(np.float32)
    weight = 4.12
    weighted_loss = util.weighted_sigmoid_cross_entropy_with_logits(
        targets,
        logits,
        weight,
        weight)
    expected_loss = (
        weight *
        tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
            logits, targets))
    with self.test_session():
      self.assertAllClose(expected_loss.eval(),
                          weighted_loss.eval(),
                          atol=0.000001)

  def testDifferentSizeWeightedSigmoidCrossEntropy(self):
    """Tests correctness on 3D tensors.

    Tests that the function works as expected when logits is a 3D tensor and
    targets is a 2D tensor.
    """
    targets_shape = [30, 4]
    logits_shape = [targets_shape[0], targets_shape[1], 3]
    targets = np.random.random_sample(targets_shape).astype(np.float32)
    logits = np.random.randn(*logits_shape).astype(np.float32)

    weight_vector = [2.0, 3.0, 13.0]
    loss = util.weighted_sigmoid_cross_entropy_with_logits(targets,
                                                           logits,
                                                           weight_vector)

    with self.test_session():
      loss = loss.eval()
      for i in range(0, len(weight_vector)):
        expected = weighted_sigmoid_cross_entropy(targets, logits[:, :, i],
                                                  weight_vector[i])
        self.assertAllClose(loss[:, :, i], expected, atol=0.000001)

  @parameterized.parameters((300, 10, 0.3), (20, 4, 2.0), (30, 4, 3.9))
  def testWeightedSigmoidCrossEntropy(self, batch_size, num_labels, weight):
    """Tests thats the tf and numpy functions agree on many instances."""
    x_shape = [batch_size, num_labels]
    targets = np.random.random_sample(x_shape).astype(np.float32)
    logits = np.random.randn(*x_shape).astype(np.float32)

    with self.test_session():
      loss = util.weighted_sigmoid_cross_entropy_with_logits(
          targets,
          logits,
          weight,
          1.0,
          name='weighted-loss')
      expected = weighted_sigmoid_cross_entropy(targets, logits, weight)
      self.assertAllClose(expected, loss.eval(), atol=0.000001)

  def testGradients(self):
    """Tests that weighted loss gradients behave as expected."""
    dummy_tensor = tf.constant(1.0)

    positives_shape = [10, 1]
    positives_logits = dummy_tensor * tf.Variable(
        tf.random_normal(positives_shape) + 1.0)
    positives_targets = tf.ones(positives_shape)
    positives_weight = 4.6
    positives_loss = (
        tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
            positives_logits, positives_targets) * positives_weight)

    negatives_shape = [190, 1]
    negatives_logits = dummy_tensor * tf.Variable(
        tf.random_normal(negatives_shape))
    negatives_targets = tf.zeros(negatives_shape)
    negatives_weight = 0.9
    negatives_loss = (
        tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
            negatives_logits, negatives_targets) * negatives_weight)

    all_logits = tf.concat([positives_logits, negatives_logits], 0)
    all_targets = tf.concat([positives_targets, negatives_targets], 0)
    weighted_loss = tf.reduce_sum(
        util.weighted_sigmoid_cross_entropy_with_logits(
            all_targets, all_logits, positives_weight, negatives_weight))
    weighted_gradients = tf.gradients(weighted_loss, dummy_tensor)

    expected_loss = tf.add(
        tf.reduce_sum(positives_loss),
        tf.reduce_sum(negatives_loss))
    expected_gradients = tf.gradients(expected_loss, dummy_tensor)

    with tf.Session() as session:
      tf.global_variables_initializer().run()
      grad, expected_grad = session.run(
          [weighted_gradients, expected_gradients])
      self.assertAllClose(grad, expected_grad)

  def testDtypeFlexibility(self):
    """Tests the loss on inputs of varying data types."""
    shape = [20, 3]
    logits = np.random.randn(*shape)
    targets = tf.truncated_normal(shape)
    positive_weights = tf.constant(3, dtype=tf.int64)
    negative_weights = 1

    loss = util.weighted_sigmoid_cross_entropy_with_logits(
        targets, logits, positive_weights, negative_weights)

    with self.test_session():
      self.assertEqual(loss.eval().dtype, np.float)


class WeightedHingeLossTest(tf.test.TestCase):

  def testTrivialCompatibilityWithHinge(self):
    # Tests compatibility with unweighted hinge loss.
    x_shape = [55, 10]
    logits = tf.constant(np.random.randn(*x_shape).astype(np.float32))
    targets = tf.to_float(tf.constant(np.random.random_sample(x_shape) > 0.3))
    weighted_loss = util.weighted_hinge_loss(targets, logits)
    expected_loss = hinge_loss(targets, logits)
    with self.test_session():
      self.assertAllClose(expected_loss.eval(), weighted_loss.eval())

  def testLessTrivialCompatibilityWithHinge(self):
    # Tests compatibility with a constant weight for positives and negatives.
    x_shape = [56, 11]
    logits = tf.constant(np.random.randn(*x_shape).astype(np.float32))
    targets = tf.to_float(tf.constant(np.random.random_sample(x_shape) > 0.7))
    weight = 1.0 + 1.0/2 + 1.0/3 + 1.0/4 + 1.0/5 + 1.0/6 + 1.0/7
    weighted_loss = util.weighted_hinge_loss(targets, logits, weight, weight)
    expected_loss = hinge_loss(targets, logits) * weight
    with self.test_session():
      self.assertAllClose(expected_loss.eval(), weighted_loss.eval())

  def testNontrivialCompatibilityWithHinge(self):
    # Tests compatibility with different positive and negative weights.
    x_shape = [23, 8]
    logits_positives = tf.constant(np.random.randn(*x_shape).astype(np.float32))
    logits_negatives = tf.constant(np.random.randn(*x_shape).astype(np.float32))
    targets_positives = tf.ones(x_shape)
    targets_negatives = tf.zeros(x_shape)
    logits = tf.concat([logits_positives, logits_negatives], 0)
    targets = tf.concat([targets_positives, targets_negatives], 0)

    raw_loss = util.weighted_hinge_loss(targets,
                                        logits,
                                        positive_weights=3.4,
                                        negative_weights=1.2)
    loss = tf.reduce_sum(raw_loss, 0)
    positives_hinge = hinge_loss(targets_positives, logits_positives)
    negatives_hinge = hinge_loss(targets_negatives, logits_negatives)
    expected = tf.add(tf.reduce_sum(3.4 * positives_hinge, 0),
                      tf.reduce_sum(1.2 * negatives_hinge, 0))

    with self.test_session():
      self.assertAllClose(loss.eval(), expected.eval())

  def test3DLogitsAndTargets(self):
    # Tests correctness when logits is 3D and targets is 2D.
    targets_shape = [30, 4]
    logits_shape = [targets_shape[0], targets_shape[1], 3]
    targets = tf.to_float(
        tf.constant(np.random.random_sample(targets_shape) > 0.7))
    logits = tf.constant(np.random.randn(*logits_shape).astype(np.float32))
    weight_vector = [1.0, 1.0, 1.0]
    loss = util.weighted_hinge_loss(targets, logits, weight_vector)

    with self.test_session():
      loss_value = loss.eval()
      for i in range(len(weight_vector)):
        expected = hinge_loss(targets, logits[:, :, i]).eval()
        self.assertAllClose(loss_value[:, :, i], expected)


class BuildLabelPriorsTest(tf.test.TestCase):

  def testLabelPriorConsistency(self):
    # Checks that, with zero pseudocounts, the returned label priors reproduce
    # label frequencies in the batch.
    batch_shape = [4, 10]
    labels = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.678)))

    label_priors_update = util.build_label_priors(
        labels=labels, positive_pseudocount=0, negative_pseudocount=0)
    expected_priors = tf.reduce_mean(labels, 0)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(label_priors_update.eval(), expected_priors.eval())

  def testLabelPriorsUpdate(self):
    # Checks that the update of label priors behaves as expected.
    batch_shape = [1, 5]
    labels = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))
    label_priors_update = util.build_label_priors(labels)

    label_sum = np.ones(shape=batch_shape)
    weight_sum = 2.0 * np.ones(shape=batch_shape)

    with self.test_session() as session:
      tf.global_variables_initializer().run()

      for _ in range(3):
        label_sum += labels.eval()
        weight_sum += np.ones(shape=batch_shape)
        expected_posteriors = label_sum / weight_sum
        label_priors = label_priors_update.eval().reshape(batch_shape)
        self.assertAllClose(label_priors, expected_posteriors)

        # Re-initialize labels to get a new random sample.
        session.run(labels.initializer)

  def testLabelPriorsUpdateWithWeights(self):
    # Checks the update of label priors with per-example weights.
    batch_size = 6
    num_labels = 5
    batch_shape = [batch_size, num_labels]
    labels = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.6)))
    weights = tf.Variable(tf.random_uniform(batch_shape) * 6.2)

    update_op = util.build_label_priors(labels, weights=weights)

    expected_weighted_label_counts = 1.0 + tf.reduce_sum(weights * labels, 0)
    expected_weight_sum = 2.0 + tf.reduce_sum(weights, 0)
    expected_label_posteriors = tf.divide(expected_weighted_label_counts,
                                          expected_weight_sum)

    with self.test_session() as session:
      tf.global_variables_initializer().run()

      updated_priors, expected_posteriors = session.run(
          [update_op, expected_label_posteriors])
      self.assertAllClose(updated_priors, expected_posteriors)


class WeightedSurrogateLossTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('hinge', util.weighted_hinge_loss),
      ('xent', util.weighted_sigmoid_cross_entropy_with_logits))
  def testCompatibilityLoss(self, loss_name, loss_fn):
    x_shape = [28, 4]
    logits = tf.constant(np.random.randn(*x_shape).astype(np.float32))
    targets = tf.to_float(tf.constant(np.random.random_sample(x_shape) > 0.5))
    positive_weights = 0.66
    negative_weights = 11.1
    expected_loss = loss_fn(
        targets,
        logits,
        positive_weights=positive_weights,
        negative_weights=negative_weights)
    computed_loss = util.weighted_surrogate_loss(
        targets,
        logits,
        loss_name,
        positive_weights=positive_weights,
        negative_weights=negative_weights)
    with self.test_session():
      self.assertAllClose(expected_loss.eval(), computed_loss.eval())

  def testSurrogatgeError(self):
    x_shape = [7, 3]
    logits = tf.constant(np.random.randn(*x_shape).astype(np.float32))
    targets = tf.to_float(tf.constant(np.random.random_sample(x_shape) > 0.5))

    with self.assertRaises(ValueError):
      util.weighted_surrogate_loss(logits, targets, 'bug')


if __name__ == '__main__':
  tf.test.main()
