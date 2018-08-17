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
"""Tests for global objectives loss layers."""

# Dependency imports
from absl.testing import parameterized
import numpy
import tensorflow as tf

from global_objectives import loss_layers
from global_objectives import util


# TODO: Include weights in the lagrange multiplier update tests.
class PrecisionRecallAUCLossTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('_xent', 'xent', 0.7),
      ('_hinge', 'hinge', 0.7),
      ('_hinge_2', 'hinge', 0.5)
  )
  def testSinglePointAUC(self, surrogate_type, target_precision):
    # Tests a case with only one anchor point, where the loss should equal
    # recall_at_precision_loss
    batch_shape = [10, 2]
    logits = tf.Variable(tf.random_normal(batch_shape))
    labels = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))

    auc_loss, _ = loss_layers.precision_recall_auc_loss(
        labels,
        logits,
        precision_range=(target_precision - 0.01, target_precision  + 0.01),
        num_anchors=1,
        surrogate_type=surrogate_type)
    point_loss, _ = loss_layers.recall_at_precision_loss(
        labels, logits, target_precision=target_precision,
        surrogate_type=surrogate_type)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(auc_loss.eval(), point_loss.eval())

  def testThreePointAUC(self):
    # Tests a case with three anchor points against a weighted sum of recall
    # at precision losses.
    batch_shape = [11, 3]
    logits = tf.Variable(tf.random_normal(batch_shape))
    labels = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))

    # TODO: Place the hing/xent loss in a for loop.
    auc_loss, _ = loss_layers.precision_recall_auc_loss(
        labels, logits, num_anchors=1)
    first_point_loss, _ = loss_layers.recall_at_precision_loss(
        labels, logits, target_precision=0.25)
    second_point_loss, _ = loss_layers.recall_at_precision_loss(
        labels, logits, target_precision=0.5)
    third_point_loss, _ = loss_layers.recall_at_precision_loss(
        labels, logits, target_precision=0.75)
    expected_loss = (first_point_loss + second_point_loss +
                     third_point_loss) / 3

    auc_loss_hinge, _ = loss_layers.precision_recall_auc_loss(
        labels, logits, num_anchors=1, surrogate_type='hinge')
    first_point_hinge, _ = loss_layers.recall_at_precision_loss(
        labels, logits, target_precision=0.25, surrogate_type='hinge')
    second_point_hinge, _ = loss_layers.recall_at_precision_loss(
        labels, logits, target_precision=0.5, surrogate_type='hinge')
    third_point_hinge, _ = loss_layers.recall_at_precision_loss(
        labels, logits, target_precision=0.75, surrogate_type='hinge')
    expected_hinge = (first_point_hinge + second_point_hinge +
                      third_point_hinge) / 3

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(auc_loss.eval(), expected_loss.eval())
      self.assertAllClose(auc_loss_hinge.eval(), expected_hinge.eval())

  def testLagrangeMultiplierUpdateDirection(self):
    for target_precision in [0.35, 0.65]:
      precision_range = (target_precision - 0.01, target_precision + 0.01)

      for surrogate_type in ['xent', 'hinge']:
        kwargs = {'precision_range': precision_range,
                  'num_anchors': 1,
                  'surrogate_type': surrogate_type,
                  'scope': 'pr-auc_{}_{}'.format(target_precision,
                                                 surrogate_type)}
        run_lagrange_multiplier_test(
            global_objective=loss_layers.precision_recall_auc_loss,
            objective_kwargs=kwargs,
            data_builder=_multilabel_data,
            test_object=self)
        kwargs['scope'] = 'other-' + kwargs['scope']
        run_lagrange_multiplier_test(
            global_objective=loss_layers.precision_recall_auc_loss,
            objective_kwargs=kwargs,
            data_builder=_other_multilabel_data(surrogate_type),
            test_object=self)


class ROCAUCLossTest(parameterized.TestCase, tf.test.TestCase):

  def testSimpleScores(self):
    # Tests the loss on data with only one negative example with score zero.
    # In this case, the loss should equal the surrogate loss on the scores with
    # positive labels.
    num_positives = 10
    scores_positives = tf.constant(3.0 * numpy.random.randn(num_positives),
                                   shape=[num_positives, 1])
    labels = tf.constant([0.0] + [1.0] * num_positives,
                         shape=[num_positives + 1, 1])
    scores = tf.concat([[[0.0]], scores_positives], 0)

    loss = tf.reduce_sum(
        loss_layers.roc_auc_loss(labels, scores, surrogate_type='hinge')[0])
    expected_loss = tf.reduce_sum(
        tf.maximum(1.0 - scores_positives, 0)) / (num_positives + 1)
    with self.test_session():
      self.assertAllClose(expected_loss.eval(), loss.eval())

  def testRandomROCLoss(self):
    # Checks that random Bernoulli scores and labels has ~25% swaps.
    shape = [1000, 30]
    scores = tf.constant(
        numpy.random.randint(0, 2, size=shape), shape=shape, dtype=tf.float32)
    labels = tf.constant(
        numpy.random.randint(0, 2, size=shape), shape=shape, dtype=tf.float32)
    loss = tf.reduce_mean(loss_layers.roc_auc_loss(
        labels, scores, surrogate_type='hinge')[0])
    with self.test_session():
      self.assertAllClose(0.25, loss.eval(), 1e-2)

  @parameterized.named_parameters(
      ('_zero_hinge', 'xent',
       [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
       [-5.0, -7.0, -9.0, 8.0, 10.0, 14.0],
       0.0),
      ('_zero_xent', 'hinge',
       [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
       [-0.2, 0, -0.1, 1.0, 1.1, 1.0],
       0.0),
      ('_xent', 'xent',
       [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
       [0.0, -17.0, -19.0, 1.0, 14.0, 14.0],
       numpy.log(1.0 + numpy.exp(-1.0)) / 6),
      ('_hinge', 'hinge',
       [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
       [-0.2, -0.05, 0.0, 0.95, 0.8, 1.0],
       0.4 / 6)
  )
  def testManualROCLoss(self, surrogate_type, labels, logits, expected_value):
    labels = tf.constant(labels)
    logits = tf.constant(logits)
    loss, _ = loss_layers.roc_auc_loss(
        labels=labels, logits=logits, surrogate_type=surrogate_type)

    with self.test_session():
      self.assertAllClose(expected_value, tf.reduce_sum(loss).eval())

  def testMultiLabelROCLoss(self):
    # Tests the loss on multi-label data against manually computed loss.
    targets = numpy.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    scores = numpy.array([[0.1, 1.0, 1.1, 1.0], [1.0, 0.0, 1.3, 1.1]])
    class_1_auc = tf.reduce_sum(
        loss_layers.roc_auc_loss(targets[0], scores[0])[0])
    class_2_auc = tf.reduce_sum(
        loss_layers.roc_auc_loss(targets[1], scores[1])[0])
    total_auc = tf.reduce_sum(loss_layers.roc_auc_loss(
        targets.transpose(), scores.transpose())[0])

    with self.test_session():
      self.assertAllClose(total_auc.eval(),
                          class_1_auc.eval() + class_2_auc.eval())

  def testWeights(self):
    # Test the loss with per-example weights.
    # The logits_negatives below are repeated, so that setting half their
    # weights to 2 and the other half to 0 should leave the loss unchanged.
    logits_positives = tf.constant([2.54321, -0.26, 3.334334], shape=[3, 1])
    logits_negatives = tf.constant([-0.6, 1, -1.3, -1.3, -0.6, 1], shape=[6, 1])
    logits = tf.concat([logits_positives, logits_negatives], 0)
    targets = tf.constant([1, 1, 1, 0, 0, 0, 0, 0, 0],
                          shape=[9, 1], dtype=tf.float32)
    weights = tf.constant([1, 1, 1, 0, 0, 0, 2, 2, 2],
                          shape=[9, 1], dtype=tf.float32)

    loss = tf.reduce_sum(loss_layers.roc_auc_loss(targets, logits)[0])
    weighted_loss = tf.reduce_sum(
        loss_layers.roc_auc_loss(targets, logits, weights)[0])

    with self.test_session():
      self.assertAllClose(loss.eval(), weighted_loss.eval())


class RecallAtPrecisionTest(tf.test.TestCase):

  def testEqualWeightLoss(self):
    # Tests a special case where the loss should equal cross entropy loss.
    target_precision = 1.0
    num_labels = 5
    batch_shape = [20, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.7)))
    label_priors = tf.constant(0.34, shape=[num_labels])

    loss, _ = loss_layers.recall_at_precision_loss(
        targets, logits, target_precision, label_priors=label_priors)
    expected_loss = (
        tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
            logits, targets))

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      loss_val, expected_val = session.run([loss, expected_loss])
      self.assertAllClose(loss_val, expected_val)

  def testEqualWeightLossWithMultiplePrecisions(self):
    """Tests a case where the loss equals xent loss with multiple precisions."""
    target_precision = [1.0, 1.0]
    num_labels = 2
    batch_size = 20
    target_shape = [batch_size, num_labels]
    logits = tf.Variable(tf.random_normal(target_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(target_shape), 0.7)))
    label_priors = tf.constant([0.34], shape=[num_labels])

    loss, _ = loss_layers.recall_at_precision_loss(
        targets,
        logits,
        target_precision,
        label_priors=label_priors,
        surrogate_type='xent',
    )

    expected_loss = (
        tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
            logits, targets))

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      loss_val, expected_val = session.run([loss, expected_loss])
      self.assertAllClose(loss_val, expected_val)

  def testPositivesOnlyLoss(self):
    # Tests a special case where the loss should equal cross entropy loss
    # on the negatives only.
    target_precision = 1.0
    num_labels = 3
    batch_shape = [30, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))
    label_priors = tf.constant(0.45, shape=[num_labels])

    loss, _ = loss_layers.recall_at_precision_loss(
        targets, logits, target_precision, label_priors=label_priors,
        lambdas_initializer=tf.zeros_initializer())
    expected_loss = util.weighted_sigmoid_cross_entropy_with_logits(
        targets,
        logits,
        positive_weights=1.0,
        negative_weights=0.0)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      loss_val, expected_val = session.run([loss, expected_loss])
      self.assertAllClose(loss_val, expected_val)

  def testEquivalenceBetweenSingleAndMultiplePrecisions(self):
    """Checks recall at precision with different precision values.

    Runs recall at precision with multiple precision values, and runs each label
    seperately with its own precision value as a scalar. Validates that the
    returned loss values are the same.
    """
    target_precision = [0.2, 0.9, 0.4]
    num_labels = 3
    batch_shape = [30, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))
    label_priors = tf.constant([0.45, 0.8, 0.3], shape=[num_labels])

    multi_label_loss, _ = loss_layers.recall_at_precision_loss(
        targets, logits, target_precision, label_priors=label_priors,
    )

    single_label_losses = [
        loss_layers.recall_at_precision_loss(
            tf.expand_dims(targets[:, i], -1),
            tf.expand_dims(logits[:, i], -1),
            target_precision[i],
            label_priors=label_priors[i])[0]
        for i in range(num_labels)
    ]

    single_label_losses = tf.concat(single_label_losses, 1)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      multi_label_loss_val, single_label_loss_val = session.run(
          [multi_label_loss, single_label_losses])
      self.assertAllClose(multi_label_loss_val, single_label_loss_val)

  def testEquivalenceBetweenSingleAndEqualMultiplePrecisions(self):
    """Compares single and multiple target precisions with the same value.

    Checks that using a single target precision and multiple target precisions
    with the same value would result in the same loss value.
    """
    num_labels = 2
    target_shape = [20, num_labels]
    logits = tf.Variable(tf.random_normal(target_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(target_shape), 0.7)))
    label_priors = tf.constant([0.34], shape=[num_labels])

    multi_precision_loss, _ = loss_layers.recall_at_precision_loss(
        targets,
        logits,
        [0.75, 0.75],
        label_priors=label_priors,
        surrogate_type='xent',
    )

    single_precision_loss, _ = loss_layers.recall_at_precision_loss(
        targets,
        logits,
        0.75,
        label_priors=label_priors,
        surrogate_type='xent',
    )

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      multi_precision_loss_val, single_precision_loss_val = session.run(
          [multi_precision_loss, single_precision_loss])
      self.assertAllClose(multi_precision_loss_val, single_precision_loss_val)

  def testLagrangeMultiplierUpdateDirection(self):
    for target_precision in [0.35, 0.65]:
      for surrogate_type in ['xent', 'hinge']:
        kwargs = {'target_precision': target_precision,
                  'surrogate_type': surrogate_type,
                  'scope': 'r-at-p_{}_{}'.format(target_precision,
                                                 surrogate_type)}
        run_lagrange_multiplier_test(
            global_objective=loss_layers.recall_at_precision_loss,
            objective_kwargs=kwargs,
            data_builder=_multilabel_data,
            test_object=self)
        kwargs['scope'] = 'other-' + kwargs['scope']
        run_lagrange_multiplier_test(
            global_objective=loss_layers.recall_at_precision_loss,
            objective_kwargs=kwargs,
            data_builder=_other_multilabel_data(surrogate_type),
            test_object=self)

  def testLagrangeMultiplierUpdateDirectionWithMultiplePrecisions(self):
    """Runs Lagrange multiplier test with multiple precision values."""
    target_precision = [0.65, 0.35]

    for surrogate_type in ['xent', 'hinge']:
      scope_str = 'r-at-p_{}_{}'.format(
          '_'.join([str(precision) for precision in target_precision]),
          surrogate_type)
      kwargs = {
          'target_precision': target_precision,
          'surrogate_type': surrogate_type,
          'scope': scope_str,
      }
      run_lagrange_multiplier_test(
          global_objective=loss_layers.recall_at_precision_loss,
          objective_kwargs=kwargs,
          data_builder=_multilabel_data,
          test_object=self)
      kwargs['scope'] = 'other-' + kwargs['scope']
      run_lagrange_multiplier_test(
          global_objective=loss_layers.recall_at_precision_loss,
          objective_kwargs=kwargs,
          data_builder=_other_multilabel_data(surrogate_type),
          test_object=self)


class PrecisionAtRecallTest(tf.test.TestCase):

  def testCrossEntropyEquivalence(self):
    # Checks a special case where the loss should equal cross-entropy loss.
    target_recall = 1.0
    num_labels = 3
    batch_shape = [10, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))

    loss, _ = loss_layers.precision_at_recall_loss(
        targets, logits, target_recall,
        lambdas_initializer=tf.constant_initializer(1.0))
    expected_loss = util.weighted_sigmoid_cross_entropy_with_logits(
        targets, logits)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(loss.eval(), expected_loss.eval())

  def testNegativesOnlyLoss(self):
    # Checks a special case where the loss should equal the loss on
    # the negative examples only.
    target_recall = 0.61828
    num_labels = 4
    batch_shape = [8, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.6)))

    loss, _ = loss_layers.precision_at_recall_loss(
        targets,
        logits,
        target_recall,
        surrogate_type='hinge',
        lambdas_initializer=tf.constant_initializer(0.0),
        scope='negatives_only_test')
    expected_loss = util.weighted_hinge_loss(
        targets, logits, positive_weights=0.0, negative_weights=1.0)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(expected_loss.eval(), loss.eval())

  def testLagrangeMultiplierUpdateDirection(self):
    for target_recall in [0.34, 0.66]:
      for surrogate_type in ['xent', 'hinge']:
        kwargs = {'target_recall': target_recall,
                  'dual_rate_factor': 1.0,
                  'surrogate_type': surrogate_type,
                  'scope': 'p-at-r_{}_{}'.format(target_recall, surrogate_type)}

        run_lagrange_multiplier_test(
            global_objective=loss_layers.precision_at_recall_loss,
            objective_kwargs=kwargs,
            data_builder=_multilabel_data,
            test_object=self)
        kwargs['scope'] = 'other-' + kwargs['scope']
        run_lagrange_multiplier_test(
            global_objective=loss_layers.precision_at_recall_loss,
            objective_kwargs=kwargs,
            data_builder=_other_multilabel_data(surrogate_type),
            test_object=self)

  def testCrossEntropyEquivalenceWithMultipleRecalls(self):
    """Checks a case where the loss equals xent loss with multiple recalls."""
    num_labels = 3
    target_recall = [1.0] * num_labels
    batch_shape = [10, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))

    loss, _ = loss_layers.precision_at_recall_loss(
        targets, logits, target_recall,
        lambdas_initializer=tf.constant_initializer(1.0))
    expected_loss = util.weighted_sigmoid_cross_entropy_with_logits(
        targets, logits)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(loss.eval(), expected_loss.eval())

  def testNegativesOnlyLossWithMultipleRecalls(self):
    """Tests a case where the loss equals the loss on the negative examples.

    Checks this special case using multiple target recall values.
    """
    num_labels = 4
    target_recall = [0.61828] * num_labels
    batch_shape = [8, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.6)))

    loss, _ = loss_layers.precision_at_recall_loss(
        targets,
        logits,
        target_recall,
        surrogate_type='hinge',
        lambdas_initializer=tf.constant_initializer(0.0),
        scope='negatives_only_test')
    expected_loss = util.weighted_hinge_loss(
        targets, logits, positive_weights=0.0, negative_weights=1.0)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(expected_loss.eval(), loss.eval())

  def testLagrangeMultiplierUpdateDirectionWithMultipleRecalls(self):
    """Runs Lagrange multiplier test with multiple recall values."""
    target_recall = [0.34, 0.66]
    for surrogate_type in ['xent', 'hinge']:
      scope_str = 'p-at-r_{}_{}'.format(
          '_'.join([str(recall) for recall in target_recall]),
          surrogate_type)
      kwargs = {'target_recall': target_recall,
                'dual_rate_factor': 1.0,
                'surrogate_type': surrogate_type,
                'scope': scope_str}

      run_lagrange_multiplier_test(
          global_objective=loss_layers.precision_at_recall_loss,
          objective_kwargs=kwargs,
          data_builder=_multilabel_data,
          test_object=self)
      kwargs['scope'] = 'other-' + kwargs['scope']
      run_lagrange_multiplier_test(
          global_objective=loss_layers.precision_at_recall_loss,
          objective_kwargs=kwargs,
          data_builder=_other_multilabel_data(surrogate_type),
          test_object=self)

  def testEquivalenceBetweenSingleAndMultipleRecalls(self):
    """Checks precision at recall with multiple different recall values.

    Runs precision at recall with multiple recall values, and runs each label
    seperately with its own recall value as a scalar. Validates that the
    returned loss values are the same.
    """
    target_precision = [0.7, 0.9, 0.4]
    num_labels = 3
    batch_shape = [30, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))
    label_priors = tf.constant(0.45, shape=[num_labels])

    multi_label_loss, _ = loss_layers.precision_at_recall_loss(
        targets, logits, target_precision, label_priors=label_priors
    )

    single_label_losses = [
        loss_layers.precision_at_recall_loss(
            tf.expand_dims(targets[:, i], -1),
            tf.expand_dims(logits[:, i], -1),
            target_precision[i],
            label_priors=label_priors[i])[0]
        for i in range(num_labels)
    ]

    single_label_losses = tf.concat(single_label_losses, 1)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      multi_label_loss_val, single_label_loss_val = session.run(
          [multi_label_loss, single_label_losses])
      self.assertAllClose(multi_label_loss_val, single_label_loss_val)

  def testEquivalenceBetweenSingleAndEqualMultipleRecalls(self):
    """Compares single and multiple target recalls of the same value.

    Checks that using a single target recall and multiple recalls with the
    same value would result in the same loss value.
    """
    num_labels = 2
    target_shape = [20, num_labels]
    logits = tf.Variable(tf.random_normal(target_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(target_shape), 0.7)))
    label_priors = tf.constant([0.34], shape=[num_labels])

    multi_precision_loss, _ = loss_layers.precision_at_recall_loss(
        targets,
        logits,
        [0.75, 0.75],
        label_priors=label_priors,
        surrogate_type='xent',
    )

    single_precision_loss, _ = loss_layers.precision_at_recall_loss(
        targets,
        logits,
        0.75,
        label_priors=label_priors,
        surrogate_type='xent',
    )

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      multi_precision_loss_val, single_precision_loss_val = session.run(
          [multi_precision_loss, single_precision_loss])
      self.assertAllClose(multi_precision_loss_val, single_precision_loss_val)


class FalsePositiveRateAtTruePositiveRateTest(tf.test.TestCase):

  def testNegativesOnlyLoss(self):
    # Checks a special case where the loss returned should be the loss on the
    # negative examples.
    target_recall = 0.6
    num_labels = 3
    batch_shape = [3, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))
    label_priors = tf.constant(numpy.random.uniform(size=[num_labels]),
                               dtype=tf.float32)

    xent_loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
        targets, logits, target_recall, label_priors=label_priors,
        lambdas_initializer=tf.constant_initializer(0.0))
    xent_expected = util.weighted_sigmoid_cross_entropy_with_logits(
        targets,
        logits,
        positive_weights=0.0,
        negative_weights=1.0)
    hinge_loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
        targets, logits, target_recall, label_priors=label_priors,
        lambdas_initializer=tf.constant_initializer(0.0),
        surrogate_type='hinge')
    hinge_expected = util.weighted_hinge_loss(
        targets,
        logits,
        positive_weights=0.0,
        negative_weights=1.0)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      xent_val, xent_expected = session.run([xent_loss, xent_expected])
      self.assertAllClose(xent_val, xent_expected)
      hinge_val, hinge_expected = session.run([hinge_loss, hinge_expected])
      self.assertAllClose(hinge_val, hinge_expected)

  def testPositivesOnlyLoss(self):
    # Checks a special case where the loss returned should be the loss on the
    # positive examples only.
    target_recall = 1.0
    num_labels = 5
    batch_shape = [5, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.ones_like(logits)
    label_priors = tf.constant(numpy.random.uniform(size=[num_labels]),
                               dtype=tf.float32)

    loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
        targets, logits, target_recall, label_priors=label_priors)
    expected_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=logits)
    hinge_loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
        targets, logits, target_recall, label_priors=label_priors,
        surrogate_type='hinge')
    expected_hinge = util.weighted_hinge_loss(
        targets, logits)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(loss.eval(), expected_loss.eval())
      self.assertAllClose(hinge_loss.eval(), expected_hinge.eval())

  def testEqualWeightLoss(self):
    # Checks a special case where the loss returned should be proportional to
    # the ordinary loss.
    target_recall = 1.0
    num_labels = 4
    batch_shape = [40, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.6)))
    label_priors = tf.constant(0.5, shape=[num_labels])

    loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
        targets, logits, target_recall, label_priors=label_priors)
    expected_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=logits)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(loss.eval(), expected_loss.eval())

  def testLagrangeMultiplierUpdateDirection(self):
    for target_rate in [0.35, 0.65]:
      for surrogate_type in ['xent', 'hinge']:
        kwargs = {'target_rate': target_rate,
                  'surrogate_type': surrogate_type,
                  'scope': 'fpr-at-tpr_{}_{}'.format(target_rate,
                                                     surrogate_type)}
        # True positive rate is a synonym for recall, so we use the
        # recall constraint data.
        run_lagrange_multiplier_test(
            global_objective=(
                loss_layers.false_positive_rate_at_true_positive_rate_loss),
            objective_kwargs=kwargs,
            data_builder=_multilabel_data,
            test_object=self)
        kwargs['scope'] = 'other-' + kwargs['scope']
        run_lagrange_multiplier_test(
            global_objective=(
                loss_layers.false_positive_rate_at_true_positive_rate_loss),
            objective_kwargs=kwargs,
            data_builder=_other_multilabel_data(surrogate_type),
            test_object=self)

  def testLagrangeMultiplierUpdateDirectionWithMultipleRates(self):
    """Runs Lagrange multiplier test with multiple target rates."""
    target_rate = [0.35, 0.65]
    for surrogate_type in ['xent', 'hinge']:
      kwargs = {'target_rate': target_rate,
                'surrogate_type': surrogate_type,
                'scope': 'fpr-at-tpr_{}_{}'.format(
                    '_'.join([str(target) for target in target_rate]),
                    surrogate_type)}
      # True positive rate is a synonym for recall, so we use the
      # recall constraint data.
      run_lagrange_multiplier_test(
          global_objective=(
              loss_layers.false_positive_rate_at_true_positive_rate_loss),
          objective_kwargs=kwargs,
          data_builder=_multilabel_data,
          test_object=self)
      kwargs['scope'] = 'other-' + kwargs['scope']
      run_lagrange_multiplier_test(
          global_objective=(
              loss_layers.false_positive_rate_at_true_positive_rate_loss),
          objective_kwargs=kwargs,
          data_builder=_other_multilabel_data(surrogate_type),
          test_object=self)

  def testEquivalenceBetweenSingleAndEqualMultipleRates(self):
    """Compares single and multiple target rates of the same value.

    Checks that using a single target rate and multiple rates with the
    same value would result in the same loss value.
    """
    num_labels = 2
    target_shape = [20, num_labels]
    logits = tf.Variable(tf.random_normal(target_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(target_shape), 0.7)))
    label_priors = tf.constant([0.34], shape=[num_labels])

    multi_label_loss, _ = (
        loss_layers.false_positive_rate_at_true_positive_rate_loss(
            targets, logits, [0.75, 0.75], label_priors=label_priors))

    single_label_loss, _ = (
        loss_layers.false_positive_rate_at_true_positive_rate_loss(
            targets, logits, 0.75, label_priors=label_priors))

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      multi_label_loss_val, single_label_loss_val = session.run(
          [multi_label_loss, single_label_loss])
      self.assertAllClose(multi_label_loss_val, single_label_loss_val)

  def testEquivalenceBetweenSingleAndMultipleRates(self):
    """Compares single and multiple target rates of different values.

    Runs false_positive_rate_at_true_positive_rate_loss with multiple target
    rates, and runs each label seperately with its own target rate as a
    scalar. Validates that the returned loss values are the same.
    """
    target_precision = [0.7, 0.9, 0.4]
    num_labels = 3
    batch_shape = [30, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))
    label_priors = tf.constant(0.45, shape=[num_labels])

    multi_label_loss, _ = (
        loss_layers.false_positive_rate_at_true_positive_rate_loss(
            targets, logits, target_precision, label_priors=label_priors))

    single_label_losses = [
        loss_layers.false_positive_rate_at_true_positive_rate_loss(
            tf.expand_dims(targets[:, i], -1),
            tf.expand_dims(logits[:, i], -1),
            target_precision[i],
            label_priors=label_priors[i])[0]
        for i in range(num_labels)
    ]

    single_label_losses = tf.concat(single_label_losses, 1)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      multi_label_loss_val, single_label_loss_val = session.run(
          [multi_label_loss, single_label_losses])
      self.assertAllClose(multi_label_loss_val, single_label_loss_val)


class TruePositiveRateAtFalsePositiveRateTest(tf.test.TestCase):

  def testPositivesOnlyLoss(self):
    # A special case where the loss should equal the loss on the positive
    # examples.
    target_rate = numpy.random.uniform()
    num_labels = 3
    batch_shape = [20, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.6)))
    label_priors = tf.constant(numpy.random.uniform(size=[num_labels]),
                               dtype=tf.float32)

    xent_loss, _ = loss_layers.true_positive_rate_at_false_positive_rate_loss(
        targets, logits, target_rate, label_priors=label_priors,
        lambdas_initializer=tf.constant_initializer(0.0))
    xent_expected = util.weighted_sigmoid_cross_entropy_with_logits(
        targets,
        logits,
        positive_weights=1.0,
        negative_weights=0.0)
    hinge_loss, _ = loss_layers.true_positive_rate_at_false_positive_rate_loss(
        targets, logits, target_rate, label_priors=label_priors,
        lambdas_initializer=tf.constant_initializer(0.0),
        surrogate_type='hinge')
    hinge_expected = util.weighted_hinge_loss(
        targets,
        logits,
        positive_weights=1.0,
        negative_weights=0.0)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(xent_expected.eval(), xent_loss.eval())
      self.assertAllClose(hinge_expected.eval(), hinge_loss.eval())

  def testNegativesOnlyLoss(self):
    # A special case where the loss should equal the loss on the negative
    # examples, minus target_rate * (1 - label_priors) * maybe_log2.
    target_rate = numpy.random.uniform()
    num_labels = 3
    batch_shape = [25, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.zeros_like(logits)
    label_priors = tf.constant(numpy.random.uniform(size=[num_labels]),
                               dtype=tf.float32)

    xent_loss, _ = loss_layers.true_positive_rate_at_false_positive_rate_loss(
        targets, logits, target_rate, label_priors=label_priors)
    xent_expected = tf.subtract(
        util.weighted_sigmoid_cross_entropy_with_logits(targets,
                                                        logits,
                                                        positive_weights=0.0,
                                                        negative_weights=1.0),
        target_rate * (1.0 - label_priors) * numpy.log(2))
    hinge_loss, _ = loss_layers.true_positive_rate_at_false_positive_rate_loss(
        targets, logits, target_rate, label_priors=label_priors,
        surrogate_type='hinge')
    hinge_expected = util.weighted_hinge_loss(
        targets, logits) - target_rate * (1.0 - label_priors)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(xent_expected.eval(), xent_loss.eval())
      self.assertAllClose(hinge_expected.eval(), hinge_loss.eval())

  def testLagrangeMultiplierUpdateDirection(self):
    for target_rate in [0.35, 0.65]:
      for surrogate_type in ['xent', 'hinge']:
        kwargs = {'target_rate': target_rate,
                  'surrogate_type': surrogate_type,
                  'scope': 'tpr-at-fpr_{}_{}'.format(target_rate,
                                                     surrogate_type)}
        run_lagrange_multiplier_test(
            global_objective=(
                loss_layers.true_positive_rate_at_false_positive_rate_loss),
            objective_kwargs=kwargs,
            data_builder=_multilabel_data,
            test_object=self)
        kwargs['scope'] = 'other-' + kwargs['scope']
        run_lagrange_multiplier_test(
            global_objective=(
                loss_layers.true_positive_rate_at_false_positive_rate_loss),
            objective_kwargs=kwargs,
            data_builder=_other_multilabel_data(surrogate_type),
            test_object=self)

  def testLagrangeMultiplierUpdateDirectionWithMultipleRates(self):
    """Runs Lagrange multiplier test with multiple target rates."""
    target_rate = [0.35, 0.65]
    for surrogate_type in ['xent', 'hinge']:
      kwargs = {'target_rate': target_rate,
                'surrogate_type': surrogate_type,
                'scope': 'tpr-at-fpr_{}_{}'.format(
                    '_'.join([str(target) for target in target_rate]),
                    surrogate_type)}
      run_lagrange_multiplier_test(
          global_objective=(
              loss_layers.true_positive_rate_at_false_positive_rate_loss),
          objective_kwargs=kwargs,
          data_builder=_multilabel_data,
          test_object=self)
      kwargs['scope'] = 'other-' + kwargs['scope']
      run_lagrange_multiplier_test(
          global_objective=(
              loss_layers.true_positive_rate_at_false_positive_rate_loss),
          objective_kwargs=kwargs,
          data_builder=_other_multilabel_data(surrogate_type),
          test_object=self)

  def testEquivalenceBetweenSingleAndEqualMultipleRates(self):
    """Compares single and multiple target rates of the same value.

    Checks that using a single target rate and multiple rates with the
    same value would result in the same loss value.
    """
    num_labels = 2
    target_shape = [20, num_labels]
    logits = tf.Variable(tf.random_normal(target_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(target_shape), 0.7)))
    label_priors = tf.constant([0.34], shape=[num_labels])

    multi_label_loss, _ = (
        loss_layers.true_positive_rate_at_false_positive_rate_loss(
            targets, logits, [0.75, 0.75], label_priors=label_priors))

    single_label_loss, _ = (
        loss_layers.true_positive_rate_at_false_positive_rate_loss(
            targets, logits, 0.75, label_priors=label_priors))

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      multi_label_loss_val, single_label_loss_val = session.run(
          [multi_label_loss, single_label_loss])
      self.assertAllClose(multi_label_loss_val, single_label_loss_val)

  def testEquivalenceBetweenSingleAndMultipleRates(self):
    """Compares single and multiple target rates of different values.

    Runs true_positive_rate_at_false_positive_rate_loss with multiple target
    rates, and runs each label seperately with its own target rate as a
    scalar. Validates that the returned loss values are the same.
    """
    target_precision = [0.7, 0.9, 0.4]
    num_labels = 3
    batch_shape = [30, num_labels]
    logits = tf.Variable(tf.random_normal(batch_shape))
    targets = tf.Variable(
        tf.to_float(tf.greater(tf.random_uniform(batch_shape), 0.4)))
    label_priors = tf.constant(0.45, shape=[num_labels])

    multi_label_loss, _ = (
        loss_layers.true_positive_rate_at_false_positive_rate_loss(
            targets, logits, target_precision, label_priors=label_priors))

    single_label_losses = [
        loss_layers.true_positive_rate_at_false_positive_rate_loss(
            tf.expand_dims(targets[:, i], -1),
            tf.expand_dims(logits[:, i], -1),
            target_precision[i],
            label_priors=label_priors[i])[0]
        for i in range(num_labels)
    ]

    single_label_losses = tf.concat(single_label_losses, 1)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      multi_label_loss_val, single_label_loss_val = session.run(
          [multi_label_loss, single_label_losses])
      self.assertAllClose(multi_label_loss_val, single_label_loss_val)


class UtilityFunctionsTest(tf.test.TestCase):

  def testTrainableDualVariable(self):
    # Confirm correct behavior of a trainable dual variable.
    x = tf.get_variable('primal', dtype=tf.float32, initializer=2.0)
    y_value, y = loss_layers._create_dual_variable(
        'dual', shape=None, dtype=tf.float32, initializer=1.0, collections=None,
        trainable=True, dual_rate_factor=0.3)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    update = optimizer.minimize(0.5 * tf.square(x - y_value))

    with self.test_session():
      tf.global_variables_initializer().run()
      update.run()
      self.assertAllClose(0.7, y.eval())

  def testUntrainableDualVariable(self):
    # Confirm correct behavior of dual variable which is not trainable.
    x = tf.get_variable('primal', dtype=tf.float32, initializer=-2.0)
    y_value, y = loss_layers._create_dual_variable(
        'dual', shape=None, dtype=tf.float32, initializer=1.0, collections=None,
        trainable=False, dual_rate_factor=0.8)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    update = optimizer.minimize(tf.square(x) * y_value + tf.exp(y_value))

    with self.test_session():
      tf.global_variables_initializer().run()
      update.run()
      self.assertAllClose(1.0, y.eval())


class BoundTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('_xent', 'xent', 1.0, [2.0, 1.0]),
      ('_xent_weighted', 'xent',
       numpy.array([0, 2, 0.5, 1, 2, 3]).reshape(6, 1), [2.5, 0]),
      ('_hinge', 'hinge', 1.0, [2.0, 1.0]),
      ('_hinge_weighted', 'hinge',
       numpy.array([1.0, 2, 3, 4, 5, 6]).reshape(6, 1), [5.0, 1]))
  def testLowerBoundMultilabel(self, surrogate_type, weights, expected):
    labels, logits, _ = _multilabel_data()
    lower_bound = loss_layers.true_positives_lower_bound(
        labels, logits, weights, surrogate_type)

    with self.test_session():
      self.assertAllClose(lower_bound.eval(), expected)

  @parameterized.named_parameters(
      ('_xent', 'xent'), ('_hinge', 'hinge'))
  def testLowerBoundOtherMultilabel(self, surrogate_type):
    labels, logits, _ = _other_multilabel_data(surrogate_type)()
    lower_bound = loss_layers.true_positives_lower_bound(
        labels, logits, 1.0, surrogate_type)

    with self.test_session():
      self.assertAllClose(lower_bound.eval(), [4.0, 2.0], atol=1e-5)

  @parameterized.named_parameters(
      ('_xent', 'xent', 1.0, [1.0, 2.0]),
      ('_xent_weighted', 'xent',
       numpy.array([3.0, 2, 1, 0, 1, 2]).reshape(6, 1), [2.0, 1.0]),
      ('_hinge', 'hinge', 1.0, [1.0, 2.0]),
      ('_hinge_weighted', 'hinge',
       numpy.array([13, 12, 11, 0.5, 0, 0.5]).reshape(6, 1), [0.5, 0.5]))
  def testUpperBoundMultilabel(self, surrogate_type, weights, expected):
    labels, logits, _ = _multilabel_data()
    upper_bound = loss_layers.false_positives_upper_bound(
        labels, logits, weights, surrogate_type)

    with self.test_session():
      self.assertAllClose(upper_bound.eval(), expected)

  @parameterized.named_parameters(
      ('_xent', 'xent'), ('_hinge', 'hinge'))
  def testUpperBoundOtherMultilabel(self, surrogate_type):
    labels, logits, _ = _other_multilabel_data(surrogate_type)()
    upper_bound = loss_layers.false_positives_upper_bound(
        labels, logits, 1.0, surrogate_type)

    with self.test_session():
      self.assertAllClose(upper_bound.eval(), [2.0, 4.0], atol=1e-5)

  @parameterized.named_parameters(
      ('_lower', 'lower'), ('_upper', 'upper'))
  def testThreeDimensionalLogits(self, bound):
    bound_function = loss_layers.false_positives_upper_bound
    if bound == 'lower':
      bound_function = loss_layers.true_positives_lower_bound
    random_labels = numpy.float32(numpy.random.uniform(size=[2, 3]) > 0.5)
    random_logits = numpy.float32(numpy.random.randn(2, 3, 2))
    first_slice_logits = random_logits[:, :, 0].reshape(2, 3)
    second_slice_logits = random_logits[:, :, 1].reshape(2, 3)

    full_bound = bound_function(
        tf.constant(random_labels), tf.constant(random_logits), 1.0, 'xent')
    first_slice_bound = bound_function(tf.constant(random_labels),
                                       tf.constant(first_slice_logits),
                                       1.0,
                                       'xent')
    second_slice_bound = bound_function(tf.constant(random_labels),
                                        tf.constant(second_slice_logits),
                                        1.0,
                                        'xent')
    stacked_bound = tf.stack([first_slice_bound, second_slice_bound], axis=1)

    with self.test_session():
      self.assertAllClose(full_bound.eval(), stacked_bound.eval())


def run_lagrange_multiplier_test(global_objective,
                                 objective_kwargs,
                                 data_builder,
                                 test_object):
  """Runs a test for the Lagrange multiplier update of `global_objective`.

  The test checks that the constraint for `global_objective` is satisfied on
  the first label of the data produced by `data_builder` but not the second.

  Args:
    global_objective: One of the global objectives.
    objective_kwargs: A dictionary of keyword arguments to pass to
      `global_objective`. Must contain an entry for the constraint argument
      of `global_objective`, e.g. 'target_rate' or 'target_precision'.
    data_builder: A function  which returns tensors corresponding to labels,
      logits, and label priors.
    test_object: An instance of tf.test.TestCase.
  """
  # Construct global objective kwargs from a copy of `objective_kwargs`.
  kwargs = dict(objective_kwargs)
  targets, logits, priors = data_builder()
  kwargs['labels'] = targets
  kwargs['logits'] = logits
  kwargs['label_priors'] = priors

  loss, output_dict = global_objective(**kwargs)
  lambdas = tf.squeeze(output_dict['lambdas'])
  opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
  update_op = opt.minimize(loss, var_list=[output_dict['lambdas']])

  with test_object.test_session() as session:
    tf.global_variables_initializer().run()
    lambdas_before = session.run(lambdas)
    session.run(update_op)
    lambdas_after = session.run(lambdas)
    test_object.assertLess(lambdas_after[0], lambdas_before[0])
    test_object.assertGreater(lambdas_after[1], lambdas_before[1])


class CrossFunctionTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('_auc01xent', loss_layers.precision_recall_auc_loss, {
          'precision_range': (0.0, 1.0), 'surrogate_type': 'xent'
      }),
      ('_auc051xent', loss_layers.precision_recall_auc_loss, {
          'precision_range': (0.5, 1.0), 'surrogate_type': 'xent'
      }),
      ('_auc01)hinge', loss_layers.precision_recall_auc_loss, {
          'precision_range': (0.0, 1.0), 'surrogate_type': 'hinge'
      }),
      ('_ratp04', loss_layers.recall_at_precision_loss, {
          'target_precision': 0.4, 'surrogate_type': 'xent'
      }),
      ('_ratp066', loss_layers.recall_at_precision_loss, {
          'target_precision': 0.66, 'surrogate_type': 'xent'
      }),
      ('_ratp07_hinge', loss_layers.recall_at_precision_loss, {
          'target_precision': 0.7, 'surrogate_type': 'hinge'
      }),
      ('_fpattp066', loss_layers.false_positive_rate_at_true_positive_rate_loss,
       {'target_rate': 0.66, 'surrogate_type': 'xent'}),
      ('_fpattp046', loss_layers.false_positive_rate_at_true_positive_rate_loss,
       {
           'target_rate': 0.46, 'surrogate_type': 'xent'
       }),
      ('_fpattp076_hinge',
       loss_layers.false_positive_rate_at_true_positive_rate_loss, {
           'target_rate': 0.76, 'surrogate_type': 'hinge'
       }),
      ('_fpattp036_hinge',
       loss_layers.false_positive_rate_at_true_positive_rate_loss, {
           'target_rate': 0.36, 'surrogate_type': 'hinge'
       }),
  )
  def testWeigtedGlobalObjective(self,
                                 global_objective,
                                 objective_kwargs):
    """Runs a test of `global_objective` with per-example weights.

    Args:
      global_objective: One of the global objectives.
      objective_kwargs: A dictionary of keyword arguments to pass to
        `global_objective`. Must contain keys 'surrogate_type', and the keyword
        for the constraint argument of `global_objective`, e.g. 'target_rate' or
        'target_precision'.
    """
    logits_positives = tf.constant([1, -0.5, 3], shape=[3, 1])
    logits_negatives = tf.constant([-0.5, 1, -1, -1, -0.5, 1], shape=[6, 1])

    # Dummy tensor is used to compute the gradients.
    dummy = tf.constant(1.0)
    logits = tf.concat([logits_positives, logits_negatives], 0)
    logits = tf.multiply(logits, dummy)
    targets = tf.constant([1, 1, 1, 0, 0, 0, 0, 0, 0],
                          shape=[9, 1], dtype=tf.float32)
    priors = tf.constant(1.0/3.0, shape=[1])
    weights = tf.constant([1, 1, 1, 0, 0, 0, 2, 2, 2],
                          shape=[9, 1], dtype=tf.float32)

    # Construct global objective kwargs.
    objective_kwargs['labels'] = targets
    objective_kwargs['logits'] = logits
    objective_kwargs['label_priors'] = priors

    scope = 'weighted_test'
    # Unweighted loss.
    objective_kwargs['scope'] = scope + '_plain'
    raw_loss, update = global_objective(**objective_kwargs)
    loss = tf.reduce_sum(raw_loss)

    # Weighted loss.
    objective_kwargs['weights'] = weights
    objective_kwargs['scope'] = scope + '_weighted'
    raw_weighted_loss, weighted_update = global_objective(**objective_kwargs)
    weighted_loss = tf.reduce_sum(raw_weighted_loss)

    lambdas = tf.contrib.framework.get_unique_variable(scope + '_plain/lambdas')
    weighted_lambdas = tf.contrib.framework.get_unique_variable(
        scope + '_weighted/lambdas')
    logits_gradient = tf.gradients(loss, dummy)
    weighted_logits_gradient = tf.gradients(weighted_loss, dummy)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      self.assertAllClose(loss.eval(), weighted_loss.eval())

      logits_grad, weighted_logits_grad = session.run(
          [logits_gradient, weighted_logits_gradient])
      self.assertAllClose(logits_grad, weighted_logits_grad)

      session.run([update, weighted_update])
      lambdas_value, weighted_lambdas_value = session.run(
          [lambdas, weighted_lambdas])
      self.assertAllClose(lambdas_value, weighted_lambdas_value)

  @parameterized.named_parameters(
      ('_prauc051xent', loss_layers.precision_recall_auc_loss, {
          'precision_range': (0.5, 1.0), 'surrogate_type': 'xent'
      }),
      ('_prauc01hinge', loss_layers.precision_recall_auc_loss, {
          'precision_range': (0.0, 1.0), 'surrogate_type': 'hinge'
      }),
      ('_rocxent', loss_layers.roc_auc_loss, {'surrogate_type': 'xent'}),
      ('_rochinge', loss_layers.roc_auc_loss, {'surrogate_type': 'xent'}),
      ('_ratp04', loss_layers.recall_at_precision_loss, {
          'target_precision': 0.4, 'surrogate_type': 'xent'
      }),
      ('_ratp07_hinge', loss_layers.recall_at_precision_loss, {
          'target_precision': 0.7, 'surrogate_type': 'hinge'
      }),
      ('_patr05', loss_layers.precision_at_recall_loss, {
          'target_recall': 0.4, 'surrogate_type': 'xent'
      }),
      ('_patr08_hinge', loss_layers.precision_at_recall_loss, {
          'target_recall': 0.7, 'surrogate_type': 'hinge'
      }),
      ('_fpattp046', loss_layers.false_positive_rate_at_true_positive_rate_loss,
       {
           'target_rate': 0.46, 'surrogate_type': 'xent'
       }),
      ('_fpattp036_hinge',
       loss_layers.false_positive_rate_at_true_positive_rate_loss, {
           'target_rate': 0.36, 'surrogate_type': 'hinge'
       }),
      ('_tpatfp076', loss_layers.true_positive_rate_at_false_positive_rate_loss,
       {
           'target_rate': 0.76, 'surrogate_type': 'xent'
       }),
      ('_tpatfp036_hinge',
       loss_layers.true_positive_rate_at_false_positive_rate_loss, {
           'target_rate': 0.36, 'surrogate_type': 'hinge'
       }),
  )
  def testVectorAndMatrixLabelEquivalence(self,
                                          global_objective,
                                          objective_kwargs):
    """Tests equivalence between label shape [batch_size] or [batch_size, 1]."""
    vector_labels = tf.constant([1.0, 1.0, 0.0, 0.0], shape=[4])
    vector_logits = tf.constant([1.0, 0.1, 0.1, -1.0], shape=[4])

    # Construct vector global objective kwargs and loss.
    vector_kwargs = objective_kwargs.copy()
    vector_kwargs['labels'] = vector_labels
    vector_kwargs['logits'] = vector_logits
    vector_loss, _ = global_objective(**vector_kwargs)
    vector_loss_sum = tf.reduce_sum(vector_loss)

    # Construct matrix global objective kwargs and loss.
    matrix_kwargs = objective_kwargs.copy()
    matrix_kwargs['labels'] = tf.expand_dims(vector_labels, 1)
    matrix_kwargs['logits'] = tf.expand_dims(vector_logits, 1)
    matrix_loss, _ = global_objective(**matrix_kwargs)
    matrix_loss_sum = tf.reduce_sum(matrix_loss)

    self.assertEqual(1, vector_loss.get_shape().ndims)
    self.assertEqual(2, matrix_loss.get_shape().ndims)

    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(vector_loss_sum.eval(), matrix_loss_sum.eval())

  @parameterized.named_parameters(
      ('_prauc', loss_layers.precision_recall_auc_loss, None),
      ('_roc', loss_layers.roc_auc_loss, None),
      ('_rap', loss_layers.recall_at_precision_loss, {'target_precision': 0.8}),
      ('_patr', loss_layers.precision_at_recall_loss, {'target_recall': 0.7}),
      ('_fpattp', loss_layers.false_positive_rate_at_true_positive_rate_loss,
       {'target_rate': 0.9}),
      ('_tpatfp', loss_layers.true_positive_rate_at_false_positive_rate_loss,
       {'target_rate': 0.1})
  )
  def testUnknownBatchSize(self, global_objective, objective_kwargs):
    # Tests that there are no errors when the batch size is not known.
    batch_shape = [5, 2]
    logits = tf.placeholder(tf.float32)
    logits_feed = numpy.random.randn(*batch_shape)
    labels = tf.placeholder(tf.float32)
    labels_feed = logits_feed > 0.1
    logits.set_shape([None, 2])
    labels.set_shape([None, 2])

    if objective_kwargs is None:
      objective_kwargs = {}

    placeholder_kwargs = objective_kwargs.copy()
    placeholder_kwargs['labels'] = labels
    placeholder_kwargs['logits'] = logits
    placeholder_loss, _ = global_objective(**placeholder_kwargs)

    kwargs = objective_kwargs.copy()
    kwargs['labels'] = labels_feed
    kwargs['logits'] = logits_feed
    loss, _ = global_objective(**kwargs)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      feed_loss_val = session.run(placeholder_loss,
                                  feed_dict={logits: logits_feed,
                                             labels: labels_feed})
      loss_val = session.run(loss)
      self.assertAllClose(feed_loss_val, loss_val)


# Both sets of logits below are designed so that the surrogate precision and
# recall (true positive rate) of class 1 is ~ 2/3, and the same surrogates for
# class 2 are ~ 1/3. The false positive rate surrogates are ~ 1/3 and 2/3.
def _multilabel_data():
  targets = tf.constant([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], shape=[6, 1])
  targets = tf.concat([targets, targets], 1)
  logits_positives = tf.constant([[0.0, 15],
                                  [16, 0.0],
                                  [14, 0.0]], shape=[3, 2])
  logits_negatives = tf.constant([[-17, 0.0],
                                  [-15, 0.0],
                                  [0.0, -101]], shape=[3, 2])
  logits = tf.concat([logits_positives, logits_negatives], 0)
  priors = tf.constant(0.5, shape=[2])

  return targets, logits, priors


def _other_multilabel_data(surrogate_type):
  targets = tf.constant(
      [1.0] * 6 + [0.0] * 6, shape=[12, 1])
  targets = tf.concat([targets, targets], 1)
  logits_positives = tf.constant([[0.0, 13],
                                  [12, 0.0],
                                  [15, 0.0],
                                  [0.0, 30],
                                  [13, 0.0],
                                  [18, 0.0]], shape=[6, 2])
  # A score of cost_2 incurs a loss of ~2.0.
  cost_2 = 1.0 if surrogate_type == 'hinge' else 1.09861229
  logits_negatives = tf.constant([[-16, cost_2],
                                  [-15, cost_2],
                                  [cost_2, -111],
                                  [-133, -14,],
                                  [-14.0100101, -16,],
                                  [-19.888828882, -101]], shape=[6, 2])
  logits = tf.concat([logits_positives, logits_negatives], 0)
  priors = tf.constant(0.5, shape=[2])

  def builder():
    return targets, logits, priors

  return builder


if __name__ == '__main__':
  tf.test.main()
