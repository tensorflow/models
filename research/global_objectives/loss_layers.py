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
"""Loss functions for learning global objectives.

These functions have two return values: a Tensor with the value of
the loss, and a dictionary of internal quantities for customizability.
"""

# Dependency imports
import numpy
import tensorflow as tf

from global_objectives import util


def precision_recall_auc_loss(
    labels,
    logits,
    precision_range=(0.0, 1.0),
    num_anchors=20,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):
  """Computes precision-recall AUC loss.

  The loss is based on a sum of losses for recall at a range of
  precision values (anchor points). This sum is a Riemann sum that
  approximates the area under the precision-recall curve.

  The per-example `weights` argument changes not only the coefficients of
  individual training examples, but how the examples are counted toward the
  constraint. If `label_priors` is given, it MUST take `weights` into account.
  That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).

  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    precision_range: A length-two tuple, the range of precision values over
      which to compute AUC. The entries must be nonnegative, increasing, and
      less than or equal to 1.0.
    num_anchors: The number of grid points used to approximate the Riemann sum.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
    lambdas_initializer: An initializer for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [1, num_labels, num_anchors] consisting of the
        Lagrange multipliers.
      biases: A Tensor of shape [1, num_labels, num_anchors] consisting of the
        learned bias term for each.
      label_priors: A Tensor of shape [1, num_labels, 1] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.

  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  with tf.variable_scope(scope,
                         'precision_recall_auc',
                         [labels, logits, label_priors],
                         reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)
    num_labels = util.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    dual_rate_factor = util.convert_and_cast(
        dual_rate_factor, 'dual_rate_factor', logits.dtype)

    # Create Tensor of anchor points and distance between anchors.
    precision_values, delta = _range_to_anchors_and_delta(
        precision_range, num_anchors, logits.dtype)
    # Create lambdas with shape [1, num_labels, num_anchors].
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[1, num_labels, num_anchors],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Create biases with shape [1, num_labels, num_anchors].
    biases = tf.contrib.framework.model_variable(
        name='biases',
        shape=[1, num_labels, num_anchors],
        dtype=logits.dtype,
        initializer=tf.zeros_initializer(),
        collections=variables_collections,
        trainable=trainable)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(
        label_priors, labels, weights, variables_collections)
    label_priors = tf.reshape(label_priors, [1, num_labels, 1])

    # Expand logits, labels, and weights to shape [batch_size, num_labels, 1].
    logits = tf.expand_dims(logits, 2)
    labels = tf.expand_dims(labels, 2)
    weights = tf.expand_dims(weights, 2)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    loss = weights * util.weighted_surrogate_loss(
        labels,
        logits + biases,
        surrogate_type=surrogate_type,
        positive_weights=1.0 + lambdas * (1.0 - precision_values),
        negative_weights=lambdas * precision_values)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * (1.0 - precision_values) * label_priors * maybe_log2
    per_anchor_loss = loss - lambda_term
    per_label_loss = delta * tf.reduce_sum(per_anchor_loss, 2)
    # Normalize the AUC such that a perfect score function will have AUC 1.0.
    # Because precision_range is discretized into num_anchors + 1 intervals
    # but only num_anchors terms are included in the Riemann sum, the
    # effective length of the integration interval is `delta` less than the
    # length of precision_range.
    scaled_loss = tf.div(per_label_loss,
                         precision_range[1] - precision_range[0] - delta,
                         name='AUC_Normalize')
    scaled_loss = tf.reshape(scaled_loss, original_shape)

    other_outputs = {
        'lambdas': lambdas_variable,
        'biases': biases,
        'label_priors': label_priors,
        'true_positives_lower_bound': true_positives_lower_bound(
            labels, logits, weights, surrogate_type),
        'false_positives_upper_bound': false_positives_upper_bound(
            labels, logits, weights, surrogate_type)}

    return scaled_loss, other_outputs


def roc_auc_loss(
    labels,
    logits,
    weights=1.0,
    surrogate_type='xent',
    scope=None):
  """Computes ROC AUC loss.

  The area under the ROC curve is the probability p that a randomly chosen
  positive example will be scored higher than a randomly chosen negative
  example. This loss approximates 1-p by using a surrogate (either hinge loss or
  cross entropy) for the indicator function. Specifically, the loss is:

    sum_i sum_j w_i*w_j*loss(logit_i - logit_j)

  where i ranges over the positive datapoints, j ranges over the negative
  datapoints, logit_k denotes the logit (or score) of the k-th datapoint, and
  loss is either the hinge or log loss given a positive label.

  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape and dtype as `labels`.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for the indicator function.
    scope: Optional scope for `name_scope`.

  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise loss.
    other_outputs: An empty dictionary, for consistency.

  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
    # Convert inputs to tensors and standardize dtypes.
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)

    # Create tensors of pairwise differences for logits and labels, and
    # pairwise products of weights. These have shape
    # [batch_size, batch_size, num_labels].
    logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
    labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
    weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

    signed_logits_difference = labels_difference * logits_difference
    raw_loss = util.weighted_surrogate_loss(
        labels=tf.ones_like(signed_logits_difference),
        logits=signed_logits_difference,
        surrogate_type=surrogate_type)
    weighted_loss = weights_product * raw_loss

    # Zero out entries of the loss where labels_difference zero (so loss is only
    # computed on pairs with different labels).
    loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
    loss = tf.reshape(loss, original_shape)
    return loss, {}


def recall_at_precision_loss(
    labels,
    logits,
    target_precision,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):
  """Computes recall at precision loss.

  The loss is based on a surrogate of the form
      wt * w(+) * loss(+) + wt * w(-) * loss(-) - c * pi,
  where:
  - w(+) =  1 + lambdas * (1 - target_precision)
  - loss(+) is the cross-entropy loss on the positive examples
  - w(-) = lambdas * target_precision
  - loss(-) is the cross-entropy loss on the negative examples
  - wt is a scalar or tensor of per-example weights
  - c = lambdas * (1 - target_precision)
  - pi is the label_priors.

  The per-example weights change not only the coefficients of individual
  training examples, but how the examples are counted toward the constraint.
  If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).

  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_precision: The precision at which to compute the loss. Can be a
      floating point value between 0 and 1 for a single precision value, or a
      `Tensor` of shape [num_labels], holding each label's target precision
      value.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
    lambdas_initializer: An initializer for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.

  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  with tf.variable_scope(scope,
                         'recall_at_precision',
                         [logits, labels, label_priors],
                         reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)
    num_labels = util.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_precision = util.convert_and_cast(
        target_precision, 'target_precision', logits.dtype)
    dual_rate_factor = util.convert_and_cast(
        dual_rate_factor, 'dual_rate_factor', logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(
        label_priors, labels, weights, variables_collections)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * util.weighted_surrogate_loss(
        labels,
        logits,
        surrogate_type=surrogate_type,
        positive_weights=1.0 + lambdas * (1.0 - target_precision),
        negative_weights=lambdas * target_precision)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * (1.0 - target_precision) * label_priors * maybe_log2
    loss = tf.reshape(weighted_loss - lambda_term, original_shape)
    other_outputs = {
        'lambdas': lambdas_variable,
        'label_priors': label_priors,
        'true_positives_lower_bound': true_positives_lower_bound(
            labels, logits, weights, surrogate_type),
        'false_positives_upper_bound': false_positives_upper_bound(
            labels, logits, weights, surrogate_type)}

    return loss, other_outputs


def precision_at_recall_loss(
    labels,
    logits,
    target_recall,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):
  """Computes precision at recall loss.

  The loss is based on a surrogate of the form
     wt * loss(-) + lambdas * (pi * (b - 1) + wt * loss(+))
  where:
  - loss(-) is the cross-entropy loss on the negative examples
  - loss(+) is the cross-entropy loss on the positive examples
  - wt is a scalar or tensor of per-example weights
  - b is the target recall
  - pi is the label_priors.

  The per-example weights change not only the coefficients of individual
  training examples, but how the examples are counted toward the constraint.
  If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).

  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_recall: The recall at which to compute the loss. Can be a floating
      point value between 0 and 1 for a single target recall value, or a
      `Tensor` of shape [num_labels] holding each label's target recall value.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
    lambdas_initializer: An initializer for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.
  """
  with tf.variable_scope(scope,
                         'precision_at_recall',
                         [logits, labels, label_priors],
                         reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)
    num_labels = util.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_recall = util.convert_and_cast(
        target_recall, 'target_recall', logits.dtype)
    dual_rate_factor = util.convert_and_cast(
        dual_rate_factor, 'dual_rate_factor', logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(
        label_priors, labels, weights, variables_collections)

    # Calculate weighted loss and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * util.weighted_surrogate_loss(
        labels,
        logits,
        surrogate_type,
        positive_weights=lambdas,
        negative_weights=1.0)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * label_priors * (target_recall - 1.0) * maybe_log2
    loss = tf.reshape(weighted_loss + lambda_term, original_shape)
    other_outputs = {
        'lambdas': lambdas_variable,
        'label_priors': label_priors,
        'true_positives_lower_bound': true_positives_lower_bound(
            labels, logits, weights, surrogate_type),
        'false_positives_upper_bound': false_positives_upper_bound(
            labels, logits, weights, surrogate_type)}

    return loss, other_outputs


def false_positive_rate_at_true_positive_rate_loss(
    labels,
    logits,
    target_rate,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):
  """Computes false positive rate at true positive rate loss.

  Note that `true positive rate` is a synonym for Recall, and that minimizing
  the false positive rate and maximizing precision are equivalent for a fixed
  Recall. Therefore, this function is identical to precision_at_recall_loss.

  The per-example weights change not only the coefficients of individual
  training examples, but how the examples are counted toward the constraint.
  If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).

  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_rate: The true positive rate at which to compute the loss. Can be a
      floating point value between 0 and 1 for a single true positive rate, or
      a `Tensor` of shape [num_labels] holding each label's true positive rate.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions. 'xent' will use the cross-entropy
      loss surrogate, and 'hinge' will use the hinge loss.
    lambdas_initializer: An initializer op for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.

  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  return precision_at_recall_loss(labels=labels,
                                  logits=logits,
                                  target_recall=target_rate,
                                  weights=weights,
                                  dual_rate_factor=dual_rate_factor,
                                  label_priors=label_priors,
                                  surrogate_type=surrogate_type,
                                  lambdas_initializer=lambdas_initializer,
                                  reuse=reuse,
                                  variables_collections=variables_collections,
                                  trainable=trainable,
                                  scope=scope)


def true_positive_rate_at_false_positive_rate_loss(
    labels,
    logits,
    target_rate,
    weights=1.0,
    dual_rate_factor=0.1,
    label_priors=None,
    surrogate_type='xent',
    lambdas_initializer=tf.constant_initializer(1.0),
    reuse=None,
    variables_collections=None,
    trainable=True,
    scope=None):
  """Computes true positive rate at false positive rate loss.

  The loss is based on a surrogate of the form
      wt * loss(+) + lambdas * (wt * loss(-) - r * (1 - pi))
  where:
  - loss(-) is the loss on the negative examples
  - loss(+) is the loss on the positive examples
  - wt is a scalar or tensor of per-example weights
  - r is the target rate
  - pi is the label_priors.

  The per-example weights change not only the coefficients of individual
  training examples, but how the examples are counted toward the constraint.
  If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
  where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).

  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_rate: The false positive rate at which to compute the loss. Can be a
      floating point value between 0 and 1 for a single false positive rate, or
      a `Tensor` of shape [num_labels] holding each label's false positive rate.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions. 'xent' will use the cross-entropy
      loss surrogate, and 'hinge' will use the hinge loss.
    lambdas_initializer: An initializer op for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

  Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.

  Raises:
    ValueError: If `surrogate_type` is not `xent` or `hinge`.
  """
  with tf.variable_scope(scope,
                         'tpr_at_fpr',
                         [labels, logits, label_priors],
                         reuse=reuse):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)
    num_labels = util.get_num_labels(logits)

    # Convert other inputs to tensors and standardize dtypes.
    target_rate = util.convert_and_cast(
        target_rate, 'target_rate', logits.dtype)
    dual_rate_factor = util.convert_and_cast(
        dual_rate_factor, 'dual_rate_factor', logits.dtype)

    # Create lambdas.
    lambdas, lambdas_variable = _create_dual_variable(
        'lambdas',
        shape=[num_labels],
        dtype=logits.dtype,
        initializer=lambdas_initializer,
        collections=variables_collections,
        trainable=trainable,
        dual_rate_factor=dual_rate_factor)
    # Maybe create label_priors.
    label_priors = maybe_create_label_priors(
        label_priors, labels, weights, variables_collections)

    # Loss op and other outputs. The log(2.0) term corrects for
    # logloss not being an upper bound on the indicator function.
    weighted_loss = weights * util.weighted_surrogate_loss(
        labels,
        logits,
        surrogate_type=surrogate_type,
        positive_weights=1.0,
        negative_weights=lambdas)
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    lambda_term = lambdas * target_rate * (1.0 - label_priors) * maybe_log2
    loss = tf.reshape(weighted_loss - lambda_term, original_shape)
    other_outputs = {
        'lambdas': lambdas_variable,
        'label_priors': label_priors,
        'true_positives_lower_bound': true_positives_lower_bound(
            labels, logits, weights, surrogate_type),
        'false_positives_upper_bound': false_positives_upper_bound(
            labels, logits, weights, surrogate_type)}

  return loss, other_outputs


def _prepare_labels_logits_weights(labels, logits, weights):
  """Validates labels, logits, and weights.

  Converts inputs to tensors, checks shape compatibility, and casts dtype if
  necessary.

  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    weights: Either `None` or a `Tensor` with shape broadcastable to `logits`.

  Returns:
    labels: Same as `labels` arg after possible conversion to tensor, cast, and
      reshape.
    logits: Same as `logits` arg after possible conversion to tensor and
      reshape.
    weights: Same as `weights` arg after possible conversion, cast, and reshape.
    original_shape: Shape of `labels` and `logits` before reshape.

  Raises:
    ValueError: If `labels` and `logits` do not have the same shape.
  """
  # Convert `labels` and `logits` to Tensors and standardize dtypes.
  logits = tf.convert_to_tensor(logits, name='logits')
  labels = util.convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
  weights = util.convert_and_cast(weights, 'weights', logits.dtype.base_dtype)

  try:
    labels.get_shape().merge_with(logits.get_shape())
  except ValueError:
    raise ValueError('logits and labels must have the same shape (%s vs %s)' %
                     (logits.get_shape(), labels.get_shape()))

  original_shape = labels.get_shape().as_list()
  if labels.get_shape().ndims > 0:
    original_shape[0] = -1
  if labels.get_shape().ndims <= 1:
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.reshape(logits, [-1, 1])

  if weights.get_shape().ndims == 1:
    # Weights has shape [batch_size]. Reshape to [batch_size, 1].
    weights = tf.reshape(weights, [-1, 1])
  if weights.get_shape().ndims == 0:
    # Weights is a scalar. Change shape of weights to match logits.
    weights *= tf.ones_like(logits)

  return labels, logits, weights, original_shape


def _range_to_anchors_and_delta(precision_range, num_anchors, dtype):
  """Calculates anchor points from precision range.

  Args:
    precision_range: As required in precision_recall_auc_loss.
    num_anchors: int, number of equally spaced anchor points.
    dtype: Data type of returned tensors.

  Returns:
    precision_values: A `Tensor` of data type dtype with equally spaced values
      in the interval precision_range.
    delta: The spacing between the values in precision_values.

  Raises:
    ValueError: If precision_range is invalid.
  """
  # Validate precision_range.
  if not 0 <= precision_range[0] <= precision_range[-1] <= 1:
    raise ValueError('precision values must obey 0 <= %f <= %f <= 1' %
                     (precision_range[0], precision_range[-1]))
  if not 0 < len(precision_range) < 3:
    raise ValueError('length of precision_range (%d) must be 1 or 2' %
                     len(precision_range))

  # Sets precision_values uniformly between min_precision and max_precision.
  values = numpy.linspace(start=precision_range[0],
                          stop=precision_range[1],
                          num=num_anchors+2)[1:-1]
  precision_values = util.convert_and_cast(
      values, 'precision_values', dtype)
  delta = util.convert_and_cast(
      values[0] - precision_range[0], 'delta', dtype)
  # Makes precision_values [1, 1, num_anchors].
  precision_values = util.expand_outer(precision_values, 3)
  return precision_values, delta


def _create_dual_variable(name, shape, dtype, initializer, collections,
                          trainable, dual_rate_factor):
  """Creates a new dual variable.

  Dual variables are required to be nonnegative. If trainable, their gradient
  is reversed so that they are maximized (rather than minimized) by the
  optimizer.

  Args:
    name: A string, the name for the new variable.
    shape: Shape of the new variable.
    dtype: Data type for the new variable.
    initializer: Initializer for the new variable.
    collections: List of graph collections keys. The new variable is added to
      these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
    trainable: If `True`, the default, also adds the variable to the graph
      collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
      the default list of variables to use by the `Optimizer` classes.
    dual_rate_factor: A floating point value or `Tensor`. The learning rate for
      the dual variable is scaled by this factor.

  Returns:
    dual_value: An op that computes the absolute value of the dual variable
      and reverses its gradient.
    dual_variable: The underlying variable itself.
  """
  # We disable partitioning while constructing dual variables because they will
  # be updated with assign, which is not available for partitioned variables.
  partitioner = tf.get_variable_scope().partitioner
  try:
    tf.get_variable_scope().set_partitioner(None)
    dual_variable = tf.contrib.framework.model_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        collections=collections,
        trainable=trainable)
  finally:
    tf.get_variable_scope().set_partitioner(partitioner)
  # Using the absolute value enforces nonnegativity.
  dual_value = tf.abs(dual_variable)

  if trainable:
    # To reverse the gradient on the dual variable, multiply the gradient by
    # -dual_rate_factor
    dual_value = (tf.stop_gradient((1.0 + dual_rate_factor) * dual_value)
                  - dual_rate_factor * dual_value)
  return dual_value, dual_variable


def maybe_create_label_priors(label_priors,
                              labels,
                              weights,
                              variables_collections):
  """Creates moving average ops to track label priors, if necessary.

  Args:
    label_priors: As required in e.g. precision_recall_auc_loss.
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    weights: As required in e.g. precision_recall_auc_loss.
    variables_collections: Optional list of collections for the variables, if
      any must be created.

  Returns:
    label_priors: A Tensor of shape [num_labels] consisting of the
      weighted label priors, after updating with moving average ops if created.
  """
  if label_priors is not None:
    label_priors = util.convert_and_cast(
        label_priors, name='label_priors', dtype=labels.dtype.base_dtype)
    return tf.squeeze(label_priors)

  label_priors = util.build_label_priors(
      labels,
      weights,
      variables_collections=variables_collections)
  return label_priors


def true_positives_lower_bound(labels, logits, weights, surrogate_type):
  """Calculate a lower bound on the number of true positives.

  This lower bound on the number of true positives given `logits` and `labels`
  is the same one used in the global objectives loss functions.

  Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` of shape [batch_size, num_labels] or
      [batch_size, num_labels, num_anchors]. If the third dimension is present,
      the lower bound is computed on each slice [:, :, k] independently.
    weights: Per-example loss coefficients, with shape broadcast-compatible with
        that of `labels`.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.

  Returns:
    A `Tensor` of shape [num_labels] or [num_labels, num_anchors].
  """
  maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
  maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
  if logits.get_shape().ndims == 3 and labels.get_shape().ndims < 3:
    labels = tf.expand_dims(labels, 2)
  loss_on_positives = util.weighted_surrogate_loss(
      labels, logits, surrogate_type, negative_weights=0.0) / maybe_log2
  return tf.reduce_sum(weights * (labels - loss_on_positives), 0)


def false_positives_upper_bound(labels, logits, weights, surrogate_type):
  """Calculate an upper bound on the number of false positives.

  This upper bound on the number of false positives given `logits` and `labels`
  is the same one used in the global objectives loss functions.

  Args:
    labels: A `Tensor` of shape [batch_size, num_labels]
    logits: A `Tensor` of shape [batch_size, num_labels]  or
      [batch_size, num_labels, num_anchors]. If the third dimension is present,
      the lower bound is computed on each slice [:, :, k] independently.
    weights: Per-example loss coefficients, with shape broadcast-compatible with
        that of `labels`.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.

  Returns:
    A `Tensor` of shape [num_labels] or [num_labels, num_anchors].
  """
  maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
  maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
  loss_on_negatives = util.weighted_surrogate_loss(
      labels, logits, surrogate_type, positive_weights=0.0) / maybe_log2
  return tf.reduce_sum(weights *  loss_on_negatives, 0)
