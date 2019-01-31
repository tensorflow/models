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
"""Contains utility functions for the global objectives library."""

# Dependency imports
import tensorflow as tf


def weighted_sigmoid_cross_entropy_with_logits(labels,
                                               logits,
                                               positive_weights=1.0,
                                               negative_weights=1.0,
                                               name=None):
  """Computes a weighting of sigmoid cross entropy given `logits`.

  Measures the weighted probability error in discrete classification tasks in
  which classes are independent and not mutually exclusive.  For instance, one
  could perform multilabel classification where a picture can contain both an
  elephant and a dog at the same time. The class weight multiplies the
  different types of errors.
  For brevity, let `x = logits`, `z = labels`, `c = positive_weights`,
  `d = negative_weights`  The
  weighed logistic loss is

  ```
  c * z * -log(sigmoid(x)) + d * (1 - z) * -log(1 - sigmoid(x))
  = c * z * -log(1 / (1 + exp(-x))) - d * (1 - z) * log(exp(-x) / (1 + exp(-x)))
  = c * z * log(1 + exp(-x)) + d * (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
  = c * z * log(1 + exp(-x)) + d * (1 - z) * (x + log(1 + exp(-x)))
  = (1 - z) * x * d + (1 - z + c * z ) * log(1 + exp(-x))
  =  - d * x * z + d * x + (d - d * z + c * z ) * log(1 + exp(-x))
  ```

  To ensure stability and avoid overflow, the implementation uses the identity
      log(1 + exp(-x)) = max(0,-x) + log(1 + exp(-abs(x)))
  and the result is computed as

    ```
    = -d * x * z + d * x
      + (d - d * z + c * z ) * (max(0,-x) + log(1 + exp(-abs(x))))
    ```

  Note that the loss is NOT an upper bound on the 0-1 loss, unless it is divided
  by log(2).

  Args:
    labels: A `Tensor` of type `float32` or `float64`. `labels` can be a 2D
      tensor with shape [batch_size, num_labels] or a 3D tensor with shape
      [batch_size, num_labels, K].
    logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
      shape [batch_size, num_labels, K], the loss is computed separately on each
      slice [:, :, k] of `logits`.
    positive_weights: A `Tensor` that holds positive weights and has the
      following semantics according to its shape:
        scalar - A global positive weight.
        1D tensor - must be of size K, a weight for each 'attempt'
        2D tensor - of size [num_labels, K'] where K' is either K or 1.
      The `positive_weights` will be expanded to the left to match the
      dimensions of logits and labels.
    negative_weights: A `Tensor` that holds positive weight and has the
      semantics identical to positive_weights.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
      weighted logistic losses.
  """
  with tf.name_scope(
      name,
      'weighted_logistic_loss',
      [logits, labels, positive_weights, negative_weights]) as name:
    labels, logits, positive_weights, negative_weights = prepare_loss_args(
        labels, logits, positive_weights, negative_weights)

    softplus_term = tf.add(tf.maximum(-logits, 0.0),
                           tf.log(1.0 + tf.exp(-tf.abs(logits))))
    weight_dependent_factor = (
        negative_weights + (positive_weights - negative_weights) * labels)
    return (negative_weights * (logits - labels * logits) +
            weight_dependent_factor * softplus_term)


def weighted_hinge_loss(labels,
                        logits,
                        positive_weights=1.0,
                        negative_weights=1.0,
                        name=None):
  """Computes weighted hinge loss given logits `logits`.

  The loss applies to multi-label classification tasks where labels are
  independent and not mutually exclusive. See also
  `weighted_sigmoid_cross_entropy_with_logits`.

  Args:
    labels: A `Tensor` of type `float32` or `float64`. Each entry must be
      either 0 or 1. `labels` can be a 2D tensor with shape
      [batch_size, num_labels] or a 3D tensor with shape
      [batch_size, num_labels, K].
    logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
      shape [batch_size, num_labels, K], the loss is computed separately on each
      slice [:, :, k] of `logits`.
    positive_weights: A `Tensor` that holds positive weights and has the
      following semantics according to its shape:
        scalar - A global positive weight.
        1D tensor - must be of size K, a weight for each 'attempt'
        2D tensor - of size [num_labels, K'] where K' is either K or 1.
      The `positive_weights` will be expanded to the left to match the
      dimensions of logits and labels.
    negative_weights: A `Tensor` that holds positive weight and has the
      semantics identical to positive_weights.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
      weighted hinge loss.
  """
  with tf.name_scope(
      name, 'weighted_hinge_loss',
      [logits, labels, positive_weights, negative_weights]) as name:
    labels, logits, positive_weights, negative_weights = prepare_loss_args(
        labels, logits, positive_weights, negative_weights)

    positives_term = positive_weights * labels * tf.maximum(1.0 - logits, 0)
    negatives_term = (negative_weights * (1.0 - labels)
                      * tf.maximum(1.0 + logits, 0))
    return positives_term + negatives_term


def weighted_surrogate_loss(labels,
                            logits,
                            surrogate_type='xent',
                            positive_weights=1.0,
                            negative_weights=1.0,
                            name=None):
  """Returns either weighted cross-entropy or hinge loss.

  For example `surrogate_type` is 'xent' returns the weighted cross
  entropy loss.

  Args:
   labels: A `Tensor` of type `float32` or `float64`. Each entry must be
      between 0 and 1. `labels` can be a 2D tensor with shape
      [batch_size, num_labels] or a 3D tensor with shape
      [batch_size, num_labels, K].
    logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
      shape [batch_size, num_labels, K], each slice [:, :, k] represents an
      'attempt' to predict `labels` and the loss is computed per slice.
    surrogate_type: A string that determines which loss to return, supports
    'xent' for cross-entropy and 'hinge' for hinge loss.
    positive_weights: A `Tensor` that holds positive weights and has the
      following semantics according to its shape:
        scalar - A global positive weight.
        1D tensor - must be of size K, a weight for each 'attempt'
        2D tensor - of size [num_labels, K'] where K' is either K or 1.
      The `positive_weights` will be expanded to the left to match the
      dimensions of logits and labels.
    negative_weights: A `Tensor` that holds positive weight and has the
      semantics identical to positive_weights.
    name: A name for the operation (optional).

  Returns:
    The weigthed loss.

  Raises:
    ValueError: If value of `surrogate_type` is not supported.
  """
  with tf.name_scope(
      name, 'weighted_loss',
      [logits, labels, surrogate_type, positive_weights,
       negative_weights]) as name:
    if surrogate_type == 'xent':
      return weighted_sigmoid_cross_entropy_with_logits(
          logits=logits,
          labels=labels,
          positive_weights=positive_weights,
          negative_weights=negative_weights,
          name=name)
    elif surrogate_type == 'hinge':
      return weighted_hinge_loss(
          logits=logits,
          labels=labels,
          positive_weights=positive_weights,
          negative_weights=negative_weights,
          name=name)
    raise ValueError('surrogate_type %s not supported.' % surrogate_type)


def expand_outer(tensor, rank):
  """Expands the given `Tensor` outwards to a target rank.

  For example if rank = 3 and tensor.shape is [3, 4], this function will expand
  to such that the resulting shape will be  [1, 3, 4].

  Args:
    tensor: The tensor to expand.
    rank: The target dimension.

  Returns:
    The expanded tensor.

  Raises:
    ValueError: If rank of `tensor` is unknown, or if `rank` is smaller than
      the rank of `tensor`.
  """
  if tensor.get_shape().ndims is None:
    raise ValueError('tensor dimension must be known.')
  if len(tensor.get_shape()) > rank:
    raise ValueError(
        '`rank` must be at least the current tensor dimension: (%s vs %s).' %
        (rank, len(tensor.get_shape())))
  while len(tensor.get_shape()) < rank:
    tensor = tf.expand_dims(tensor, 0)
  return tensor


def build_label_priors(labels,
                       weights=None,
                       positive_pseudocount=1.0,
                       negative_pseudocount=1.0,
                       variables_collections=None):
  """Creates an op to maintain and update label prior probabilities.

  For each label, the label priors are estimated as
      (P + sum_i w_i y_i) / (P + N + sum_i w_i),
  where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
  positive labels, and N is a pseudo-count of negative labels. The index i
  ranges over all labels observed during all evaluations of the returned op.

  Args:
    labels: A `Tensor` with shape [batch_size, num_labels]. Entries should be
      in [0, 1].
    weights: Coefficients representing the weight of each label. Must be either
      a Tensor of shape [batch_size, num_labels] or `None`, in which case each
      weight is treated as 1.0.
    positive_pseudocount: Number of positive labels used to initialize the label
      priors.
    negative_pseudocount: Number of negative labels used to initialize the label
      priors.
    variables_collections: Optional list of collections for created variables.

  Returns:
    label_priors: An op to update the weighted label_priors. Gives the
      current value of the label priors when evaluated.
  """
  dtype = labels.dtype.base_dtype
  num_labels = get_num_labels(labels)

  if weights is None:
    weights = tf.ones_like(labels)

  # We disable partitioning while constructing dual variables because they will
  # be updated with assign, which is not available for partitioned variables.
  partitioner = tf.get_variable_scope().partitioner
  try:
    tf.get_variable_scope().set_partitioner(None)
    # Create variable and update op for weighted label counts.
    weighted_label_counts = tf.contrib.framework.model_variable(
        name='weighted_label_counts',
        shape=[num_labels],
        dtype=dtype,
        initializer=tf.constant_initializer(
            [positive_pseudocount] * num_labels, dtype=dtype),
        collections=variables_collections,
        trainable=False)
    weighted_label_counts_update = weighted_label_counts.assign_add(
        tf.reduce_sum(weights * labels, 0))

    # Create variable and update op for the sum of the weights.
    weight_sum = tf.contrib.framework.model_variable(
        name='weight_sum',
        shape=[num_labels],
        dtype=dtype,
        initializer=tf.constant_initializer(
            [positive_pseudocount + negative_pseudocount] * num_labels,
            dtype=dtype),
        collections=variables_collections,
        trainable=False)
    weight_sum_update = weight_sum.assign_add(tf.reduce_sum(weights, 0))

  finally:
    tf.get_variable_scope().set_partitioner(partitioner)

  label_priors = tf.div(
      weighted_label_counts_update,
      weight_sum_update)
  return label_priors


def convert_and_cast(value, name, dtype):
  """Convert input to tensor and cast to dtype.

  Args:
    value: An object whose type has a registered Tensor conversion function,
        e.g. python numerical type or numpy array.
    name: Name to use for the new Tensor, if one is created.
    dtype: Optional element type for the returned tensor.

  Returns:
    A tensor.
  """
  return tf.cast(tf.convert_to_tensor(value, name=name), dtype=dtype)


def prepare_loss_args(labels, logits, positive_weights, negative_weights):
  """Prepare arguments for weighted loss functions.

  If needed, will convert given arguments to appropriate type and shape.

  Args:
    labels: labels or labels of the loss function.
    logits: Logits of the loss function.
    positive_weights: Weight on the positive examples.
    negative_weights: Weight on the negative examples.

  Returns:
    Converted labels, logits, positive_weights, negative_weights.
  """
  logits = tf.convert_to_tensor(logits, name='logits')
  labels = convert_and_cast(labels, 'labels', logits.dtype)
  if len(labels.get_shape()) == 2 and len(logits.get_shape()) == 3:
    labels = tf.expand_dims(labels, [2])

  positive_weights = convert_and_cast(positive_weights, 'positive_weights',
                                      logits.dtype)
  positive_weights = expand_outer(positive_weights, logits.get_shape().ndims)
  negative_weights = convert_and_cast(negative_weights, 'negative_weights',
                                      logits.dtype)
  negative_weights = expand_outer(negative_weights, logits.get_shape().ndims)
  return labels, logits, positive_weights, negative_weights


def get_num_labels(labels_or_logits):
  """Returns the number of labels inferred from labels_or_logits."""
  if labels_or_logits.get_shape().ndims <= 1:
    return 1
  return labels_or_logits.get_shape()[1].value
