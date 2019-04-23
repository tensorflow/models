# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Defines NeuMF model for NCF framework.

Some abbreviations used in the code base:
NeuMF: Neural Matrix Factorization
NCF: Neural Collaborative Filtering
GMF: Generalized Matrix Factorization
MLP: Multi-Layer Perceptron

GMF applies a linear kernel to model the latent feature interactions, and MLP
uses a nonlinear kernel to learn the interaction function from data. NeuMF model
is a fused model of GMF and MLP to better model the complex user-item
interactions, and unifies the strengths of linearity of MF and non-linearity of
MLP for modeling the user-item latent structures.

In NeuMF model, it allows GMF and MLP to learn separate embeddings, and combine
the two models by concatenating their last hidden layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from official.datasets import movielens  # pylint: disable=g-bad-import-order
from official.recommendation import constants as rconst
from official.recommendation import ncf_common
from official.recommendation import stat_utils
from official.utils.logs import mlperf_helper


def _sparse_to_dense_grads(grads_and_vars):
  """Convert sparse gradients to dense gradients.

  All sparse gradients, which are represented as instances of tf.IndexedSlices,
  are converted to dense Tensors. Dense gradients, which are represents as
  Tensors, are unchanged.

  The purpose of this conversion is that for small embeddings, which are used by
  this model, applying dense gradients with the AdamOptimizer is faster than
  applying sparse gradients.

  Args
    grads_and_vars: A list of (gradient, variable) tuples. Each gradient can
      be a Tensor or an IndexedSlices. Tensors are unchanged, and IndexedSlices
      are converted to dense Tensors.
  Returns:
    The same list of (gradient, variable) as `grads_and_vars`, except each
    IndexedSlices gradient is converted to a Tensor.
  """

  # Calling convert_to_tensor changes IndexedSlices into Tensors, and leaves
  # Tensors unchanged.
  return [(tf.convert_to_tensor(g), v) for g, v in grads_and_vars]


def neumf_model_fn(features, labels, mode, params):
  """Model Function for NeuMF estimator."""
  if params.get("use_seed"):
    tf.set_random_seed(stat_utils.random_int32())

  users = features[movielens.USER_COLUMN]
  items = features[movielens.ITEM_COLUMN]

  user_input = tf.keras.layers.Input(tensor=users)
  item_input = tf.keras.layers.Input(tensor=items)
  logits = construct_model(user_input, item_input, params).output

  # Softmax with the first column of zeros is equivalent to sigmoid.
  softmax_logits = ncf_common.convert_to_softmax_logits(logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    duplicate_mask = tf.cast(features[rconst.DUPLICATE_MASK], tf.float32)
    return _get_estimator_spec_with_metrics(
        logits,
        softmax_logits,
        duplicate_mask,
        params["num_neg"],
        params["match_mlperf"],
        use_tpu_spec=params["use_xla_for_gpu"])

  elif mode == tf.estimator.ModeKeys.TRAIN:
    labels = tf.cast(labels, tf.int32)
    valid_pt_mask = features[rconst.VALID_POINT_MASK]

    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_NAME, value="adam")
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_LR,
                            value=params["learning_rate"])
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_HP_ADAM_BETA1,
                            value=params["beta1"])
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_HP_ADAM_BETA2,
                            value=params["beta2"])
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.OPT_HP_ADAM_EPSILON,
                            value=params["epsilon"])


    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=params["learning_rate"],
        beta1=params["beta1"],
        beta2=params["beta2"],
        epsilon=params["epsilon"])
    if params["use_tpu"]:
      # TODO(seemuch): remove this contrib import
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.MODEL_HP_LOSS_FN,
                            value=mlperf_helper.TAGS.BCE)

    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=softmax_logits,
        weights=tf.cast(valid_pt_mask, tf.float32)
    )

    # This tensor is used by logging hooks.
    tf.identity(loss, name="cross_entropy")

    global_step = tf.compat.v1.train.get_global_step()
    tvars = tf.compat.v1.trainable_variables()
    gradients = optimizer.compute_gradients(
        loss, tvars, colocate_gradients_with_ops=True)
    gradients = _sparse_to_dense_grads(gradients)
    minimize_op = optimizer.apply_gradients(
        gradients, global_step=global_step, name="train")
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  else:
    raise NotImplementedError


def _strip_first_and_last_dimension(x, batch_size):
  return tf.reshape(x[0, :], (batch_size,))


def construct_model(user_input, item_input, params, need_strip=False):
  # type: (tf.Tensor, tf.Tensor, dict) -> tf.keras.Model
  """Initialize NeuMF model.

  Args:
    user_input: keras input layer for users
    item_input: keras input layer for items
    params: Dict of hyperparameters.
  Raises:
    ValueError: if the first model layer is not even.
  Returns:
    model:  a keras Model for computing the logits
  """
  num_users = params["num_users"]
  num_items = params["num_items"]

  model_layers = params["model_layers"]

  mf_regularization = params["mf_regularization"]
  mlp_reg_layers = params["mlp_reg_layers"]

  mf_dim = params["mf_dim"]

  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.MODEL_HP_MF_DIM, value=mf_dim)
  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.MODEL_HP_MLP_LAYER_SIZES,
                          value=model_layers)

  if model_layers[0] % 2 != 0:
    raise ValueError("The first layer size should be multiple of 2!")

  # Initializer for embedding layers
  embedding_initializer = "glorot_uniform"

  if need_strip:
    batch_size = params["batch_size"]

    user_input_reshaped = tf.keras.layers.Lambda(
        lambda x: _strip_first_and_last_dimension(
            x, batch_size))(user_input)

    item_input_reshaped = tf.keras.layers.Lambda(
        lambda x: _strip_first_and_last_dimension(
            x, batch_size))(item_input)

  # It turns out to be significantly more effecient to store the MF and MLP
  # embedding portions in the same table, and then slice as needed.
  mf_slice_fn = lambda x: x[:, :mf_dim]
  mlp_slice_fn = lambda x: x[:, mf_dim:]
  embedding_user = tf.keras.layers.Embedding(
      num_users, mf_dim + model_layers[0] // 2,
      embeddings_initializer=embedding_initializer,
      embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
      input_length=1, name="embedding_user")(
          user_input_reshaped if need_strip else user_input)

  embedding_item = tf.keras.layers.Embedding(
      num_items, mf_dim + model_layers[0] // 2,
      embeddings_initializer=embedding_initializer,
      embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
      input_length=1, name="embedding_item")(
          item_input_reshaped if need_strip else item_input)

  # GMF part
  mf_user_latent = tf.keras.layers.Lambda(
      mf_slice_fn, name="embedding_user_mf")(embedding_user)
  mf_item_latent = tf.keras.layers.Lambda(
      mf_slice_fn, name="embedding_item_mf")(embedding_item)

  # MLP part
  mlp_user_latent = tf.keras.layers.Lambda(
      mlp_slice_fn, name="embedding_user_mlp")(embedding_user)
  mlp_item_latent = tf.keras.layers.Lambda(
      mlp_slice_fn, name="embedding_item_mlp")(embedding_item)

  # Element-wise multiply
  mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

  # Concatenation of two latent features
  mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

  num_layer = len(model_layers)  # Number of layers in the MLP
  for layer in xrange(1, num_layer):
    model_layer = tf.keras.layers.Dense(
        model_layers[layer],
        kernel_regularizer=tf.keras.regularizers.l2(mlp_reg_layers[layer]),
        activation="relu")
    mlp_vector = model_layer(mlp_vector)

  # Concatenate GMF and MLP parts
  predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_vector])

  # Final prediction layer
  logits = tf.keras.layers.Dense(
      1, activation=None, kernel_initializer="lecun_uniform",
      name=movielens.RATING_COLUMN)(predict_vector)

  # Print model topology.
  model = tf.keras.models.Model([user_input, item_input], logits)
  model.summary()
  sys.stdout.flush()

  return model


def _get_estimator_spec_with_metrics(logits,              # type: tf.Tensor
                                     softmax_logits,      # type: tf.Tensor
                                     duplicate_mask,      # type: tf.Tensor
                                     num_training_neg,    # type: int
                                     match_mlperf=False,  # type: bool
                                     use_tpu_spec=False   # type: bool
                                    ):
  """Returns a EstimatorSpec that includes the metrics."""
  cross_entropy, \
  metric_fn, \
  in_top_k, \
  ndcg, \
  metric_weights = compute_eval_loss_and_metrics_helper(
      logits,
      softmax_logits,
      duplicate_mask,
      num_training_neg,
      match_mlperf,
      use_tpu_spec)

  if use_tpu_spec:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=cross_entropy,
        eval_metrics=(metric_fn, [in_top_k, ndcg, metric_weights]))

  return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.EVAL,
      loss=cross_entropy,
      eval_metric_ops=metric_fn(in_top_k, ndcg, metric_weights)
  )


def compute_eval_loss_and_metrics_helper(logits,              # type: tf.Tensor
                                         softmax_logits,      # type: tf.Tensor
                                         duplicate_mask,      # type: tf.Tensor
                                         num_training_neg,    # type: int
                                         match_mlperf=False,  # type: bool
                                         use_tpu_spec=False   # type: bool
                                        ):
  """Model evaluation with HR and NDCG metrics.

  The evaluation protocol is to rank the test interacted item (truth items)
  among the randomly chosen 999 items that are not interacted by the user.
  The performance of the ranked list is judged by Hit Ratio (HR) and Normalized
  Discounted Cumulative Gain (NDCG).

  For evaluation, the ranked list is truncated at 10 for both metrics. As such,
  the HR intuitively measures whether the test item is present on the top-10
  list, and the NDCG accounts for the position of the hit by assigning higher
  scores to hits at top ranks. Both metrics are calculated for each test user,
  and the average scores are reported.

  If `match_mlperf` is True, then the HR and NDCG computations are done in a
  slightly unusual way to match the MLPerf reference implementation.
  Specifically, if the evaluation negatives contain duplicate items, it will be
  treated as if the item only appeared once. Effectively, for duplicate items in
  a row, the predicted score for all but one of the items will be set to
  -infinity

  For example, suppose we have that following inputs:
  logits_by_user:     [[ 2,  3,  3],
                       [ 5,  4,  4]]

  items_by_user:     [[10, 20, 20],
                      [30, 40, 40]]

  # Note: items_by_user is not explicitly present. Instead the relevant \
          information is contained within `duplicate_mask`

  top_k: 2

  Then with match_mlperf=True, the HR would be 2/2 = 1.0. With
  match_mlperf=False, the HR would be 1/2 = 0.5. This is because each user has
  predicted scores for only 2 unique items: 10 and 20 for the first user, and 30
  and 40 for the second. Therefore, with match_mlperf=True, it's guaranteed the
  first item's score is in the top 2. With match_mlperf=False, this function
  would compute the first user's first item is not in the top 2, because item 20
  has a higher score, and item 20 occurs twice.

  Args:
    logits: A tensor containing the predicted logits for each user. The shape
      of logits is (num_users_per_batch * (1 + NUM_EVAL_NEGATIVES),) Logits
      for a user are grouped, and the last element of the group is the true
      element.

    softmax_logits: The same tensor, but with zeros left-appended.

    duplicate_mask: A vector with the same shape as logits, with a value of 1
      if the item corresponding to the logit at that position has already
      appeared for that user.

    num_training_neg: The number of negatives per positive during training.

    match_mlperf: Use the MLPerf reference convention for computing rank.

    use_tpu_spec: Should a TPUEstimatorSpec be returned instead of an
      EstimatorSpec. Required for TPUs and if XLA is done on a GPU. Despite its
      name, TPUEstimatorSpecs work with GPUs

  Returns:
    cross_entropy: the loss
    metric_fn: the metrics function
    in_top_k: hit rate metric
    ndcg: ndcg metric
    metric_weights: metric weights
  """
  in_top_k, ndcg, metric_weights, logits_by_user = compute_top_k_and_ndcg(
      logits, duplicate_mask, match_mlperf)

  # Examples are provided by the eval Dataset in a structured format, so eval
  # labels can be reconstructed on the fly.
  eval_labels = tf.reshape(shape=(-1,), tensor=tf.one_hot(
      tf.zeros(shape=(logits_by_user.shape[0],), dtype=tf.int32) +
      rconst.NUM_EVAL_NEGATIVES, logits_by_user.shape[1], dtype=tf.int32))

  eval_labels_float = tf.cast(eval_labels, tf.float32)

  # During evaluation, the ratio of negatives to positives is much higher
  # than during training. (Typically 999 to 1 vs. 4 to 1) By adjusting the
  # weights for the negative examples we compute a loss which is consistent with
  # the training data. (And provides apples-to-apples comparison)
  negative_scale_factor = num_training_neg / rconst.NUM_EVAL_NEGATIVES
  example_weights = (
      (eval_labels_float + (1 - eval_labels_float) * negative_scale_factor) *
      (1 + rconst.NUM_EVAL_NEGATIVES) / (1 + num_training_neg))

  # Tile metric weights back to logit dimensions
  expanded_metric_weights = tf.reshape(tf.tile(
      metric_weights[:, tf.newaxis], (1, rconst.NUM_EVAL_NEGATIVES + 1)), (-1,))

  # ignore padded examples
  example_weights *= tf.cast(expanded_metric_weights, tf.float32)

  cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
      logits=softmax_logits, labels=eval_labels, weights=example_weights)

  def metric_fn(top_k_tensor, ndcg_tensor, weight_tensor):
    return {
        rconst.HR_KEY: tf.compat.v1.metrics.mean(top_k_tensor,
                                                 weights=weight_tensor,
                                                 name=rconst.HR_METRIC_NAME),
        rconst.NDCG_KEY: tf.compat.v1.metrics.mean(ndcg_tensor,
                                                   weights=weight_tensor,
                                                   name=rconst.NDCG_METRIC_NAME)
    }

  return cross_entropy, metric_fn, in_top_k, ndcg, metric_weights


def compute_top_k_and_ndcg(logits,              # type: tf.Tensor
                           duplicate_mask,      # type: tf.Tensor
                           match_mlperf=False   # type: bool
                          ):
  """Compute inputs of metric calculation.

  Args:
    logits: A tensor containing the predicted logits for each user. The shape
      of logits is (num_users_per_batch * (1 + NUM_EVAL_NEGATIVES),) Logits
      for a user are grouped, and the first element of the group is the true
      element.
    duplicate_mask: A vector with the same shape as logits, with a value of 1
      if the item corresponding to the logit at that position has already
      appeared for that user.
    match_mlperf: Use the MLPerf reference convention for computing rank.

  Returns:
    is_top_k, ndcg and weights, all of which has size (num_users_in_batch,), and
    logits_by_user which has size
    (num_users_in_batch, (rconst.NUM_EVAL_NEGATIVES + 1)).
  """
  logits_by_user = tf.reshape(logits, (-1, rconst.NUM_EVAL_NEGATIVES + 1))
  duplicate_mask_by_user = tf.reshape(duplicate_mask,
                                      (-1, rconst.NUM_EVAL_NEGATIVES + 1))

  if match_mlperf:
    # Set duplicate logits to the min value for that dtype. The MLPerf
    # reference dedupes during evaluation.
    logits_by_user *= (1 - duplicate_mask_by_user)
    logits_by_user += duplicate_mask_by_user * logits_by_user.dtype.min

  # Determine the location of the first element in each row after the elements
  # are sorted.
  sort_indices = tf.argsort(
      logits_by_user, axis=1, direction="DESCENDING")

  # Use matrix multiplication to extract the position of the true item from the
  # tensor of sorted indices. This approach is chosen because both GPUs and TPUs
  # perform matrix multiplications very quickly. This is similar to np.argwhere.
  # However this is a special case because the target will only appear in
  # sort_indices once.
  one_hot_position = tf.cast(tf.equal(sort_indices, rconst.NUM_EVAL_NEGATIVES),
                             tf.int32)
  sparse_positions = tf.multiply(
      one_hot_position, tf.range(logits_by_user.shape[1])[tf.newaxis, :])
  position_vector = tf.reduce_sum(sparse_positions, axis=1)

  in_top_k = tf.cast(tf.less(position_vector, rconst.TOP_K), tf.float32)
  ndcg = tf.math.log(2.) / tf.math.log(
      tf.cast(position_vector, tf.float32) + 2)
  ndcg *= in_top_k

  # If a row is a padded row, all but the first element will be a duplicate.
  metric_weights = tf.not_equal(tf.reduce_sum(duplicate_mask_by_user, axis=1),
                                rconst.NUM_EVAL_NEGATIVES)

  return in_top_k, ndcg, metric_weights, logits_by_user
