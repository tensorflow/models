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
import typing

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
  softmax_logits = ncf_common.softmax_logitfy(logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    duplicate_mask = tf.cast(features[rconst.DUPLICATE_MASK], tf.float32)
    return compute_eval_loss_and_metrics(
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

    optimizer = get_optimizer(params)

    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.MODEL_HP_LOSS_FN,
                            value=mlperf_helper.TAGS.BCE)

    print(">>>>>>>>>>>>>>labels: ", labels)
    print(">>>>>>>>>>>>>>softmax_logits: ", softmax_logits)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=softmax_logits,
        weights=tf.cast(valid_pt_mask, tf.float32)
    )

    # This tensor is used by logging hooks.
    tf.identity(loss, name="cross_entropy")

    global_step = tf.train.get_global_step()
    tvars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
        loss, tvars, colocate_gradients_with_ops=True)
    gradients = _sparse_to_dense_grads(gradients)
    minimize_op = optimizer.apply_gradients(
        gradients, global_step=global_step, name="train")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  else:
    raise NotImplementedError


def get_optimizer(params):
  optimizer = tf.train.AdamOptimizer(
      learning_rate=params["learning_rate"], beta1=params["beta1"],
      beta2=params["beta2"], epsilon=params["epsilon"])
  if params["use_tpu"]:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  return optimizer


def construct_model(user_input, item_input, params):
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

  # It turns out to be significantly more effecient to store the MF and MLP
  # embedding portions in the same table, and then slice as needed.
  mf_slice_fn = lambda x: x[:, :mf_dim]
  mlp_slice_fn = lambda x: x[:, mf_dim:]
  embedding_user = tf.keras.layers.Embedding(
      num_users, mf_dim + model_layers[0] // 2,
      embeddings_initializer=embedding_initializer,
      embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
      input_length=1, name="embedding_user")(user_input)

  embedding_item = tf.keras.layers.Embedding(
      num_items, mf_dim + model_layers[0] // 2,
      embeddings_initializer=embedding_initializer,
      embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
      input_length=1, name="embedding_item")(item_input)

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


def compute_eval_loss_and_metrics(logits,              # type: tf.Tensor
                                  softmax_logits,      # type: tf.Tensor
                                  duplicate_mask,      # type: tf.Tensor
                                  num_training_neg,    # type: int
                                  match_mlperf=False,  # type: bool
                                  use_tpu_spec=False   # type: bool
                                 ):

  cross_entropy, \
  metric_fn, \
  in_top_k, \
  ndcg, \
  metric_weights = ncf_common.compute_eval_loss_and_metrics_helper(
      logits, softmax_logits, duplicate_mask, num_training_neg, match_mlperf, use_tpu_spec)

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

