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


def neumf_model_fn(features, labels, mode, params):
  """Model Function for NeuMF estimator."""
  users = features[movielens.USER_COLUMN]
  items = tf.cast(features[movielens.ITEM_COLUMN], tf.int32)

  logits = construct_model(users=users, items=items, params=params)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        movielens.ITEM_COLUMN: items,
        movielens.RATING_COLUMN: logits,
    }

    if params["use_tpu"]:
      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  elif mode == tf.estimator.ModeKeys.TRAIN:
    labels = tf.cast(labels, tf.int32)
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    if params["use_tpu"]:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Softmax with the first column of ones is equivalent to sigmoid.
    logits = tf.concat([tf.ones(logits.shape, dtype=logits.dtype), logits]
                       , axis=1)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    global_step = tf.train.get_global_step()
    tvars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
        loss, tvars, colocate_gradients_with_ops=True)
    minimize_op = optimizer.apply_gradients(
        gradients, global_step=global_step, name="train")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    if params["use_tpu"]:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  else:
    raise NotImplementedError


def construct_model(users, items, params):
  # type: (tf.Tensor, tf.Tensor, dict) -> tf.Tensor
  """Initialize NeuMF model.

  Args:
    users: Tensor of user ids.
    items: Tensor of item ids.
    params: Dict of hyperparameters.

  Raises:
    ValueError: if the first model layer is not even.
  """

  num_users = params["num_users"]
  num_items = params["num_items"]

  model_layers = params["model_layers"]

  mf_regularization = params["mf_regularization"]
  mlp_reg_layers = params["mlp_reg_layers"]

  mf_dim = params["mf_dim"]

  if model_layers[0] % 2 != 0:
    raise ValueError("The first layer size should be multiple of 2!")

  # Input variables
  user_input = tf.keras.layers.Input(tensor=users)
  item_input = tf.keras.layers.Input(tensor=items)

  # Initializer for embedding layers
  embedding_initializer = "glorot_uniform"

  # Embedding layers of GMF and MLP
  mf_embedding_user = tf.keras.layers.Embedding(
      num_users,
      mf_dim,
      embeddings_initializer=embedding_initializer,
      embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
      input_length=1)
  mf_embedding_item = tf.keras.layers.Embedding(
      num_items,
      mf_dim,
      embeddings_initializer=embedding_initializer,
      embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
      input_length=1)

  mlp_embedding_user = tf.keras.layers.Embedding(
      num_users,
      model_layers[0]//2,
      embeddings_initializer=embedding_initializer,
      embeddings_regularizer=tf.keras.regularizers.l2(mlp_reg_layers[0]),
      input_length=1)
  mlp_embedding_item = tf.keras.layers.Embedding(
      num_items,
      model_layers[0]//2,
      embeddings_initializer=embedding_initializer,
      embeddings_regularizer=tf.keras.regularizers.l2(mlp_reg_layers[0]),
      input_length=1)

  # GMF part
  mf_user_latent = mf_embedding_user(user_input)
  mf_item_latent = mf_embedding_item(item_input)
  # Element-wise multiply
  mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

  # MLP part
  mlp_user_latent = mlp_embedding_user(user_input)
  mlp_item_latent = mlp_embedding_item(item_input)
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
  tf.keras.models.Model([user_input, item_input], logits).summary()
  sys.stdout.flush()

  return logits
