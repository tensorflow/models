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

import tensorflow as tf  # pylint: disable=g-bad-import-order

from six.moves import xrange  # pylint: disable=redefined-builtin


def _validate_batch_size_for_multi_gpu(batch_size):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Args:
    batch_size: The number of examples processed in each training batch.

  Returns:
    num_gpus: The number of available GPUs.

  Raises:
    ValueError: If no GPUs are found, or selected batch_size is invalid.
  """
  local_device_protos = tf.python.client.device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. '
           'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)

  return num_gpus


class NeuMF(object):
  """Neural matrix factorization (NeuMF) model for recommendations."""

  def __init__(self, num_users, num_items, mf_dim, model_layers, learning_rate):
    """Initialize NeuMF model.

    Args:
      num_users: An integer, the number of users.
      num_items: An integer, the number of items.
      mf_dim: An integer, the embedding size of Matrix Factorization (MF) model.
      model_layers: A list of integers for Multi-Layer Perceptron (MLP) layers.
        Note that the first layer is the concatenation of user and item
        embeddings. So layers[0]//2 is the embedding size for MLP.
      learning_rate: The learning rate for optimizer

    Raises:
      ValueError: if the first model layer is not even.
    """
    if model_layers[0] % 2 != 0:
      raise ValueError('The first layer size should be multiple of 2!')

    self.num_users = num_users
    self.num_items = num_items
    self.mf_dim = mf_dim
    self.model_layers = model_layers
    self.learning_rate = learning_rate

  def __call__(self, multi_gpu, batch_size):
    """Create a NeuMF recommendation model.

    Args:
      multi_gpu: A boolean flag for multi_gpu option
      batch_size: The number of examples processed in each training batch.
    Returns:
      neumf_model: An instance of tf.keras Model as the NeuMF model.
    """
    # Input variables
    user_input = tf.keras.layers.Input(
        shape=(1,), dtype='int32', name='user_input')
    item_input = tf.keras.layers.Input(
        shape=(1,), dtype='int32', name='item_input')

    # Initializer for embedding layer
    embedding_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    # Embedding layer of GMF and MLP
    mf_embedding_user = tf.keras.layers.Embedding(
        self.num_users,
        self.mf_dim,
        embeddings_initializer=embedding_initializer,
        input_length=1)
    mf_embedding_item = tf.keras.layers.Embedding(
        self.num_items,
        self.mf_dim,
        embeddings_initializer=embedding_initializer,
        input_length=1)

    mlp_embedding_user = tf.keras.layers.Embedding(
        self.num_users,
        self.model_layers[0]//2,
        embeddings_initializer=embedding_initializer,
        input_length=1)
    mlp_embedding_item = tf.keras.layers.Embedding(
        self.num_items,
        self.model_layers[0]//2,
        embeddings_initializer=embedding_initializer,
        input_length=1)

    # GMF part
    mf_user_latent = tf.keras.layers.Flatten()(mf_embedding_user(user_input))
    mf_item_latent = tf.keras.layers.Flatten()(mf_embedding_item(item_input))
    # Element-wise multiply
    mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_user_latent = tf.keras.layers.Flatten()(mlp_embedding_user(user_input))
    mlp_item_latent = tf.keras.layers.Flatten()(mlp_embedding_item(item_input))
    # Concatenation of two latent features
    mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

    num_layer = len(self.model_layers)  # Number of layers in the MLP
    for idx in xrange(1, num_layer):
      model_layer = tf.keras.layers.Dense(
          self.model_layers[idx],
          activation='relu'
      )
      mlp_vector = model_layer(mlp_vector)

    # Concatenate GMF and MLP parts
    predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = tf.keras.layers.Dense(
        1, activation='sigmoid', kernel_initializer='lecun_uniform',
        name='prediction')(predict_vector)

    # Instantiate a tf.keras model
    model = tf.keras.models.Model(
        inputs=[user_input, item_input], outputs=prediction)

    # Use multiple gpus
    if multi_gpu:
      num_gpus = _validate_batch_size_for_multi_gpu(batch_size)
      try:
        model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpus)
        print('Training using {} GPUs..'.format(num_gpus))
      except ValueError:
        print('multi_gpu_model fails. Training using single GPU or CPU..')

    # Configure model
    model.compile(optimizer=tf.keras.optimizers.Adam(
        lr=self.learning_rate), loss='binary_crossentropy')

    return model
