# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Layers for Mixture of Experts (MoE) routing.

For MoE routing, we need to separate a set of tokens to sets of tokens.
Later on, different sets of tokens can potentially go to different experts.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Text")
class TokenImportanceWithMovingAvg(tf.keras.layers.Layer):
  """Routing based on per-token importance value."""

  def __init__(self,
               vocab_size,
               init_importance,
               moving_average_beta=0.995,
               **kwargs):
    self._vocab_size = vocab_size
    self._init_importance = init_importance
    self._moving_average_beta = moving_average_beta
    super(TokenImportanceWithMovingAvg, self).__init__(**kwargs)

  def build(self, input_shape):
    self._importance_embedding = self.add_weight(
        name="importance_embed",
        shape=(self._vocab_size),
        initializer=tf.keras.initializers.Constant(self._init_importance),
        trainable=False)

  def get_config(self):
    config = {
        "vocab_size":
            self._vocab_size,
        "init_importance":
            self._init_importance,
        "moving_average_beta":
            self._moving_average_beta,
    }
    base_config = super(TokenImportanceWithMovingAvg, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def update_token_importance(self, token_ids, importance):
    token_ids = tf.reshape(token_ids, shape=[-1])
    importance = tf.reshape(importance, shape=[-1])

    beta = self._moving_average_beta
    old_importance = tf.gather(self._importance_embedding, token_ids)
    self._importance_embedding.assign(tf.tensor_scatter_nd_update(
        self._importance_embedding,
        tf.expand_dims(token_ids, axis=1),
        old_importance * beta + tf.cast(importance * (1.0 - beta),
                                        dtype=tf.float32)))

  def call(self, inputs):
    return tf.gather(self._importance_embedding, inputs)


@tf.keras.utils.register_keras_serializable(package="Text")
class SelectTopK(tf.keras.layers.Layer):
  """Select top-k + random-k tokens according to importance."""

  def __init__(self,
               top_k=None,
               random_k=None,
               **kwargs):
    self._top_k = top_k
    self._random_k = random_k
    super(SelectTopK, self).__init__(**kwargs)

  def get_config(self):
    config = {
        "top_k":
            self._top_k,
        "random_k":
            self._random_k,
    }
    base_config = super(SelectTopK, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    if self._random_k is None:
      # Pure top-k, not randomness.
      pos = tf.argsort(inputs, direction="DESCENDING")
      selected = tf.slice(pos, [0, 0], [-1, self._top_k])
      not_selected = tf.slice(pos, [0, self._top_k], [-1, -1])
    elif self._top_k is None:
      # Pure randomness, no top-k.
      pos = tf.argsort(tf.random.uniform(shape=tf.shape(inputs)),
                       direction="DESCENDING")
      selected = tf.slice(pos, [0, 0], [-1, self._random_k])
      not_selected = tf.slice(pos, [0, self._random_k], [-1, -1])
    else:
      # Top-k plus randomness.
      pos = tf.argsort(inputs, direction="DESCENDING")
      selected_top_k = tf.slice(pos, [0, 0], [-1, self._top_k])
      pos_left = tf.slice(pos, [0, self._top_k], [-1, -1])

      # Randomly shuffle pos_left
      sort_index = tf.argsort(
          tf.random.uniform(shape=tf.shape(pos_left)),
          direction="DESCENDING")
      pos_left = tf.gather(pos_left, sort_index, batch_dims=1, axis=1)

      selected_rand = tf.slice(pos_left, [0, 0], [-1, self._random_k])
      not_selected = tf.slice(pos_left, [0, self._random_k], [-1, -1])

      selected = tf.concat([selected_top_k, selected_rand], axis=1)

    # Return the indices of selected and not-selected tokens.
    return selected, not_selected
