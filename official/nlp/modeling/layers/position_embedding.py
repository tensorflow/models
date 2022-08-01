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

"""Keras-based positional embedding layer."""
# pylint: disable=g-classes-have-attributes
import math
from typing import Optional

import tensorflow as tf

from official.modeling import tf_utils

Initializer = tf.keras.initializers.Initializer


@tf.keras.utils.register_keras_serializable(package="Text")
class PositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  Example:
  ```python
  position_embedding = PositionEmbedding(max_length=100)
  inputs = tf.keras.Input((100, 32), dtype=tf.float32)
  outputs = position_embedding(inputs)
  ```


  Args:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    seq_axis: The axis of the input tensor where we add the embeddings.

  Reference: This layer creates a positional embedding as described in
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).
  """

  def __init__(self,
               max_length,
               initializer="glorot_uniform",
               seq_axis=1,
               **kwargs):

    super().__init__(**kwargs)
    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self._max_length = max_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._seq_axis = seq_axis

  def get_config(self):
    config = {
        "max_length": self._max_length,
        "initializer": tf.keras.initializers.serialize(self._initializer),
        "seq_axis": self._seq_axis,
    }
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    dimension_list = input_shape.as_list()
    width = dimension_list[-1]
    weight_sequence_length = self._max_length

    self._position_embeddings = self.add_weight(
        "embeddings",
        shape=[weight_sequence_length, width],
        initializer=self._initializer)

    super().build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    actual_seq_len = input_shape[self._seq_axis]
    position_embeddings = self._position_embeddings[:actual_seq_len, :]
    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)


@tf.keras.utils.register_keras_serializable(package="Text")
class RelativePositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  This layer calculates the position encoding as a mix of sine and cosine
  functions with geometrically increasing wavelengths. Defined and formulized in
   "Attention is All You Need", section 3.5.
  (https://arxiv.org/abs/1706.03762).

  Args:
    hidden_size: Size of the hidden layer.
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position.
  """

  def __init__(self,
               hidden_size: int,
               min_timescale: float = 1.0,
               max_timescale: float = 1.0e4,
               **kwargs):
    # We need to have a default dtype of float32, since the inputs (which Keras
    # usually uses to infer the dtype) will always be int32.
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically
    # unstable in float16.
    if "dtype" not in kwargs:
      kwargs["dtype"] = "float32"

    super().__init__(**kwargs)
    self._hidden_size = hidden_size
    self._min_timescale = min_timescale
    self._max_timescale = max_timescale

  def get_config(self):
    config = {
        "hidden_size": self._hidden_size,
        "min_timescale": self._min_timescale,
        "max_timescale": self._max_timescale,
    }
    base_config = super(RelativePositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, length=None):
    """Implements call() for the layer.

    Args:
      inputs: An tensor whose second dimension will be used as `length`. If
        `None`, the other `length` argument must be specified.
      length: An optional integer specifying the number of positions. If both
        `inputs` and `length` are spcified, `length` must be equal to the second
        dimension of `inputs`.

    Returns:
      A tensor in shape of `(length, hidden_size)`.
    """
    if inputs is None and length is None:
      raise ValueError("If inputs is None, `length` must be set in "
                       "RelativePositionEmbedding().")
    if inputs is not None:
      input_shape = tf_utils.get_shape_list(inputs)
      if length is not None and length != input_shape[1]:
        raise ValueError(
            "If inputs is not None, `length` must equal to input_shape[1].")
      length = input_shape[1]
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = self._hidden_size // 2
    min_timescale, max_timescale = self._min_timescale, self._max_timescale
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) *
        -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    position_embeddings = tf.concat(
        [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return position_embeddings


def _relative_position_bucket(relative_position,
                              bidirectional=True,
                              num_buckets=32,
                              max_distance=128):
  """Translate relative position to a bucket number for relative attention.

  The relative position is defined as memory_position - query_position, i.e.
  the distance in tokens from the attending position to the attended-to
  position.

  If `bidirectional=False`, then positive relative positions are invalid.

  We use smaller buckets for small absolute relative_position and larger
  buckets for larger absolute relative_positions.

  All relative positions >=max_distance map to the same bucket.

  All relative positions <=-max_distance map to the same bucket.

  This should allow for more graceful generalization to longer sequences
  than the model has been trained on.

  Args:
    relative_position: An int32 Tensor
    bidirectional: A boolean - whether the attention is bidirectional
    num_buckets: An integer
    max_distance: An integer

  Returns:
    A Tensor with the same shape as relative_position, containing int32
    values in the range [0, num_buckets)
  """
  ret = 0
  n = -relative_position
  if bidirectional:
    num_buckets //= 2
    ret += tf.cast(tf.math.less(n, 0), tf.int32) * num_buckets
    n = tf.math.abs(n)
  else:
    n = tf.math.maximum(n, 0)
  # now n is in the range [0, inf)
  max_exact = num_buckets // 2
  is_small = tf.math.less(n, max_exact)
  val_if_large = max_exact + tf.dtypes.cast(
      tf.math.log(tf.cast(n, tf.float32) / max_exact) /
      math.log(max_distance / max_exact) * (num_buckets - max_exact),
      tf.int32,
  )
  val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
  ret += tf.where(is_small, n, val_if_large)
  return ret


@tf.keras.utils.register_keras_serializable(package="Text")
class RelativePositionBias(tf.keras.layers.Layer):
  """Relative position embedding via per-head bias in T5 style.

  Reference implementation in MeshTF:
  https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L1000

  This layer implements the relative position bias used in "Exploring the Limits
  of Transfer Learning with a Unified Text-to-Text Transformer"
  (https://arxiv.org/abs/1910.10683)
  """

  def __init__(self,
               num_heads: int,
               relative_attention_num_buckets: int = 32,
               relative_attention_max_distance: int = 128,
               bidirectional: bool = True,
               embeddings_initializer: Optional[Initializer] = None,
               **kwargs):
    super().__init__(**kwargs)
    self.num_heads = num_heads
    self.relative_attention_num_buckets = relative_attention_num_buckets
    self.bidirectional = bidirectional
    self.relative_attention_max_distance = relative_attention_max_distance
    if embeddings_initializer:
      self._embed_init = embeddings_initializer
    else:
      self._embed_init = tf.keras.initializers.TruncatedNormal(stddev=1.0)
    with tf.name_scope(self.name):
      self._relative_attention_bias = self.add_weight(
          "rel_embedding",
          shape=[self.relative_attention_num_buckets, self.num_heads],
          initializer=self._embed_init,
          dtype=self.dtype,
          trainable=True)

  def get_config(self):
    config = {
        "num_heads":
            self.num_heads,
        "relative_attention_num_buckets":
            self.relative_attention_num_buckets,
        "relative_attention_max_distance":
            self.relative_attention_max_distance,
        "bidirectional":
            self.bidirectional,
        "embeddings_initializer":
            tf.keras.initializers.serialize(self._embed_init),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, query: tf.Tensor, key: tf.Tensor):
    """Implements the forward pass.

    Args:
      query: query input tensor shape [batch, query length, hidden size].
      key: key input tensor shape [batch, key length, hidden size].

    Returns:
      A tensor in shape of [batch, heads, query length, key length].
    """
    batch_size, qlen = tf_utils.get_shape_list(query)[:2]
    klen = tf_utils.get_shape_list(key)[1]
    context_position = tf.range(qlen)[:, None]
    memory_position = tf.range(klen)[None, :]
    relative_position = memory_position - context_position
    rp_bucket = _relative_position_bucket(
        relative_position,
        bidirectional=self.bidirectional,
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance)
    values = tf.nn.embedding_lookup(self._relative_attention_bias, rp_bucket)
    values = tf.expand_dims(
        tf.transpose(values, [2, 0, 1]),
        axis=0)  # shape (1, num_heads, qlen, klen)
    values = tf.tile(values, [batch_size, 1, 1, 1])
    return values
