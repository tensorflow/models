# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Pack sequence optimization on accelerators."""
from typing import Dict
import tensorflow as tf, tf_keras
from official.modeling import tf_utils
from official.nlp.modeling.layers import rezero_transformer
from official.nlp.modeling.layers import self_attention_mask
from official.nlp.modeling.layers import transformer_encoder_block
from official.nlp.modeling.layers import transformer_scaffold


@tf_keras.utils.register_keras_serializable(package='Text')
class PackBertEmbeddings(tf_keras.layers.Layer):
  """Performs packing tricks for BERT inputs to improve TPU utilization."""

  def __init__(self, pack_sequences: int, **kwargs):
    super().__init__(**kwargs)
    self.pack_sequences = pack_sequences

  def call(self, input_embeddings: tf.Tensor,
           input_mask: tf.Tensor) -> Dict[str, tf.Tensor]:
    batch_size, seq_len, embedding_dim = tf_utils.get_shape_list(
        input_embeddings, expected_rank=3)
    reduced_batch_size = batch_size // self.pack_sequences
    packed_seq_len = self.pack_sequences * seq_len
    packed_embeddings = tf.reshape(
        input_embeddings, [reduced_batch_size, packed_seq_len, embedding_dim])
    input_mask = tf.reshape(input_mask, [reduced_batch_size, packed_seq_len])
    example_ids = 1 + tf.range(self.pack_sequences)
    # Shape: [batch_size, seq_len, pack_sequences].
    example_ids = tf.tile(example_ids[None, :, None],
                          [reduced_batch_size, 1, seq_len])
    example_ids = tf.reshape(example_ids, [reduced_batch_size, packed_seq_len])
    example_ids = tf.where(
        tf.math.equal(input_mask, 0), tf.zeros_like(example_ids), example_ids)
    packing_mask = tf.cast(
        tf.equal(
            tf.expand_dims(example_ids, 2), tf.expand_dims(example_ids, 1)),
        dtype=tf.bool)

    attention_mask = self_attention_mask.get_mask(
        packed_embeddings, input_mask, dtype=tf.bool)

    combined_attention_mask = tf.cast(
        tf.math.logical_and(attention_mask, packing_mask), tf.float32)

    return dict(
        packed_embeddings=packed_embeddings,
        combined_attention_mask=combined_attention_mask)


@tf_keras.utils.register_keras_serializable(package='Text')
class StridedTransformerEncoderBlock(
    transformer_encoder_block.TransformerEncoderBlock):
  """Transformer layer for packing optimization to stride over inputs."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self._output_range is not None:
      raise ValueError('StridedTransformerEncoderBlock does not '
                       'support `output_range` argument.')
    # TODO(b/337888023): Support block sparse attention with strided inputs.
    if self._src_block_size is not None:
      raise ValueError('StridedTransformerEncoderBlock does not '
                       'support block sparse attention.')

  def call(self, inputs, stride: tf.Tensor):
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        input_tensor, attention_mask = inputs
        key_value = None
      elif len(inputs) == 3:
        input_tensor, key_value, attention_mask = inputs
      else:
        raise ValueError('Unexpected inputs to %s with length at %d' %
                         (self.__class__, len(inputs)))
    else:
      input_tensor, key_value, attention_mask = (inputs, None, None)

    if self._norm_first:
      source_tensor = input_tensor[:, ::stride, :]
      input_tensor = self._attention_layer_norm(input_tensor)
      if key_value is not None:
        key_value = self._attention_layer_norm_kv(key_value)
    target_tensor = input_tensor[:, ::stride, :]
    if attention_mask is not None:
      attention_mask = attention_mask[:, ::stride, :]

    if key_value is None:
      key_value = input_tensor
    attention_output = self._attention_layer(
        query=target_tensor, value=key_value, attention_mask=attention_mask)
    attention_output = self._attention_dropout(attention_output)

    if self._norm_first:
      # Important to not combine `self._norm_first` and
      # `self._use_query_residual` into one if clause because else is only for
      # `_norm_first == False`.
      if self._use_query_residual:
        attention_output = source_tensor + attention_output
    else:
      if self._use_query_residual:
        attention_output = target_tensor + attention_output
      attention_output = self._attention_layer_norm(attention_output)

    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)
    inner_output = self._intermediate_dense(attention_output)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self._inner_dropout_layer(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self._output_dropout(layer_output)

    if self._norm_first:
      return source_attention_output + layer_output

    layer_output = tf.cast(layer_output, tf.float32)
    return self._output_layer_norm(layer_output + attention_output)


@tf_keras.utils.register_keras_serializable(package='Text')
class StridedReZeroTransformer(rezero_transformer.ReZeroTransformer):
  """ReZeroTransformer for packing optimization to stride over inputs."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self._output_range is not None:
      raise ValueError(f'{self.__class__} does not '
                       'support `output_range` argument.')
    # TODO(b/337888023): Support block sparse attention with strided inputs.
    if self._src_block_size is not None:
      raise ValueError(f'{self.__class__} does not '
                       'support block sparse attention.')

  def call(self, inputs, stride: tf.Tensor):
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        input_tensor, attention_mask = inputs
        key_value = None
      elif len(inputs) == 3:
        input_tensor, key_value, attention_mask = inputs
      else:
        raise ValueError(f'Unexpected inputs to {self.__class__} with '
                         f'length at {len(inputs)}.')
    else:
      input_tensor, key_value, attention_mask = (inputs, None, None)

    target_tensor = input_tensor[:, ::stride, :]
    if attention_mask is not None:
      attention_mask = attention_mask[:, ::stride, :]

    if key_value is None:
      key_value = input_tensor

    attention_output = self._attention_layer(
        query=target_tensor, value=key_value, attention_mask=attention_mask)
    attention_output = self._attention_dropout(attention_output)
    attention_output = target_tensor + self._rezero_a * attention_output
    if self._use_layer_norm:
      attention_output = self._attention_layer_norm(attention_output)
    else:
      attention_output = tf.cast(attention_output, tf.float32)

    intermediate_output = self._intermediate_dense(attention_output)
    intermediate_output = self._inner_activation_layer(intermediate_output)
    layer_output = self._output_dense(intermediate_output)
    layer_output = self._output_dropout(layer_output)
    layer_output = attention_output + tf.cast(self._rezero_a_ffn * layer_output,
                                              tf.float32)
    if self._use_layer_norm:
      layer_output = self._output_layer_norm(layer_output)

    return layer_output


@tf_keras.utils.register_keras_serializable(package='Text')
class StridedTransformerScaffold(transformer_scaffold.TransformerScaffold):
  """TransformerScaffold for packing optimization to stride over inputs."""

  def call(self, inputs, stride: tf.Tensor, training=None):
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        input_tensor, attention_mask = inputs
        key_value = None
      elif len(inputs) == 3:
        input_tensor, key_value, attention_mask = inputs
      else:
        raise ValueError('Unexpected inputs to %s with length at %d' %
                         (self.__class__, len(inputs)))
    else:
      input_tensor, key_value, attention_mask = (inputs, None, None)

    if key_value is None:
      key_value = input_tensor

    if self._norm_first:
      source_tensor = input_tensor[:, ::stride, :]
      input_tensor = self._attention_layer_norm(input_tensor, training=training)
    if attention_mask is not None:
      attention_mask = attention_mask[:, ::stride, :]
    target_tensor = input_tensor[:, ::stride, :]

    attention_output = self._attention_layer(
        query=target_tensor,
        value=key_value,
        attention_mask=attention_mask,
        training=training)
    attention_output = self._attention_dropout(
        attention_output, training=training)

    if self._norm_first:
      attention_output = source_tensor + attention_output
    else:
      attention_output = self._attention_layer_norm(
          target_tensor + attention_output, training=training)
    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(
          attention_output, training=training)

    if self._feedforward_block is None:
      intermediate_output = self._intermediate_dense(attention_output)
      intermediate_output = self._intermediate_activation_layer(
          intermediate_output)
      layer_output = self._output_dense(intermediate_output, training=training)
      layer_output = self._output_dropout(layer_output, training=training)
      layer_output = tf.cast(layer_output, tf.float32)
      if self._norm_first:
        layer_output = source_attention_output + layer_output
      else:
        layer_output = self._output_layer_norm(
            layer_output + attention_output, training=training)
    else:
      if self._norm_first:
        # if norm_first, assume the feedforward block will not apply layer norm
        layer_output = self._feedforward_block(
            attention_output, training=training)
        layer_output += source_attention_output
      else:
        # if not norm_first, assume that the feedforwad does apply layer norm
        layer_output = self._feedforward_block(
            attention_output, training=training)

    return layer_output
