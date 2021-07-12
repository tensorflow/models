# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Library functions for ContextRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tf_slim as slim


# The negative value used in padding the invalid weights.
_NEGATIVE_PADDING_VALUE = -100000


def filter_weight_value(weights, values, valid_mask):
  """Filters weights and values based on valid_mask.

  _NEGATIVE_PADDING_VALUE will be added to invalid elements in the weights to
  avoid their contribution in softmax. 0 will be set for the invalid elements in
  the values.

  Args:
    weights: A float Tensor of shape [batch_size, input_size, context_size].
    values: A float Tensor of shape [batch_size, context_size,
      projected_dimension].
    valid_mask: A boolean Tensor of shape [batch_size, context_size]. True means
      valid and False means invalid.

  Returns:
    weights: A float Tensor of shape [batch_size, input_size, context_size].
    values: A float Tensor of shape [batch_size, context_size,
      projected_dimension].

  Raises:
    ValueError: If shape of doesn't match.
  """
  w_batch_size, _, w_context_size = weights.shape
  v_batch_size, v_context_size, _ = values.shape
  m_batch_size, m_context_size = valid_mask.shape
  if w_batch_size != v_batch_size or v_batch_size != m_batch_size:
    raise ValueError("Please make sure the first dimension of the input"
                     " tensors are the same.")

  if w_context_size != v_context_size:
    raise ValueError("Please make sure the third dimension of weights matches"
                     " the second dimension of values.")

  if w_context_size != m_context_size:
    raise ValueError("Please make sure the third dimension of the weights"
                     " matches the second dimension of the valid_mask.")

  valid_mask = valid_mask[..., tf.newaxis]

  # Force the invalid weights to be very negative so it won't contribute to
  # the softmax.

  very_negative_mask = tf.ones(
      weights.shape, dtype=weights.dtype) * _NEGATIVE_PADDING_VALUE
  valid_weight_mask = tf.tile(tf.transpose(valid_mask, perm=[0, 2, 1]),
                              [1, weights.shape[1], 1])
  weights = tf.where(valid_weight_mask,
                     x=weights, y=very_negative_mask)

  # Force the invalid values to be 0.
  values *= tf.cast(valid_mask, values.dtype)

  return weights, values


def compute_valid_mask(num_valid_elements, num_elements):
  """Computes mask of valid entries within padded context feature.

  Args:
    num_valid_elements: A int32 Tensor of shape [batch_size].
    num_elements: An int32 Tensor.

  Returns:
    A boolean Tensor of the shape [batch_size, num_elements]. True means
      valid and False means invalid.
  """
  batch_size = num_valid_elements.shape[0]
  element_idxs = tf.range(num_elements, dtype=tf.int32)
  batch_element_idxs = tf.tile(element_idxs[tf.newaxis, ...], [batch_size, 1])
  num_valid_elements = num_valid_elements[..., tf.newaxis]
  valid_mask = tf.less(batch_element_idxs, num_valid_elements)
  return valid_mask


def project_features(features, projection_dimension, is_training, normalize):
  """Projects features to another feature space.

  Args:
    features: A float Tensor of shape [batch_size, features_size,
      num_features].
    projection_dimension: A int32 Tensor.
    is_training: A boolean Tensor (affecting batch normalization).
    normalize: A boolean Tensor. If true, the output features will be l2
      normalized on the last dimension.

  Returns:
    A float Tensor of shape [batch, features_size, projection_dimension].
  """
  # TODO(guanhangwu) Figure out a better way of specifying the batch norm
  # params.
  batch_norm_params = {
      "is_training": is_training,
      "decay": 0.97,
      "epsilon": 0.001,
      "center": True,
      "scale": True
  }

  batch_size, _, num_features = features.shape
  features = tf.reshape(features, [-1, num_features])
  projected_features = slim.fully_connected(
      features,
      num_outputs=projection_dimension,
      activation_fn=tf.nn.relu6,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params)

  projected_features = tf.reshape(projected_features,
                                  [batch_size, -1, projection_dimension])

  if normalize:
    projected_features = tf.math.l2_normalize(projected_features, axis=-1)

  return projected_features


def attention_block(input_features, context_features, bottleneck_dimension,
                    output_dimension, attention_temperature,
                    keys_values_valid_mask, queries_valid_mask,
                    is_training, block_name="AttentionBlock"):
  """Generic attention block.

  Args:
    input_features: A float Tensor of shape [batch_size, input_size,
      num_input_features].
    context_features: A float Tensor of shape [batch_size, context_size,
      num_context_features].
    bottleneck_dimension: A int32 Tensor representing the bottleneck dimension
      for intermediate projections.
    output_dimension: A int32 Tensor representing the last dimension of the
      output feature.
    attention_temperature: A float Tensor. It controls the temperature of the
      softmax for weights calculation. The formula for calculation as follows:
        weights = exp(weights / temperature) / sum(exp(weights / temperature))
    keys_values_valid_mask: A boolean Tensor of shape
      [batch_size, context_size].
    queries_valid_mask: A boolean Tensor of shape
      [batch_size, max_num_proposals].
    is_training: A boolean Tensor (affecting batch normalization).
    block_name: A string to specify names for different attention blocks

  Returns:
    A float Tensor of shape [batch_size, input_size, output_dimension].
  """

  with tf.variable_scope(block_name):
    queries = project_features(
        input_features, bottleneck_dimension, is_training, normalize=True)
    keys = project_features(
        context_features, bottleneck_dimension, is_training, normalize=True)
    values = project_features(
        context_features, bottleneck_dimension, is_training, normalize=True)

    # masking out any keys which are padding
    keys *= tf.cast(keys_values_valid_mask[..., tf.newaxis], keys.dtype)
    queries *= tf.cast(queries_valid_mask[..., tf.newaxis], queries.dtype)

    weights = tf.matmul(queries, keys, transpose_b=True)

    weights, values = filter_weight_value(weights, values,
                                          keys_values_valid_mask)

    weights = tf.identity(tf.nn.softmax(weights / attention_temperature),
                          name=block_name+"AttentionWeights")

    features = tf.matmul(weights, values)

  output_features = project_features(
      features, output_dimension, is_training, normalize=False)
  return output_features


def _compute_box_context_attention(box_features, num_proposals,
                                   context_features, valid_context_size,
                                   bottleneck_dimension,
                                   attention_temperature, is_training,
                                   max_num_proposals,
                                   use_self_attention=False,
                                   use_long_term_attention=True,
                                   self_attention_in_sequence=False,
                                   num_attention_heads=1,
                                   num_attention_layers=1):
  """Computes the attention feature from the context given a batch of box.

  Args:
    box_features: A float Tensor of shape [batch_size * max_num_proposals,
      height, width, channels]. It is pooled features from first stage
      proposals.
    num_proposals: The number of valid box proposals.
    context_features: A float Tensor of shape [batch_size, context_size,
      num_context_features].
    valid_context_size: A int32 Tensor of shape [batch_size].
    bottleneck_dimension: A int32 Tensor representing the bottleneck dimension
      for intermediate projections.
    attention_temperature: A float Tensor. It controls the temperature of the
      softmax for weights calculation. The formula for calculation as follows:
        weights = exp(weights / temperature) / sum(exp(weights / temperature))
    is_training: A boolean Tensor (affecting batch normalization).
    max_num_proposals: The number of box proposals for each image.
    use_self_attention: Whether to use an attention block across the
      first stage predicted box features for the input image.
    use_long_term_attention: Whether to use an attention block into the context
      features.
    self_attention_in_sequence: Whether self-attention and long term attention
      should be in sequence or parallel.
    num_attention_heads: Number of heads for multi-headed attention.
    num_attention_layers: Number of heads for multi-layered attention.

  Returns:
    A float Tensor of shape [batch_size, max_num_proposals, 1, 1, channels].
  """
  _, context_size, _ = context_features.shape
  context_valid_mask = compute_valid_mask(valid_context_size, context_size)

  total_proposals, height, width, channels = box_features.shape

  batch_size = total_proposals // max_num_proposals
  box_features = tf.reshape(
      box_features,
      [batch_size,
       max_num_proposals,
       height,
       width,
       channels])

  # Average pools over height and width dimension so that the shape of
  # box_features becomes [batch_size, max_num_proposals, channels].
  box_features = tf.reduce_mean(box_features, [2, 3])
  box_valid_mask = compute_valid_mask(
      num_proposals,
      box_features.shape[1])

  if use_self_attention:
    self_attention_box_features = attention_block(
        box_features, box_features, bottleneck_dimension, channels.value,
        attention_temperature, keys_values_valid_mask=box_valid_mask,
        queries_valid_mask=box_valid_mask, is_training=is_training,
        block_name="SelfAttentionBlock")

  if use_long_term_attention:
    if use_self_attention and self_attention_in_sequence:
      input_features = tf.add(self_attention_box_features, box_features)
      input_features = tf.divide(input_features, 2)
    else:
      input_features = box_features
    original_input_features = input_features
    for jdx in range(num_attention_layers):
      layer_features = tf.zeros_like(input_features)
      for idx in range(num_attention_heads):
        block_name = "AttentionBlock" + str(idx) + "_AttentionLayer" +str(jdx)
        attention_features = attention_block(
            input_features,
            context_features,
            bottleneck_dimension,
            channels.value,
            attention_temperature,
            keys_values_valid_mask=context_valid_mask,
            queries_valid_mask=box_valid_mask,
            is_training=is_training,
            block_name=block_name)
        layer_features = tf.add(layer_features, attention_features)
      layer_features = tf.divide(layer_features, num_attention_heads)
      input_features = tf.add(input_features, layer_features)
    output_features = tf.add(input_features, original_input_features)
    if not self_attention_in_sequence and use_self_attention:
      output_features = tf.add(self_attention_box_features, output_features)
  elif use_self_attention:
    output_features = self_attention_box_features
  else:
    output_features = tf.zeros(self_attention_box_features.shape)

  # Expands the dimension back to match with the original feature map.
  output_features = output_features[:, :, tf.newaxis, tf.newaxis, :]

  return output_features
