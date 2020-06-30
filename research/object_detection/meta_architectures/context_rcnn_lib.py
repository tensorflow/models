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

# The negative value used in padding the invalid weights.
_NEGATIVE_PADDING_VALUE = -100000

KEY_NAME = 'key'
VALUE_NAME = 'val'
QUERY_NAME = 'query'
FEATURE_NAME = 'feature'

class ContextProjection(tf.keras.layers.Layer):
  """Custom layer to do batch normalization and projection."""
  def __init__(self, projection_dimension, freeze_batchnorm, **kwargs):
    self.batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=0.001,
        center=True,
        scale=True,
        momentum=0.97,
        trainable=(not freeze_batchnorm))
    self.projection = tf.keras.layers.Dense(units=projection_dimension,
                                            activation=tf.nn.relu6,
                                            use_bias=True)
    super(ContextProjection, self).__init__(**kwargs)

  def build(self, input_shape):
    self.batch_norm.build(input_shape)
    self.projection.build(input_shape)

  def call(self, input_features, is_training=False):
    return self.projection(self.batch_norm(input_features, is_training))

class AttentionBlock(tf.keras.layers.Layer):
  def __init__(self, bottleneck_dimension, attention_temperature, freeze_batchnorm, **kwargs):
    self.key_proj = ContextProjection(bottleneck_dimension, freeze_batchnorm)
    self.val_proj = ContextProjection(bottleneck_dimension, freeze_batchnorm)
    self.query_proj = ContextProjection(bottleneck_dimension, freeze_batchnorm)
    self.attention_temperature = attention_temperature
    self.freeze_batchnorm = freeze_batchnorm
    self.bottleneck_dimension = bottleneck_dimension
    super(AttentionBlock, self).__init__(**kwargs)

  def build(self, input_shapes):
    self.feature_proj = ContextProjection(input_shapes[0][-1], self.freeze_batchnorm)
    self.key_proj.build(input_shapes[0])
    self.val_proj.build(input_shapes[0])
    self.query_proj.build(input_shapes[0])
    self.feature_proj.build(input_shapes[0])

  def filter_weight_value(self, weights, values, valid_mask):
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
    weights += tf.transpose(
        tf.cast(tf.math.logical_not(valid_mask), weights.dtype) *
        _NEGATIVE_PADDING_VALUE,
        perm=[0, 2, 1])

    # Force the invalid values to be 0.
    values *= tf.cast(valid_mask, values.dtype)

    return weights, values

  def run_projection(self, features, bottleneck_dimension, is_training, layer, normalize=True):
    """Projects features to another feature space.

    Args:
      features: A float Tensor of shape [batch_size, features_size,
        num_features].
      projection_dimension: A int32 Tensor.
      is_training: A boolean Tensor (affecting batch normalization).
      node: Contains a custom layer specific to the particular operation
            being performed (key, value, query, features)
      normalize: A boolean Tensor. If true, the output features will be l2
        normalized on the last dimension.

    Returns:
      A float Tensor of shape [batch, features_size, projection_dimension].
    """
    shape_arr = features.shape
    batch_size, _, num_features = shape_arr
    print("Orig", features.shape)
    features = tf.reshape(features, [-1, num_features])

    projected_features = layer(features, is_training)

    projected_features = tf.reshape((batch_size, -1, bottleneck_dimension))(projected_features)
    print(projected_features.shape)

    if normalize:
      projected_features = tf.keras.backend.l2_normalize(projected_features, axis=-1)

    return projected_features

  def call(self, input_features, is_training, valid_mask):
    input_features, context_features = input_features
    with tf.variable_scope("AttentionBlock"):
      queries = self.run_projection(
          input_features, self.bottleneck_dimension, is_training,
          self.query_proj, normalize=True)
      keys = self.run_projection(
          context_features, self.bottleneck_dimension, is_training,
          self.key_proj, normalize=True)
      values = self.run_projection(
          context_features, self.bottleneck_dimension, is_training,
          self.val_proj, normalize=True)

    weights = tf.matmul(queries, keys, transpose_b=True)

    weights, values = self.filter_weight_value(weights, values, valid_mask)

    weights = tf.nn.softmax(weights / self.attention_temperature)

    features = tf.matmul(weights, values)
    output_features = self.run_projection(
        features, input_features.shape[-1], is_training,
        self.feature_proj, normalize=False)
    return output_features


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

def compute_box_context_attention(box_features, context_features,
                                  valid_context_size, bottleneck_dimension,
                                  attention_temperature, is_training,
                                  freeze_batchnorm, attention_block):
  """Computes the attention feature from the context given a batch of box.

  Args:
    box_features: A float Tensor of shape [batch_size, max_num_proposals,
      height, width, channels]. It is pooled features from first stage
      proposals.
    context_features: A float Tensor of shape [batch_size, context_size,
      num_context_features].
    valid_context_size: A int32 Tensor of shape [batch_size].
    bottleneck_dimension: A int32 Tensor representing the bottleneck dimension
      for intermediate projections.
    attention_temperature: A float Tensor. It controls the temperature of the
      softmax for weights calculation. The formula for calculation as follows:
        weights = exp(weights / temperature) / sum(exp(weights / temperature))
    is_training: A boolean Tensor (affecting batch normalization).
    freeze_batchnorm: Whether to freeze batch normalization weights.
    attention_projections: Dictionary of the projection layers.

  Returns:
    A float Tensor of shape [batch_size, max_num_proposals, 1, 1, channels].
  """
  _, context_size, _ = context_features.shape
  valid_mask = compute_valid_mask(valid_context_size, context_size)

  channels = box_features.shape[-1]
  
  # Average pools over height and width dimension so that the shape of
  # box_features becomes [batch_size, max_num_proposals, channels].
  box_features = tf.reduce_mean(box_features, [2, 3])

  output_features = attention_block([box_features, context_features], is_training, valid_mask)

  # Expands the dimension back to match with the original feature map.
  output_features = output_features[:, :, tf.newaxis, tf.newaxis, :]

  return output_features
