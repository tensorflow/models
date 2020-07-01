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
  """Custom layer to perform all attention."""
  def __init__(self, bottleneck_dimension, attention_temperature,
               freeze_batchnorm, output_dimension=None, **kwargs):
    self.key_proj = ContextProjection(bottleneck_dimension, freeze_batchnorm)
    self.val_proj = ContextProjection(bottleneck_dimension, freeze_batchnorm)
    self.query_proj = ContextProjection(bottleneck_dimension, freeze_batchnorm)
    self.feature_proj = None
    self.attention_temperature = attention_temperature
    self.freeze_batchnorm = freeze_batchnorm
    self.bottleneck_dimension = bottleneck_dimension
    self.output_dimension = output_dimension
    super(AttentionBlock, self).__init__(**kwargs)

  def set_output_dimension(self, output_dim):
    self.output_dimension = output_dim

  def build(self, input_shapes):
    pass

  def call(self, input_features, is_training, valid_context_size):
    """Handles a call by performing attention"""
    print("CALLED")
    input_features, context_features = input_features
    print(input_features.shape)
    print(context_features.shape)

    _, context_size, _ = context_features.shape
    valid_mask = compute_valid_mask(valid_context_size, context_size)

    channels = input_features.shape[-1]

    #Build the feature projection layer
    if (not self.output_dimension):
      self.output_dimension = channels
    if (not self.feature_proj):
      self.feature_proj = ContextProjection(self.output_dimension,
                                            self.freeze_batchnorm)

    # Average pools over height and width dimension so that the shape of
    # box_features becomes [batch_size, max_num_proposals, channels].
    input_features = tf.reduce_mean(input_features, [2, 3])
    
    with tf.variable_scope("AttentionBlock"):
      queries = project_features(
          input_features, self.bottleneck_dimension, is_training,
          self.query_proj, normalize=True)
      keys = project_features(
          context_features, self.bottleneck_dimension, is_training,
          self.key_proj, normalize=True)
      values = project_features(
          context_features, self.bottleneck_dimension, is_training,
          self.val_proj, normalize=True)

    weights = tf.matmul(queries, keys, transpose_b=True)

    weights, values = filter_weight_value(weights, values, valid_mask)

    weights = tf.nn.softmax(weights / self.attention_temperature)

    features = tf.matmul(weights, values)
    output_features = project_features(
        features, self.output_dimension, is_training,
        self.feature_proj, normalize=False)

    output_features = output_features[:, :, tf.newaxis, tf.newaxis, :]

    print(output_features.shape)
    return output_features


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
  weights += tf.transpose(
      tf.cast(tf.math.logical_not(valid_mask), weights.dtype) *
      _NEGATIVE_PADDING_VALUE,
      perm=[0, 2, 1])

  # Force the invalid values to be 0.
  values *= tf.cast(valid_mask, values.dtype)

  return weights, values

def project_features(features, bottleneck_dimension, is_training,
                     layer, normalize=True):
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
  features = tf.reshape(features, [-1, num_features])

  projected_features = layer(features, is_training)

  projected_features = tf.reshape(projected_features,
                                  [batch_size, -1, bottleneck_dimension])

  if normalize:
    projected_features = tf.keras.backend.l2_normalize(projected_features,
                                                       axis=-1)

  return projected_features

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
