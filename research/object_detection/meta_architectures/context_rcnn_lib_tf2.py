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
"""Library functions for Context R-CNN."""
import tensorflow as tf

from object_detection.core import freezable_batch_norm

# The negative value used in padding the invalid weights.
_NEGATIVE_PADDING_VALUE = -100000


class ContextProjection(tf.keras.layers.Layer):
  """Custom layer to do batch normalization and projection."""

  def __init__(self, projection_dimension, **kwargs):
    self.batch_norm = freezable_batch_norm.FreezableBatchNorm(
        epsilon=0.001,
        center=True,
        scale=True,
        momentum=0.97,
        trainable=True)
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
               output_dimension=None, is_training=False,
               name='AttentionBlock', **kwargs):
    """Constructs an attention block.

    Args:
      bottleneck_dimension: A int32 Tensor representing the bottleneck dimension
        for intermediate projections.
      attention_temperature: A float Tensor. It controls the temperature of the
        softmax for weights calculation. The formula for calculation as follows:
          weights = exp(weights / temperature) / sum(exp(weights / temperature))
      output_dimension: A int32 Tensor representing the last dimension of the
        output feature.
      is_training: A boolean Tensor (affecting batch normalization).
      name: A string describing what to name the variables in this block.
      **kwargs: Additional keyword arguments.
    """

    self._key_proj = ContextProjection(bottleneck_dimension)
    self._val_proj = ContextProjection(bottleneck_dimension)
    self._query_proj = ContextProjection(bottleneck_dimension)
    self._feature_proj = None
    self._attention_temperature = attention_temperature
    self._bottleneck_dimension = bottleneck_dimension
    self._is_training = is_training
    self._output_dimension = output_dimension
    if self._output_dimension:
      self._feature_proj = ContextProjection(self._output_dimension)
    super(AttentionBlock, self).__init__(name=name, **kwargs)

  def build(self, input_shapes):
    """Finishes building the attention block.

    Args:
      input_shapes: the shape of the primary input box features.
    """
    if not self._feature_proj:
      self._output_dimension = input_shapes[-1]
      self._feature_proj = ContextProjection(self._output_dimension)

  def call(self, box_features, context_features, valid_context_size):
    """Handles a call by performing attention.

    Args:
      box_features: A float Tensor of shape [batch_size, input_size,
        num_input_features].
      context_features: A float Tensor of shape [batch_size, context_size,
        num_context_features].
      valid_context_size: A int32 Tensor of shape [batch_size].

    Returns:
      A float Tensor with shape [batch_size, input_size, num_input_features]
      containing output features after attention with context features.
    """

    _, context_size, _ = context_features.shape
    valid_mask = compute_valid_mask(valid_context_size, context_size)

    # Average pools over height and width dimension so that the shape of
    # box_features becomes [batch_size, max_num_proposals, channels].
    box_features = tf.reduce_mean(box_features, [2, 3])

    queries = project_features(
        box_features, self._bottleneck_dimension, self._is_training,
        self._query_proj, normalize=True)
    keys = project_features(
        context_features, self._bottleneck_dimension, self._is_training,
        self._key_proj, normalize=True)
    values = project_features(
        context_features, self._bottleneck_dimension, self._is_training,
        self._val_proj, normalize=True)

    weights = tf.matmul(queries, keys, transpose_b=True)
    weights, values = filter_weight_value(weights, values, valid_mask)
    weights = tf.nn.softmax(weights / self._attention_temperature)

    features = tf.matmul(weights, values)
    output_features = project_features(
        features, self._output_dimension, self._is_training,
        self._feature_proj, normalize=False)

    output_features = output_features[:, :, tf.newaxis, tf.newaxis, :]

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
    raise ValueError('Please make sure the first dimension of the input'
                     ' tensors are the same.')

  if w_context_size != v_context_size:
    raise ValueError('Please make sure the third dimension of weights matches'
                     ' the second dimension of values.')

  if w_context_size != m_context_size:
    raise ValueError('Please make sure the third dimension of the weights'
                     ' matches the second dimension of the valid_mask.')

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
    bottleneck_dimension: A int32 Tensor.
    is_training: A boolean Tensor (affecting batch normalization).
    layer: Contains a custom layer specific to the particular operation
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
